"""Servidor local: biblioteca de videos/imágenes, descarga por URL y análisis con EmbeddingGemma 2.

Arranque:  .venv/bin/python server.py   →   http://127.0.0.1:8765
"""
import json
import queue
import re
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from analyze import ACT_LABELS, LABELS, Analyzer, media_type
from live import live_endpoint

ROOT = Path(__file__).parent
MEDIA = ROOT / "videos"
RESULTS = MEDIA / ".results"
MEDIA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)

app = FastAPI()
jobs: dict[str, dict] = {}
work = queue.Queue()
_analyzer = None


def result_path(name, mode="emotion"):
    return RESULTS / (name + (".scene" if mode == "scene" else "") + ".json")


def safe_name(name):
    name = Path(name).name
    name = re.sub(r"[^\w.\- ]+", "_", name).strip() or "archivo"
    return name


def lookup(name):
    """Nombre de un archivo existente: sin rutas (evita salir de videos/)."""
    name = Path(name).name
    if not name or name.startswith(".") or not (MEDIA / name).is_file():
        raise HTTPException(404)
    return name


def unique(name):
    p = MEDIA / name
    i = 1
    while p.exists():
        p = MEDIA / f"{Path(name).stem}-{i}{Path(name).suffix}"
        i += 1
    return p


# ---------------- trabajos (un solo hilo: el modelo usa la GPU) ----------------
def new_job(kind, label):
    jid = uuid.uuid4().hex[:8]
    jobs[jid] = {"id": jid, "mode": None, "kind": kind, "label": label, "stage": "queued", "progress": 0.0,
                 "error": None, "name": None, "created": time.time()}
    return jobs[jid]


def run_download(job, url, fps, mode):
    job["stage"] = "download"
    tmpl = str(MEDIA / "%(title).60B [%(id)s].%(ext)s")
    cmd = ["uvx", "yt-dlp@latest", "--no-playlist", "--newline", "--restrict-filenames",
           "-f", "bv*[height<=720][ext=mp4][vcodec^=avc1]+ba[ext=m4a]/b[height<=720][ext=mp4]/bv*[height<=720]+ba/b",
           "--merge-output-format", "mp4", "--print", "after_move:filepath", "-o", tmpl, url]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    final, tail = None, []
    for line in proc.stdout:
        line = line.strip()
        tail = (tail + [line])[-8:]
        m = re.search(r"\[download\]\s+([\d.]+)%", line)
        if m:
            job["progress"] = float(m.group(1)) / 100
        elif line.startswith(str(MEDIA)):
            final = line
    if proc.wait() != 0 or not final:
        raise RuntimeError("yt-dlp: " + " | ".join(l for l in tail if l)[-400:])
    job["name"] = Path(final).name
    run_analyze(job, job["name"], fps, mode)


_analyzer_lock = threading.Lock()


def get_analyzer():
    """Un único Gemma compartido por los análisis y la cámara."""
    global _analyzer
    with _analyzer_lock:
        if _analyzer is None:
            _analyzer = Analyzer()
    return _analyzer


def run_analyze(job, name, fps, mode):
    job["stage"] = "loading"
    job["name"] = name
    analyzer = get_analyzer()
    job["stage"] = "analyze"
    job["progress"] = 0.0
    if mode == "scene" and media_type(name) != "video":
        mode = "emotion"  # el modo escena es sólo para video
    job["mode"] = mode
    res = analyzer.analyze(MEDIA / name, fps=fps, mode=mode, progress=lambda p: job.__setitem__("progress", p))
    res["name"] = name
    result_path(name, mode).write_text(json.dumps(res))
    job["stage"] = "done"
    job["progress"] = 1.0


def worker():
    while True:
        job, fn, args = work.get()
        try:
            fn(job, *args)
        except Exception as e:  # noqa: BLE001
            job["stage"] = "error"
            job["error"] = str(e)[-500:]


threading.Thread(target=worker, daemon=True).start()


# ---------------- API ----------------
class DownloadReq(BaseModel):
    url: str
    fps: float = 4.0
    mode: str = "emotion"


class AnalyzeReq(BaseModel):
    name: str
    fps: float = 4.0
    mode: str = "emotion"


@app.get("/api/labels")
def labels():
    return {"emotion": LABELS, "scene": ACT_LABELS}


@app.get("/api/library")
def library():
    items = []
    for p in sorted(MEDIA.iterdir(), key=lambda p: -p.stat().st_mtime):
        kind = media_type(p)
        if p.is_file() and kind:
            items.append({"name": p.name, "type": kind, "size": p.stat().st_size,
                          "analyzed": result_path(p.name).exists(),
                          "scene": result_path(p.name, "scene").exists()})
    return items


@app.get("/api/result/{name}")
def result(name: str, mode: str = "emotion"):
    p = result_path(Path(name).name, mode)
    if not p.exists():
        raise HTTPException(404)
    return json.loads(p.read_text())


@app.get("/api/calibration")
def calibration():
    """Similitud media por emoción sobre todas las caras analizadas de la biblioteca."""
    rows = []
    for p in RESULTS.glob("*.json"):
        if p.name.endswith(".scene.json"):
            continue
        r = json.loads(p.read_text())
        items = r.get("samples") or r.get("faces") or []
        rows += [s["sims"] for s in items if s.get("sims")]
    return {"count": len(rows), "mean": np.mean(rows, axis=0).tolist() if rows else None}


@app.post("/api/download")
def download(req: DownloadReq):
    if not re.match(r"^https?://", req.url.strip()):
        raise HTTPException(400, "URL no válida")
    job = new_job("download", req.url.strip())
    work.put((job, run_download, (req.url.strip(), req.fps, req.mode)))
    return job


@app.post("/api/analyze")
def analyze(req: AnalyzeReq):
    name = lookup(req.name)
    job = new_job("analyze", name)
    work.put((job, run_analyze, (name, req.fps, req.mode)))
    return job


@app.post("/api/upload")
async def upload(file: UploadFile, fps: float = 4.0, mode: str = "emotion"):
    name = safe_name(file.filename or "archivo")
    if not media_type(name):
        raise HTTPException(400, "Formato no soportado")
    dest = unique(name)
    dest.write_bytes(await file.read())
    job = new_job("analyze", dest.name)
    work.put((job, run_analyze, (dest.name, fps, mode)))
    return job


@app.delete("/api/media/{name}")
def delete(name: str):
    name = lookup(name)
    for p in (MEDIA / name, result_path(name), result_path(name, "scene")):
        p.unlink(missing_ok=True)
    return {"ok": True}


@app.get("/api/jobs")
def list_jobs():
    return sorted(jobs.values(), key=lambda j: -j["created"])[:20]


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await live_endpoint(ws, get_analyzer)


app.mount("/media", StaticFiles(directory=MEDIA), name="media")


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    print("→ http://127.0.0.1:8765")
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")
