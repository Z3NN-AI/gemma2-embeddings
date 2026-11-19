"""Modo cámara en directo (WebSocket /ws/live).

El navegador manda fotogramas JPEG de la webcam. Por cada fotograma:
  - escena:    YOLO detecta y sigue objetos  → cajas al momento
  - emociones: detector de caras de OpenCV    → cajas al momento
En paralelo, un bucle llama a Gemma cada ~0,7 s con las personas/caras principales
(clip de 2 s por persona, o el recorte de la cara) y guarda sus similitudes por id.
Cada respuesta lleva las cajas actuales y la última clasificación conocida de cada id.
"""
import asyncio
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from PIL import Image

from analyze import SCENE_CLASSES, yolo_weights

ROOT = Path(__file__).parent
GEMMA_EVERY = 0.7     # s entre rondas de Gemma
CLIP_SEC = 2.0        # duración del clip de cada persona
MAX_TARGETS = 3       # personas/caras clasificadas por ronda
TOKENS = 70           # tokens por fotograma para Gemma (rápido, misma precisión en la prueba)


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


class LiveSession:
    def __init__(self, analyzer, mode):
        self.A = analyzer
        self.mode = mode
        self.hist = {}        # id -> deque[(t, box, frame)]
        self.sims = {}        # id -> (t, sims)
        self.objects = []     # última detección
        self.next_face_id = 1
        self.gemma_ms = None
        self.yolo = None
        if mode == "scene":
            from ultralytics import YOLO
            self.yolo = YOLO(yolo_weights())
            self.keep = [i for i, n in self.yolo.names.items() if n in SCENE_CLASSES]

    # ---------- detección por fotograma ----------
    def detect(self, frame, t):
        H, W = frame.shape[:2]
        objs = []
        if self.mode == "scene":
            r = self.yolo.track(frame, persist=True, classes=self.keep, conf=0.35, verbose=False,
                                tracker="bytetrack.yaml", device=self.A.device)[0]
            if r.boxes is not None and r.boxes.id is not None:
                for b, c, tid, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                                         r.boxes.id.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    objs.append({"id": int(tid), "cls": self.yolo.names[int(c)], "conf": round(float(cf), 2),
                                 "box": [int(v) for v in b]})
        else:
            prev = [o for o in self.objects]
            for f in self.A.detect_faces(frame)[:MAX_TARGETS * 2]:
                box = self.A.expand(f, W, H, 0.15)
                # id estable: la cara anterior que más se solape
                best = max(prev, key=lambda o: _iou(o["box"], box), default=None)
                if best is not None and _iou(best["box"], box) > 0.3:
                    fid = best["id"]; prev.remove(best)
                else:
                    fid = self.next_face_id; self.next_face_id += 1
                objs.append({"id": fid, "cls": "face", "conf": 1.0, "box": box})
        for o in objs:
            if o["cls"] in ("person", "face"):
                h = self.hist.setdefault(o["id"], deque())
                h.append((t, o["box"], frame))
                while h and h[0][0] < t - CLIP_SEC:
                    h.popleft()
        alive = {o["id"] for o in objs}
        for k in [k for k in self.hist if k not in alive and self.hist[k][-1][0] < t - 3]:
            del self.hist[k]
        self.objects = objs
        return objs

    # ---------- Gemma ----------
    def gemma_round(self):
        """Clasifica las personas/caras más grandes con su historia reciente. Bloqueante (hilo aparte)."""
        targets = sorted([o for o in self.objects if o["id"] in self.hist],
                         key=lambda o: -(o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]))[:MAX_TARGETS]
        if not targets:
            return
        inputs, ids = [], []
        for o in targets:
            h = list(self.hist[o["id"]])
            if self.mode == "scene":
                pick = [h[int(round(j * (len(h) - 1) / 3))] for j in range(4)] if len(h) > 1 else h
                x0 = min(p[1][0] for p in pick); y0 = min(p[1][1] for p in pick)
                x1 = max(p[1][2] for p in pick); y1 = max(p[1][3] for p in pick)
                H, W = pick[0][2].shape[:2]
                bw, bh = x1 - x0, y1 - y0
                x0, y0 = max(0, int(x0 - .2 * bw)), max(0, int(y0 - .1 * bh))
                x1, y1 = min(W, int(x1 + .2 * bw)), min(H, int(y1 + .25 * bh))
                clip = [Image.fromarray(cv2.cvtColor(p[2][y0:y1, x0:x1], cv2.COLOR_BGR2RGB)) for p in pick]
                inputs.append({"video": clip})
            else:
                _, (x0, y0, x1, y1), fr = h[-1]
                inputs.append(Image.fromarray(cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)))
            ids.append(o["id"])
        t0 = time.time()
        labels = self.A.act_emb if self.mode == "scene" else self.A.text_emb
        sims = self.A.encode(inputs, labels, tokens=TOKENS, bs=len(inputs))
        self.gemma_ms = round((time.time() - t0) * 1000)
        now = time.time()
        for i, s in zip(ids, sims):
            self.sims[i] = (now, s)


async def live_endpoint(ws: WebSocket, get_analyzer):
    await ws.accept()
    mode = ws.query_params.get("mode", "scene")
    loop = asyncio.get_running_loop()
    await ws.send_text(json.dumps({"status": "loading"}))
    analyzer = await loop.run_in_executor(None, get_analyzer)
    sess = await loop.run_in_executor(None, LiveSession, analyzer, mode)
    await ws.send_text(json.dumps({"status": "ready"}))
    running = True

    async def gemma_loop():
        while running:
            start = time.time()
            try:
                await loop.run_in_executor(None, sess.gemma_round)
            except Exception as e:  # noqa: BLE001
                print("gemma live:", e)
            await asyncio.sleep(max(0.05, GEMMA_EVERY - (time.time() - start)))

    task = asyncio.create_task(gemma_loop())
    try:
        while True:
            data = await ws.receive_bytes()
            t0 = time.time()
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            objs = await loop.run_in_executor(None, sess.detect, frame, t0)
            now = time.time()
            out = []
            for o in objs:
                s = sess.sims.get(o["id"])
                out.append({**o, "sims": s[1] if s else None, "age": round(now - s[0], 1) if s else None})
            H, W = frame.shape[:2]
            await ws.send_text(json.dumps({"w": W, "h": H, "objects": out,
                                           "det_ms": round((now - t0) * 1000), "gemma_ms": sess.gemma_ms}))
    except WebSocketDisconnect:
        pass
    finally:
        running = False
        task.cancel()
