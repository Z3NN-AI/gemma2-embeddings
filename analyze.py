"""Detección de caras (OpenCV) + clasificación de emociones con EmbeddingGemma 2 (zero-shot).

EmbeddingGemma 2 es un modelo de *embeddings*: no genera texto ni cajas.
- Las cajas las pone un detector de caras de OpenCV.
- La emoción sale de comparar el embedding del recorte de la cara con
  embeddings de descripciones de texto de cada emoción.

Uso CLI:  .venv/bin/python analyze.py videos/archivo.mp4 [--fps 4]
"""
import argparse
import contextlib
import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

MODEL_ID = "google/embeddinggemma-2"


def _use_offline_if_cached():
    """Si el modelo ya está descargado, no consultar Hugging Face en cada arranque."""
    from huggingface_hub.constants import HF_HUB_CACHE
    if (Path(HF_HUB_CACHE) / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots").exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


_use_offline_if_cached()

# clave -> (emoji, nombre ES, nombre EN, prompts)
EMOTIONS = {
    "happy":    ("😄", "feliz",    "happy",     ["a smiling happy face", "a face with a joyful expression", "a person laughing", "a face showing happiness"]),
    "sad":      ("😢", "triste",   "sad",       ["a sad face", "a face with a sorrowful expression", "a person looking unhappy and down", "a face showing sadness"]),
    "angry":    ("😠", "enojado",  "angry",     ["an angry face", "a face with a furious frowning expression", "a person looking mad", "a face showing anger"]),
    "surprise": ("😮", "sorpresa", "surprise",  ["a surprised face", "a face with an astonished expression and open mouth", "a person looking shocked", "a face showing surprise"]),
    "fear":     ("😨", "miedo",    "fear",      ["a scared face", "a face with a fearful expression", "a person looking afraid and worried", "a face showing fear"]),
    "disgust":  ("🤢", "asco",     "disgust",   ["a disgusted face", "a face with a grossed-out expression", "a person wrinkling their nose in disgust", "a face showing disgust"]),
    "neutral":  ("😐", "neutral",  "neutral",   ["a neutral face", "a face with a blank calm expression", "a person with no emotion", "an expressionless face"]),
}
LABELS = [{"key": k, "emoji": v[0], "es": v[1], "en": v[2]} for k, v in EMOTIONS.items()]

# Actividades de personas (modo escena)
ACTIVITIES = {
    "standing":  ("🧍", "de pie",       "standing",      ["a person standing still", "a person standing and talking"]),
    "walking":   ("🚶", "andando",      "walking",       ["a person walking", "a pedestrian walking down the street"]),
    "running":   ("🏃", "corriendo",    "running",       ["a person running", "a person jogging"]),
    "cycling":   ("🚴", "en bici",      "cycling",       ["a person riding a bicycle", "a cyclist pedaling a bike"]),
    "sitting":   ("🪑", "sentado",      "sitting",       ["a person sitting down", "a person sitting on a bench or chair"]),
    "horse":     ("🏇", "a caballo",    "horse riding",  ["a person riding a horse", "a rider on horseback"]),
    "skating":   ("🛹", "patinando",    "skating",       ["a person skateboarding", "a person roller skating"]),
    "dancing":   ("💃", "bailando",     "dancing",       ["a person dancing", "people dancing to music"]),
    "exercise":  ("🏋️", "ejercicio",    "working out",   ["a person doing a workout exercise", "a person lifting weights at the gym"]),
    "ball":      ("⚽", "jugando pelota", "playing ball", ["a person playing football", "a person playing with a ball"]),
    "driving":   ("🛵", "en moto",      "motorbike",     ["a person riding a motorcycle or scooter"]),
    "drinking":  ("☕", "bebiendo",     "drinking",      ["a person drinking from a cup", "a person drinking coffee"]),
}
ACT_LABELS = [{"key": k, "emoji": v[0], "es": v[1], "en": v[2]} for k, v in ACTIVITIES.items()]

# Nombres en español de las clases COCO más habituales (el resto, en inglés)
COCO_ES = {"person": "persona", "bicycle": "bici", "car": "coche", "motorcycle": "moto", "bus": "autobús",
           "truck": "camión", "train": "tren", "boat": "barco", "horse": "caballo", "dog": "perro", "cat": "gato",
           "bird": "pájaro", "cow": "vaca", "sheep": "oveja", "airplane": "avión", "skateboard": "monopatín",
           "sports ball": "pelota", "bench": "banco", "chair": "silla", "umbrella": "paraguas", "backpack": "mochila",
           "handbag": "bolso", "traffic light": "semáforo", "kite": "cometa", "surfboard": "tabla de surf",
           "skis": "esquís", "frisbee": "frisbi", "tennis racket": "raqueta"}
COCO_EMOJI = {"person": "🧑", "bicycle": "🚲", "car": "🚗", "motorcycle": "🏍️", "bus": "🚌", "truck": "🚚",
              "train": "🚆", "boat": "⛵", "horse": "🐎", "dog": "🐕", "cat": "🐈", "bird": "🐦", "cow": "🐄",
              "sheep": "🐑", "airplane": "✈️", "skateboard": "🛹", "sports ball": "⚽", "bench": "🪑", "chair": "🪑"}
# objetos que se dibujan en el modo escena (lo demás se ignora para no saturar)
SCENE_CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck", "train", "boat", "horse", "dog", "cat",
                 "bird", "cow", "sheep", "airplane", "skateboard", "sports ball", "surfboard", "skis", "kite", "frisbee"]

VIDEO_EXT = {".mp4", ".webm", ".mov", ".m4v", ".mkv"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


def yolo_weights():
    """Pesos de YOLO: models/yolov8m-seg.pt si existe; si no, ultralytics los descarga solo."""
    local = Path(__file__).parent / "models" / "yolov8m-seg.pt"
    return str(local) if local.exists() else "yolov8m-seg.pt"


def media_type(path):
    ext = Path(path).suffix.lower()
    return "video" if ext in VIDEO_EXT else "image" if ext in IMAGE_EXT else None


def _iou(a, b):
    ax0, ay0, aw, ah = a; bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


class Analyzer:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "mps" else torch.float32  # nunca float16
        self.device = device
        self.model = SentenceTransformer(MODEL_ID, device=device,
                                         model_kwargs={"torch_dtype": dtype},
                                         config_kwargs={"audio_config": None})
        self.lock = threading.Lock()   # Gemma se usa desde varios hilos (análisis por lotes y cámara)
        self.text_emb = self._label_emb(EMOTIONS)
        self.act_emb = self._label_emb(ACTIVITIES)
        self.yolo = None
        self.cascades = [cv2.CascadeClassifier(cv2.data.haarcascades + f)
                         for f in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml")]

    @contextlib.contextmanager
    def vision_tokens(self, n):
        """Tokens por imagen/fotograma (70–1120). Menos = más rápido; 70 rinde igual en actividades."""
        proc = self.model[0].processor
        old = proc.image_processor.max_soft_tokens, proc.video_processor.max_soft_tokens
        proc.image_processor.max_soft_tokens = proc.video_processor.max_soft_tokens = n
        try:
            yield
        finally:
            proc.image_processor.max_soft_tokens, proc.video_processor.max_soft_tokens = old

    def encode(self, inputs, labels, tokens=None, bs=8):
        """Similitudes (lista de listas) de imágenes/clips contra las etiquetas. Seguro entre hilos."""
        with self.lock, (self.vision_tokens(tokens) if tokens else contextlib.nullcontext()):
            e = self.model.encode(inputs, normalize_embeddings=True, show_progress_bar=False, batch_size=bs)
        return [[round(float(v), 5) for v in row] for row in e @ labels.T]

    def _label_emb(self, table):
        embs = []
        for k in table:
            e = self.model.encode(table[k][3], prompt_name="Classification", normalize_embeddings=True)
            v = e.mean(0)
            embs.append(v / np.linalg.norm(v))
        return np.stack(embs)

    # ---------- caras ----------
    def detect_faces(self, bgr):
        gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        ms = max(30, int(0.05 * min(gray.shape)))
        found = []
        for c in self.cascades:
            for f in c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(ms, ms)):
                f = tuple(int(x) for x in f)
                if all(_iou(f, g) < 0.3 for g in found):
                    found.append(f)
        return sorted(found, key=lambda f: -f[2] * f[3])

    @staticmethod
    def expand(f, W, H, m):
        x, y, w, h = f
        return [int(max(0, x - w * m)), int(max(0, y - h * m)), int(min(W, x + w * (1 + m))), int(min(H, y + h * (1 + m)))]

    def embed_sims(self, crops, progress=None, labels=None, bs=8):
        labels = self.text_emb if labels is None else labels
        out = []
        for i in range(0, len(crops), bs):
            with self.lock:
                e = self.model.encode(crops[i:i + bs], normalize_embeddings=True, show_progress_bar=False)
            out.append(e @ labels.T)
            if progress:
                progress(min(1.0, (i + bs) / len(crops)))
        sims = np.concatenate(out) if out else np.zeros((0, len(EMOTIONS)))
        return [[round(float(v), 5) for v in row] for row in sims]

    # ---------- imagen ----------
    def analyze_image(self, path, margin=0.15, progress=None):
        bgr = cv2.imread(str(path))
        if bgr is None:  # gif u otros formatos: vía PIL
            bgr = cv2.cvtColor(np.array(Image.open(path).convert("RGB")), cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]
        faces = self.detect_faces(bgr)
        boxes = [self.expand(f, W, H, margin) for f in faces]
        detected = True
        if not boxes:  # sin cara detectada: usar la imagen completa
            boxes, detected = [[0, 0, W, H]], False
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crops = [Image.fromarray(rgb[y0:y1, x0:x1]) for x0, y0, x1, y1 in boxes]
        sims = self.embed_sims(crops, progress)
        return {"type": "image", "width": W, "height": H,
                "faces": [{"box": b, "detected": detected, "sims": s} for b, s in zip(boxes, sims)]}

    # ---------- video ----------
    def analyze_video(self, path, fps=4.0, margin=0.15, progress=None):
        cap = cv2.VideoCapture(str(path))
        vfps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        step = max(1, round(vfps / fps))
        samples, crops, box_s, last_det = [], [], None, -1e9
        idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                t = idx / vfps
                faces = self.detect_faces(frame)
                detected = bool(faces)
                if detected:
                    b = np.array(self.expand(faces[0], W, H, margin), dtype=float)
                    box_s = b if box_s is None else 0.6 * box_s + 0.4 * b  # suavizado
                    last_det = t
                if box_s is None or t - last_det > 2.0:  # sin cara reciente: no clasificar
                    samples.append({"t": round(t, 3), "box": None, "detected": False, "sims": None})
                else:
                    x0, y0, x1, y1 = (int(v) for v in box_s)
                    crops.append(Image.fromarray(cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)))
                    samples.append({"t": round(t, 3), "box": [x0, y0, x1, y1], "detected": detected, "sims": None})
                if progress and total:
                    progress(0.3 * idx / total)  # 30 %: detección
            idx += 1
        cap.release()
        sims = iter(self.embed_sims(crops, (lambda p: progress(0.3 + 0.7 * p)) if progress else None))
        for s in samples:
            if s["box"]:
                s["sims"] = next(sims)
        return {"type": "video", "width": W, "height": H, "fps": fps, "samples": samples}

    # ---------- escena: YOLO (cajas + seguimiento) + Gemma (actividad de cada persona) ----------
    def analyze_scene(self, path, fps=2.0, clip_sec=2.0, clip_frames=4, max_people=8, progress=None):
        from ultralytics import YOLO
        if self.yolo is None:
            self.yolo = YOLO(yolo_weights())
        names = self.yolo.names
        keep = [i for i, n in names.items() if n in SCENE_CLASSES]
        cap = cv2.VideoCapture(str(path))
        vfps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        track_step = max(1, round(vfps / 8))        # seguimiento a ~8 fps
        sample_every = max(1, round(8 / fps))       # muestra para Gemma cada N pasos de seguimiento
        hist = {}                                    # id -> [(t, box, frame_idx_en_buffer)]
        frames = {}                                  # frame_idx -> frame (sólo los recientes)
        samples, jobs = [], []                       # jobs: (sample_idx, obj_idx, [PIL frames])
        idx = step_i = 0
        prev_hist, id_offset, max_id, cuts = None, 0, 0, []
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % track_step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                t = idx / vfps
                # corte de plano: cambio brusco de colores -> reiniciar seguimiento (ids nuevos, clips sin mezclar)
                # (se ignoran las franjas negras: en videos verticales dominarían la comparación)
                small = cv2.resize(frame, (160, 90))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                mask = (gray > 12).astype(np.uint8)
                h = cv2.normalize(cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], mask,
                                               [30, 32], [0, 180, 0, 256]), None).flatten()
                cut = False
                if prev_hist is not None:
                    pgray, pmask, phist = prev_hist
                    union = (mask | pmask) > 0
                    diff = np.abs(gray.astype(float) - pgray)[union].mean() if union.any() else 0
                    cut = diff > 45 or cv2.compareHist(phist, h, cv2.HISTCMP_CORREL) < 0.6
                if cut:
                    self.yolo = YOLO(yolo_weights())
                    id_offset, hist = max_id, {}
                    cuts.append(round(t, 3))
                prev_hist = (gray.astype(float), mask, h)
                r = self.yolo.track(frame, persist=True, classes=keep, conf=0.35, verbose=False,
                                    tracker="bytetrack.yaml", device=self.device)[0]
                frames[idx] = frame
                for old in [k for k in frames if k < idx - (clip_sec + 0.5) * vfps]:
                    del frames[old]
                objs = []
                if r.boxes is not None and r.boxes.id is not None:
                    for b, c, tid, cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy(),
                                             r.boxes.id.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                        tid = int(tid) + id_offset
                        max_id = max(max_id, tid)
                        box = [int(v) for v in b]
                        hist.setdefault(tid, []).append((t, box, idx))
                        hist[tid] = [h for h in hist[tid] if h[0] >= t - clip_sec]
                        objs.append({"id": tid, "cls": names[int(c)], "conf": round(float(cf), 2), "box": box, "sims": None})
                if step_i % sample_every == 0:
                    people = sorted([o for o in objs if o["cls"] == "person"],
                                    key=lambda o: -(o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1]))[:max_people]
                    for o in people:
                        h = hist[o["id"]]
                        pick = [h[int(round(j * (len(h) - 1) / max(1, clip_frames - 1)))] for j in range(clip_frames)]
                        # región = unión de las cajas del clip, ampliada (para incluir bici/caballo/suelo)
                        xs0 = min(p[1][0] for p in pick); ys0 = min(p[1][1] for p in pick)
                        xs1 = max(p[1][2] for p in pick); ys1 = max(p[1][3] for p in pick)
                        bw, bh = xs1 - xs0, ys1 - ys0
                        x0, y0 = max(0, int(xs0 - .2 * bw)), max(0, int(ys0 - .1 * bh))
                        x1, y1 = min(W, int(xs1 + .2 * bw)), min(H, int(ys1 + .25 * bh))
                        clip = [Image.fromarray(cv2.cvtColor(frames[p[2]][y0:y1, x0:x1], cv2.COLOR_BGR2RGB))
                                for p in pick if p[2] in frames]
                        jobs.append((len(samples), objs.index(o), clip))
                samples.append({"t": round(t, 3), "objects": objs})  # cajas a ~8 fps; actividad sólo en algunas
                step_i += 1
                if progress and total:
                    progress(0.4 * idx / total)
            idx += 1
        cap.release()
        self.yolo = None  # se recarga en el próximo video: seguimiento limpio
        with self.vision_tokens(70):  # 2x más rápido, misma precisión en la prueba de 11 clips
            sims = self.embed_sims([{"video": c} for _, _, c in jobs], labels=self.act_emb, bs=4,
                                   progress=(lambda p: progress(0.4 + 0.6 * p)) if progress else None)
        for (si, oi, _), sv in zip(jobs, sims):
            samples[si]["objects"][oi]["sims"] = sv
        classes = sorted({o["cls"] for s in samples for o in s["objects"]})
        return {"type": "scene", "width": W, "height": H, "fps": fps, "samples": samples, "cuts": cuts,
                "classes": [{"key": c, "es": COCO_ES.get(c, c), "en": c, "emoji": COCO_EMOJI.get(c, "📦")} for c in classes]}

    def analyze(self, path, fps=4.0, progress=None, mode="emotion"):
        kind = media_type(path)
        if kind == "video" and mode == "scene":
            return self.analyze_scene(path, fps=min(fps, 4), progress=progress)
        if kind == "video":
            return self.analyze_video(path, fps=fps, progress=progress)
        if kind == "image":
            return self.analyze_image(path, progress=progress)
        raise ValueError(f"formato no soportado: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--mode", choices=["emotion", "scene"], default="emotion")
    args = ap.parse_args()
    res = Analyzer().analyze(args.path, fps=args.fps, mode=args.mode, progress=lambda p: print(f"\r{p:.0%}", end="", flush=True))
    out = Path(args.path).parent / ".results" / (Path(args.path).name + (".scene" if args.mode == "scene" else "") + ".json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res))
    print("\nescrito", out)
