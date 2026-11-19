"""Monta videos/prueba-actividades.mp4: 3 s de cada clip, con su actividad conocida (clips/prueba-actividades.json)."""
import cv2, json, subprocess, numpy as np, glob
GYM = glob.glob("videos/*NAKsQjS1Gyg*.mp4")[0]
# (archivo, inicio s, actividad real, nota, zonas de texto a difuminar en px del original [x0,y0,x1,y1])
CLIPS = [
    ("clips/raw/mk_40881.mp4", 2.0, "running", "dos personas corriendo", []),
    ("clips/raw/yt_hPrv-0QDnSc.mp4", 0.3, "drinking", "tomando café", [[80, 285, 290, 355]]),
    ("clips/raw/mk_2002.mp4", 0.5, "cycling", "ciclista pequeño y lejano en un parque", []),
    ("clips/raw/yt_ZmM36viuNus.mp4", 3.0, "dancing", "chica bailando", []),
    ("clips/raw/mk_4720.mp4", 1.0, "sitting", "niño sentado en un banco", []),
    ("clips/raw/yt_FyyODwkJZJc.mp4", 5.0, "horse", "gente a caballo en la playa", []),
    ("clips/raw/mk_4823.mp4", 5.8, "walking", "chica andando por un parque", []),
    ("clips/raw/yt_Wbo0IY1tSRo.mp4", 2.0, "cycling", "ciclista en carretera visto desde un dron", []),
    (GYM, 15.0, "exercise", "chica entrenando con kettlebell", []),
    ("clips/raw/yt_zUCrIeXEWLc.mp4", 1.0, "sitting", "mujer sentada en un sofá", [[120, 45, 250, 90], [5, 460, 160, 520], [225, 155, 360, 210]]),
    ("clips/raw/yt_9dUW9YCAJ7M.mp4", 4.5, "running", "carrera de atletismo (velocistas)", []),
]
W, H, FPS, SEG = 1280, 720, 25, 3.0
out = "clips/_tmp.mp4"
vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
gt, t0 = [], 0.0
for path, start, act, note, blurs in CLIPS:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    for i in range(int(SEG * FPS)):
        cap.set(cv2.CAP_PROP_POS_MSEC, (start + i / FPS) * 1000)
        ok, f = cap.read()
        if not ok:
            break
        for x0, y0, x1, y1 in blurs:
            f[y0:y1, x0:x1] = cv2.GaussianBlur(f[y0:y1, x0:x1], (51, 51), 0)
        s = min(W / f.shape[1], H / f.shape[0])
        f = cv2.resize(f, (int(f.shape[1] * s), int(f.shape[0] * s)))
        canvas = np.zeros((H, W, 3), np.uint8)
        y, x = (H - f.shape[0]) // 2, (W - f.shape[1]) // 2
        canvas[y:y + f.shape[0], x:x + f.shape[1]] = f
        vw.write(canvas)
    gt.append({"start": round(t0, 2), "end": round(t0 + SEG, 2), "activity": act, "note": note, "source": path.split("/")[-1]})
    t0 += SEG
vw.release()
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", out, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "videos/prueba-actividades.mp4"], check=True)
subprocess.run(["rm", out])
json.dump(gt, open("clips/prueba-actividades.json", "w"), ensure_ascii=False, indent=1)
print(f"{len(gt)} clips, {t0:.0f} s")
