# Gemma: emociones y escenas

Experimento para ver qué puede hacer **[EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)**, un modelo de *embeddings* multimodal (texto, imagen, video y audio) de 740M parámetros, más allá de la búsqueda semántica:

- 😄 **Emociones**: pone un emoji sobre cada cara según su expresión.
- 🎬 **Escena**: pone una caja a cada persona, bici, coche, caballo… y dice qué está haciendo cada persona (andando, corriendo, en bici, sentado, bailando, bebiendo…).
- 📷 **En directo** con la webcam, en los dos modos.

Todo funciona en local. La interfaz está en español e inglés y el idioma se cambia con un clic.

*English: local web app that uses EmbeddingGemma 2 zero-shot to label facial emotions and per-person activities in videos, images and a live webcam feed. OpenCV/YOLO draw the boxes; Gemma decides the label by comparing image/clip embeddings with text embeddings. UI in Spanish and English.*

## Cómo funciona

EmbeddingGemma 2 **no genera texto ni sabe dónde están las cosas**: sólo convierte imagen, video o texto en un vector de 768 números. Por eso el trabajo se reparte:

| pieza | qué hace |
|---|---|
| detector de caras de OpenCV / **YOLOv8** | encuentra y sigue caras, personas y objetos (pone las cajas) |
| **EmbeddingGemma 2** | convierte el recorte (o un clip de 2 s de cada persona) en un vector y lo compara con vectores de frases como *"a person riding a bicycle"* |

La etiqueta con el vector más parecido es la respuesta. El modelo no se entrena: es clasificación *zero-shot*. Para añadir una emoción o actividad basta con escribir su descripción en `analyze.py` (`EMOTIONS`, `ACTIVITIES`).

## Lo que aprendimos probándolo

- **Lee el texto que aparece en la imagen.** En un video de ejercicios con el nombre escrito arriba acertaba el 100% sólo con la franja del título, y al taparlo bajaba al ~30%. Por eso a Gemma sólo le llega el recorte de la cara o de la persona, no el fotograma entero.
- **No distingue ejercicios concretos** (sentadilla con pase contra zancada con press, por ejemplo), pero sí **actividades generales**. En un video de prueba de 11 clips de 3 s sin texto acertó **11/11**: correr, beber, bici, bailar, sentado, a caballo, andar, ejercicio…
- **Cortes de plano**: hay que reiniciar el seguimiento en cada corte; si no, el clip de una persona mezcla fotogramas de dos escenas. Se detectan automáticamente.
- **70 tokens por fotograma** en lugar de 140 rinde igual en actividades y va el doble de rápido.

## Velocidad (Mac mini, Apple Silicon, GPU MPS)

| | |
|---|---|
| YOLO | ~36 ms/fotograma |
| Gemma, clip de 2 s (4 fotogramas, 70 tokens) | ~0,5 s |
| Gemma, recorte de cara (70 tokens) | ~0,13 s |
| analizar 33 s de video en modo escena | ~49 s |
| en directo | emociones ~12 fps, escena ~5–6 fps; Gemma actualiza cada ~0,7 s |

No es un modelo pensado para tiempo real, pero en directo da una respuesta útil con 1–2 s de retraso.

## Instalación

Requiere Python 3.12 y `ffmpeg`. Para descargar desde YouTube se usa [uv](https://docs.astral.sh/uv/) (`uvx yt-dlp@latest`).

```bash
uv venv -p 3.12 .venv
uv pip install -p .venv -r requirements.txt
.venv/bin/python server.py          # → http://127.0.0.1:8765
```

La primera vez se descargan EmbeddingGemma 2 desde Hugging Face y los pesos de YOLOv8 (`yolov8m-seg.pt`). Si ya tienes los pesos, ponlos en `models/`.

## Uso

- **Añadir**: pega una URL (YouTube u otro sitio compatible con yt-dlp) o arrastra videos o imágenes. Se guardan en `videos/` y se analizan solos en el modo elegido.
- **Modo**: 😄 Emociones / 🎬 Escena. En modo escena el panel sigue a la persona principal (⭐ Auto), y pulsando un `#número` te fijas en una persona.
- **Línea de tiempo**: muestra la etiqueta en cada momento; haz clic para saltar.
- **Calibración**: resta a cada etiqueta su similitud media (en el archivo o en toda la biblioteca). Ayuda mucho con las emociones; en escenas suele ir mejor sin calibrar.
- **📷 Cámara en directo**: usa la webcam (Chrome sólo la permite en `localhost` o con HTTPS). Para probar sin cámara: `http://127.0.0.1:8765/?livesrc=<archivo de videos/>`.

Por línea de comandos:

```bash
.venv/bin/python analyze.py videos/mi-video.mp4 --mode scene --fps 2
```

## Repetir la prueba de actividades

```bash
clips/download_raw.sh                    # descarga los clips originales (no se incluyen en el repo)
.venv/bin/python clips/make_test.py      # monta videos/prueba-actividades.mp4 (11 × 3 s)
.venv/bin/python analyze.py videos/prueba-actividades.mp4 --mode scene --fps 2
.venv/bin/python clips/evaluate.py       # compara con clips/prueba-actividades.json
```

Clips: 4 de [Mixkit](https://mixkit.co) (licencia Mixkit) y 7 shorts de YouTube de sus respectivos autores. Los videos no se redistribuyen aquí.

## Archivos

| archivo | |
|---|---|
| `analyze.py` | detección + Gemma; modos emoción (video/imagen) y escena |
| `server.py` | servidor local (FastAPI): biblioteca, descargas, análisis en cola |
| `live.py` | modo en directo por WebSocket |
| `index.html` | interfaz |
| `clips/` | scripts de la prueba de actividades y sus respuestas correctas |

## Créditos

- [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2), de Google DeepMind
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0)
- [sentence-transformers](https://www.sbert.net/), [OpenCV](https://opencv.org/), [yt-dlp](https://github.com/yt-dlp/yt-dlp)
