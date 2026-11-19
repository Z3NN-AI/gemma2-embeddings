#!/usr/bin/env bash
# Descarga los clips originales usados por make_test.py en clips/raw/.
# Los videos son de sus autores (YouTube) o de Mixkit (licencia Mixkit); no se incluyen en el repo.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p raw && cd raw
for id in 9dUW9YCAJ7M zUCrIeXEWLc hPrv-0QDnSc Wbo0IY1tSRo FyyODwkJZJc ZmM36viuNus; do
  uvx yt-dlp@latest -q -f "bv*[height<=720][ext=mp4]/b[ext=mp4]/bv*+ba/b" --merge-output-format mp4 \
    -o "yt_$id.%(ext)s" "https://youtube.com/shorts/$id"
done
for id in 2002 40881 4720 4823; do
  curl -sL -o "mk_$id.mp4" "https://assets.mixkit.co/videos/$id/$id-720.mp4"
done
# el video de entrenamiento va en videos/ (lo usa make_test.py)
cd ../..
uvx yt-dlp@latest -q -f "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]" --merge-output-format mp4 \
  -o "videos/%(title).60B [%(id)s].%(ext)s" --restrict-filenames "https://youtube.com/shorts/NAKsQjS1Gyg"
echo "listo: ahora  .venv/bin/python clips/make_test.py"
