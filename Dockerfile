FROM python:3.12-slim

# tesseract-ocr + tesseract-ocr-fra : fallback OCR pour PDF scannés (voir main.py).
# poppler-utils : requis par pdf2image pour convertir une page PDF en image avant OCR.
# L'environnement natif (non-Docker) de Render n'autorise pas apt-get au build,
# d'où le passage par une image Docker pour ce service.
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-fra \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# torch (dépendance transitive de sentence-transformers) installe par défaut
# les paquets nvidia-cu13 (CUDA), inutiles ici (projet Zéro-GPU, voir README)
# et qui gonflent l'image à ~3 Go — problématique sur le plan gratuit Render
# (RAM/temps de build limités). On installe d'abord la variante CPU-only
# officielle : pip install -r requirements.txt ensuite ne réinstalle pas
# torch, sa contrainte de version étant déjà satisfaite.
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render fournit le port d'écoute via la variable d'environnement $PORT
# (jamais un port fixe) — forme shell du CMD pour permettre son expansion.
CMD uvicorn main:app --host 0.0.0.0 --port $PORT
