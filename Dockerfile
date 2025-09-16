# Dockerfile (place at repo root)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system deps (only if you need build tools; remove if not)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python deps
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose is optional (Render sets the port), but nice for local testing
EXPOSE 8501

# Use bash -lc so $PORT expands properly. Replace app.py with your streamlit file.
CMD ["bash", "-lc", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]
