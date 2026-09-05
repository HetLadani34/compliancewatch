FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies needed by Playwright and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    wget \
    gnupg \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium and its Linux system libraries
RUN playwright install --with-deps chromium

# Copy application files
COPY . .

# Create data directories
RUN mkdir -p data/screenshots data/chroma_db

# Expose ports: Streamlit (8501/7860), FastAPI (8000), Mock Sites (8100)
EXPOSE 8501 8000 8100 7860

# Launch all 3 services
CMD ["python", "run.py", "--no-browser"]
