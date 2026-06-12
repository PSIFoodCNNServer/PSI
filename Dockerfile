FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && pip install streamlit pillow

COPY . /app

EXPOSE 8501

CMD ["sh", "-c", "echo 'Po starcie aplikacja będzie dostępna pod wskazanym w terminalu adresem.' && exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true"]
