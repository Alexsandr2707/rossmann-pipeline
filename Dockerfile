FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app ./app
COPY config ./config
COPY tests ./tests
COPY run.py README.md ./

RUN mkdir -p \
    data/external \
    data/raw \
    data/processed \
    artifacts/predictions \
    models \
    reports \
    logs

ENTRYPOINT ["python", "run.py"]
CMD ["-mode", "summary"]
