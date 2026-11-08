FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt requirements-ml.txt ./
ARG WITH_ML=0
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ "$WITH_ML" = "1" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi
COPY brainrot ./brainrot
COPY api ./api
COPY artifacts ./artifacts
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
