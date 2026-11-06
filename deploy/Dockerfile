# syntax=docker/dockerfile:1
# Two images from one file (D-040):
#   docker build -f deploy/Dockerfile --target api -t genprobe-api .
#   docker build -f deploy/Dockerfile --target worker -t genprobe-worker .
# The API image has no torch; the worker image adds CPU torch, open_clip and the CLIP ViT-L/14
# weights, so a container starts scoring without downloading anything.

FROM python:3.11-slim AS py
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app /data
WORKDIR /app
COPY deploy/requirements-service.txt /tmp/requirements-service.txt
RUN pip install -r /tmp/requirements-service.txt

FROM py AS worker-deps
# CPU wheels: the default x86_64 wheels pull in ~2 GB of CUDA libraries a CPU worker never uses.
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1; \
    else \
        pip install torch==2.5.1 torchvision==0.20.1; \
    fi
COPY deploy/requirements-worker.txt /tmp/requirements-worker.txt
RUN pip install -r /tmp/requirements-worker.txt
ENV HF_HOME=/opt/hf
RUN python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('timm/vit_large_patch14_clip_224.openai', 'open_clip_model.safetensors')" \
    && chmod -R a+rX /opt/hf

FROM py AS api
COPY pyproject.toml README.md LICENSE ./
COPY genprobe ./genprobe
RUN pip install --no-deps .
USER app
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=2)"
CMD ["uvicorn", "--factory", "genprobe.service.api:create_app", "--host", "0.0.0.0", "--port", "8000"]

FROM worker-deps AS worker
COPY pyproject.toml README.md LICENSE ./
COPY genprobe ./genprobe
RUN pip install --no-deps .
ENV HF_HUB_OFFLINE=1
USER app
CMD ["python", "-m", "genprobe.service.worker"]
