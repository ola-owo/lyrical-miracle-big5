FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG TORCH_BACKEND=cpu
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    UV_NO_DEFAULT_GROUPS=1 \
    UV_LOCKED=1

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --extra $TORCH_BACKEND --only-install-package torch
RUN uv sync --extra $TORCH_BACKEND --no-install-project

COPY ./big5 ./big5
RUN uv sync --extra $TORCH_BACKEND

ENV PYTHONUNBUFFERED=1 \
    LOGLEVEL=WARNING \
    AIP_HTTP_PORT=8080 \
    AIP_PREDICT_ROUTE=/predict \
    AIP_HEALTH_ROUTE=/health \
    BIG5_DEBUG=0 \
    BATCH_MODE=1 \
    BATCH_SIZE=32 \
    INFERENCE_TIMEOUT=60

EXPOSE $AIP_HTTP_PORT
CMD ["uv", "run", "--no-sync", "serve"]
