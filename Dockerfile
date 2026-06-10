FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ARG PYLOCK=pylock.cpu.toml
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    AIP_HTTP_PORT=8000 \
    AIP_PREDICT_ROUTE=/predict \
    AIP_HEALTH_ROUTE=/health

WORKDIR /app

COPY pyproject.toml pylock.*.toml ./
RUN uv venv && uv pip sync $PYLOCK

COPY ./big5 ./big5
RUN uv pip install --no-editable .

EXPOSE $AIP_HTTP_PORT
CMD ["uv", "run", "python", "serve.py"]
