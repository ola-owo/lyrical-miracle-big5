# Repository Guidelines

## Project Structure & Module Organization

- `globalvars.py` centralizes Hugging Face IDs, model names, W&B project names,
  and Big Five trait labels.
- `train.py` is a Marimo notebook with the training code and experimentation notes.
- `pyproject.toml` and `uv.lock` define the Python 3.13 environment and pinned dependencies.

## Relevant Libraries

- [LitServe](https://lightning.ai/docs/litserve) for serving model inference
- [SentenceTransformers](https://www.sbert.net/) for building and using transformer models
- [Gemini Agent Platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform) AKA Vertex
  AI, used for running AI/ML workflows in Google Cloud

## Testing Guidelines

Testing is done through pytest with test files located in `tests/`.

## Security & Configuration Tips

- Do not read, print, or commit `.env` files or tokens.
- Keep Hugging Face and W&B credentials in local environment files or shell configuration.
- Before sharing notebooks, scrub all sensitive outputs, private dataset paths, or access tokens.
