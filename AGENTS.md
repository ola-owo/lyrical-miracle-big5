# Repository Guidelines

## Project Structure & Module Organization

- `inference.py` loads the configured SentenceTransformer model and prints trait scores for sample text.
- `globalvars.py` centralizes Hugging Face IDs, model names, W&B project names, and Big Five trait labels.
- `train_big5_bert_st.ipynb` contains the training workflow and experimentation notes.
- `pyproject.toml` and `uv.lock` define the Python 3.13 environment and pinned dependencies.
- Secrets such as `keys.env` are ignored. Do not commit environment files, tokens, caches, or generated model artifacts.

No dedicated package directory or test directory exists yet. If the project grows, prefer `src/` for reusable modules and `tests/` for automated tests.

## Build, Test, and Development Commands

- `uv sync` installs the locked project dependencies into the local virtual environment.
- `uv run python inference.py` runs the current inference smoke test against `LORA_MODEL`.
- `uv run jupyter lab` opens the notebook workflow, if Jupyter is installed.
- `uv add <package>` adds a runtime dependency and updates `pyproject.toml` plus `uv.lock`.

There is no build step at present; this repository is script/notebook driven.

## Relevant Libraries

- [LitServe](https://lightning.ai/docs/litserve) for serving model inference
- [SentenceTransformers](https://www.sbert.net/) for building and using transformer models

## Coding Style & Naming Conventions

Use Python 3.13 syntax and keep modules simple. Follow PEP 8 with 4-space indentation, lowercase module names, `snake_case` functions and variables, and `UPPER_CASE` constants for shared configuration. Keep Hugging Face, W&B, dataset, and trait metadata in `globalvars.py`.

Prefer explicit imports and readable data transformations. If adding formatting or linting, use Ruff and keep configuration in `pyproject.toml`.

## Testing Guidelines

Automated tests are not currently present. For new reusable logic, add `pytest` tests under `tests/`, with files named `test_<module>.py` and functions named `test_<behavior>()`. Keep model/network-heavy tests optional or mocked; shape, schema, and configuration tests should run offline.

When tests are added, the expected command should be:

```bash
uv run pytest
```

## Commit & Pull Request Guidelines

The current Git history only contains `init`, so no strict convention is established. Use short, imperative commit messages such as `add inference smoke test`.

Pull requests should include a concise summary, any relevant issue links, commands run, and notes about model, dataset, or dependency changes. Include screenshots only for notebook output or visual reports when they help reviewers verify results.

## Security & Configuration Tips

Do not read, print, or commit `.env` files or tokens. Keep Hugging Face and W&B credentials in local environment files or shell configuration. Before sharing notebooks, clear sensitive outputs and avoid embedding private dataset paths or access tokens.
