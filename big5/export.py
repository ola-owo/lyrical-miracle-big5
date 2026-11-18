"""Export a standalone SentenceTransformer model for GCS-backed serving."""

import argparse
import gc
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from big5.globalvars import MODEL, MODEL_NAME

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_OFFLINE_CHECK = """
import sys

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(sys.argv[1], local_files_only=True)
embedding = model.encode(['offline verification'], convert_to_numpy=True)
assert embedding.shape == (1, 5), embedding.shape
"""


def merge_adapter(
    model: 'SentenceTransformer',
    source: str,
    revision: str | None = None,
) -> None:
    """Merge the PEFT adapter into the underlying transformer model.

    Args:
        model: A sentence-transformers model with the adapter already loaded.
        source: Hugging Face model ID or local adapter directory.
        revision: Optional Hugging Face revision to load.

    Raises:
        TypeError: If the source does not contain a PEFT adapter.
    """
    from peft import PeftConfig, PeftModel
    from transformers import AutoModel

    try:
        adapter_config = PeftConfig.from_pretrained(source, revision=revision)
    except (OSError, ValueError) as exc:
        raise TypeError(
            'The source model does not have a PEFT adapter to merge. '
            'Export the adapter repository, not an already merged model.'
        ) from exc

    base_model_name = adapter_config.base_model_name_or_path
    if not base_model_name:
        raise TypeError('The adapter config does not specify base_model_name_or_path.')

    logger.info('Loading base model %s', base_model_name)
    base_model = AutoModel.from_pretrained(base_model_name)
    peft_model = PeftModel.from_pretrained(base_model, source, revision=revision)
    model[0].model = peft_model.merge_and_unload()  # pyright: ignore[reportAttributeAccessIssue]


def export_model(
    source: str,
    output: Path,
    revision: str | None = None,
) -> Path:
    """Load an adapter, merge it, and save a self-contained model.

    Args:
        source: Hugging Face model ID or local adapter directory.
        output: Fresh directory that will receive the merged model.
        revision: Optional Hugging Face revision to load.

    Returns:
        The output directory.
    """
    from sentence_transformers import SentenceTransformer

    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f'{output} exists and is not a directory.')
        if any(output.iterdir()):
            raise FileExistsError(
                f'{output} is not empty. Choose a fresh output directory so '
                'stale adapter files cannot leak into the export.'
            )
    output.mkdir(parents=True, exist_ok=True)

    logger.info('Loading adapter from %s', source)
    model = SentenceTransformer(source, revision=revision)
    merge_adapter(model, source, revision)

    logger.info('Saving merged model to %s', output)
    model.save_pretrained(str(output))
    del model
    gc.collect()

    _assert_standalone(output)
    _verify_offline(output)
    return output


def _assert_standalone(path: Path) -> None:
    adapter_files = [
        name
        for name in (
            'adapter_config.json',
            'adapter_model.safetensors',
            'adapter_model.bin',
        )
        if (path / name).exists()
    ]
    if adapter_files:
        raise RuntimeError(
            f'Export still contains PEFT adapter files: {", ".join(adapter_files)}'
        )

    required_files = ('modules.json', 'config.json', 'model.safetensors')
    missing = [name for name in required_files if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f'Export is missing required files: {", ".join(missing)}')


def _verify_offline(path: Path) -> None:
    logger.info('Verifying offline load from %s', path)
    env = {
        **os.environ,
        'HF_HUB_OFFLINE': '1',
        'TRANSFORMERS_OFFLINE': '1',
    }
    subprocess.run(
        [sys.executable, '-c', _OFFLINE_CHECK, str(path)],
        env=env,
        check=True,
    )


def main() -> None:
    """Run the model export CLI."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(
        description='Merge a PEFT adapter into a standalone serving model.'
    )
    parser.add_argument(
        '--source',
        default=MODEL,
        help='Hugging Face model ID or local adapter directory.',
    )
    parser.add_argument(
        '--revision',
        default=None,
        help='Optional Hugging Face revision to load.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('models/export') / MODEL_NAME,
        help='Fresh directory for the merged model.',
    )
    args = parser.parse_args()

    output = export_model(args.source, args.output, args.revision)
    print(f'Exported model to {output}')
    print('Upload it with:')
    print(
        f'  gcloud storage cp -r {output}/* '
        f'gs://<bucket>/big5/models/{MODEL_NAME}/<version>/'
    )


if __name__ == '__main__':
    main()
