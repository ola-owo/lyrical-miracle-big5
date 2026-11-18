"""Resolve and fetch model artifacts for serving.

Serving is decoupled from Hugging Face: the model must be available locally or
in a Google Cloud Storage prefix. Cloud Storage access uses Application Default
Credentials (ADC), so no service-account keys are required.
"""

import hashlib
import logging
import os
import shutil
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlparse

DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / 'big5-model-cache'
MODEL_URI_ENV = 'BIG5_MODEL_URI'
VERTEX_MODEL_URI_ENV = 'AIP_STORAGE_URI'
MODEL_CACHE_ENV = 'BIG5_MODEL_CACHE'
COMPLETE_MARKER = '.complete'

logger = logging.getLogger(__name__)


class ModelSourceError(RuntimeError):
    """Raised when the serving model location cannot be resolved."""


def resolve_model_uri() -> str:
    """Return the configured model URI.

    ``BIG5_MODEL_URI`` takes precedence over ``AIP_STORAGE_URI``. The latter is
    set automatically by Vertex AI when a model is deployed with an
    ``artifactUri``.

    Returns:
        The configured ``gs://`` prefix or local model directory.

    Raises:
        ModelSourceError: If neither environment variable is set.
    """
    configured = [
        (name, os.getenv(name, '').strip())
        for name in (MODEL_URI_ENV, VERTEX_MODEL_URI_ENV)
    ]
    set_values = [(name, value) for name, value in configured if value]
    if not set_values:
        raise ModelSourceError(
            'No serving model configured. Set BIG5_MODEL_URI to a gs:// prefix '
            'or a local directory containing modules.json, or deploy with a '
            f'Vertex artifactUri so that {VERTEX_MODEL_URI_ENV} is set. '
            'Hugging Face fallback is disabled.'
        )
    if len(set_values) > 1:
        logger.info(
            'Both %s and %s are set; using %s.',
            MODEL_URI_ENV,
            VERTEX_MODEL_URI_ENV,
            set_values[0][0],
        )
    return set_values[0][1]


def resolve_model_path(
    uri: str | None = None, *, cache_dir: Path | None = None
) -> Path:
    """Resolve a model URI to a local directory containing ``modules.json``.

    Args:
        uri: Optional ``gs://`` prefix or local directory. When omitted, the
            value from :func:`resolve_model_uri` is used.
        cache_dir: Optional download cache directory. Defaults to
            ``BIG5_MODEL_CACHE`` or a directory under the system temp dir.

    Returns:
        A local directory that can be passed to ``SentenceTransformer``.

    Raises:
        ModelSourceError: If the URI is invalid, unavailable, or does not point
            to a sentence-transformers model.
    """
    candidate = (uri or resolve_model_uri()).strip()
    if not candidate:
        raise ModelSourceError('Model URI is empty.')

    parsed = urlparse(candidate)
    if parsed.scheme == 'gs':
        return _download_gcs_prefix(candidate, _get_cache_dir(cache_dir))
    if parsed.scheme == 'file':
        return _validate_local_model(Path(unquote(parsed.path)))
    if parsed.scheme:
        raise ModelSourceError(
            f'Unsupported model URI scheme {parsed.scheme!r}. '
            'Use a gs:// prefix, a file:// URI, or a local path.'
        )
    return _validate_local_model(Path(candidate).expanduser())


def _get_cache_dir(cache_dir: Path | None) -> Path:
    if cache_dir is not None:
        return cache_dir.expanduser()

    configured = os.getenv(MODEL_CACHE_ENV, '').strip()
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_CACHE_DIR


def _validate_local_model(path: Path) -> Path:
    resolved = path.resolve()
    if not (resolved / 'modules.json').is_file():
        raise ModelSourceError(
            f'Model directory {resolved} does not contain modules.json.'
        )
    return resolved


def _storage_client():
    try:
        from google.auth.exceptions import DefaultCredentialsError
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - dependency is required
        raise ModelSourceError(
            'google-cloud-storage is required to load models from gs:// URIs.'
        ) from exc

    try:
        return storage.Client()
    except DefaultCredentialsError as exc:
        raise ModelSourceError(
            'Google Cloud credentials not found. Run '
            '"gcloud auth application-default login" locally, or run in an '
            'environment with Application Default Credentials.'
        ) from exc


def _is_complete(path: Path) -> bool:
    return (path / COMPLETE_MARKER).is_file() and (path / 'modules.json').is_file()


def _download_gcs_prefix(uri: str, cache_dir: Path) -> Path:
    parsed = urlparse(uri)
    bucket_name = parsed.netloc
    prefix = parsed.path.strip('/')
    if not bucket_name:
        raise ModelSourceError(f'Invalid Cloud Storage URI: {uri}')
    if not prefix:
        raise ModelSourceError(
            'Refusing to download an entire Cloud Storage bucket. '
            'Specify a prefix that contains modules.json.'
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(uri.encode('utf-8')).hexdigest()[:16]
    destination = cache_dir / digest
    if _is_complete(destination):
        logger.info('Using cached model at %s for %s.', destination, uri)
        return destination

    temporary = cache_dir / f'.{digest}.tmp-{os.getpid()}'
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir(parents=True)

    try:
        client = _storage_client()
        blob_prefix = f'{prefix}/'
        blobs = [
            blob
            for blob in client.list_blobs(
                client.bucket(bucket_name), prefix=blob_prefix
            )
            if not blob.name.endswith('/')
        ]
        if not blobs:
            raise ModelSourceError(f'No objects found at {uri}.')

        logger.info('Downloading %d objects from %s.', len(blobs), uri)
        for blob in blobs:
            relative = blob.name[len(blob_prefix) :]
            if not relative:
                continue
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(target))

        if not (temporary / 'modules.json').is_file():
            raise ModelSourceError(
                f'{uri} does not look like a sentence-transformers model: '
                'modules.json is missing at the prefix root.'
            )

        (temporary / COMPLETE_MARKER).write_text(uri, encoding='utf-8')
        if destination.exists():
            if _is_complete(destination):
                shutil.rmtree(temporary, ignore_errors=True)
                return destination
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        try:
            os.replace(temporary, destination)
        except OSError:
            if _is_complete(destination):
                shutil.rmtree(temporary, ignore_errors=True)
                return destination
            raise
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    logger.info('Model downloaded to %s.', destination)
    return destination
