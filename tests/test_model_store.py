from pathlib import Path

import pytest

from big5 import model_store


class FakeBlob:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self.content = content

    def download_to_filename(self, filename: str) -> None:
        Path(filename).write_bytes(self.content)


class FakeBucket:
    def __init__(self, blobs: list[FakeBlob]) -> None:
        self.blobs = blobs


class FakeStorageClient:
    def __init__(self, blobs: list[FakeBlob]) -> None:
        self.blobs = blobs
        self.list_calls = 0

    def bucket(self, name: str) -> FakeBucket:
        self.bucket_name = name
        return FakeBucket(self.blobs)

    def list_blobs(self, bucket: FakeBucket, prefix: str) -> list[FakeBlob]:
        self.list_calls += 1
        return [blob for blob in bucket.blobs if blob.name.startswith(prefix)]


def test_resolve_model_uri_prefers_big5_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('BIG5_MODEL_URI', 'gs://bucket/big5')
    monkeypatch.setenv('AIP_STORAGE_URI', 'gs://managed/model')

    assert model_store.resolve_model_uri() == 'gs://bucket/big5'


def test_resolve_model_uri_falls_back_to_vertex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('BIG5_MODEL_URI', '   ')
    monkeypatch.setenv('AIP_STORAGE_URI', 'gs://managed/model')

    assert model_store.resolve_model_uri() == 'gs://managed/model'


def test_resolve_model_uri_requires_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('BIG5_MODEL_URI', raising=False)
    monkeypatch.delenv('AIP_STORAGE_URI', raising=False)

    with pytest.raises(model_store.ModelSourceError, match='BIG5_MODEL_URI'):
        model_store.resolve_model_uri()


def test_resolve_local_model_path(tmp_path: Path) -> None:
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    (model_dir / 'modules.json').write_text('[]', encoding='utf-8')

    assert model_store.resolve_model_path(str(model_dir)) == model_dir.resolve()


def test_resolve_local_model_requires_modules_json(tmp_path: Path) -> None:
    with pytest.raises(model_store.ModelSourceError, match='modules.json'):
        model_store.resolve_model_path(str(tmp_path))


def test_resolve_model_path_rejects_unknown_scheme() -> None:
    with pytest.raises(model_store.ModelSourceError, match='Unsupported'):
        model_store.resolve_model_path('s3://bucket/model')


def test_download_gcs_prefix_and_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    blobs = [
        FakeBlob('models/v1/modules.json', b'[]'),
        FakeBlob('models/v1/model.safetensors', b'weights'),
        FakeBlob('models/v1/1_Pooling/config.json', b'{}'),
        FakeBlob('models/v10/modules.json', b'wrong model'),
    ]
    client = FakeStorageClient(blobs)
    monkeypatch.setattr(model_store, '_storage_client', lambda: client)

    model_path = model_store.resolve_model_path(
        'gs://bucket/models/v1', cache_dir=tmp_path
    )

    assert (model_path / 'modules.json').read_bytes() == b'[]'
    assert (model_path / 'model.safetensors').read_bytes() == b'weights'
    assert (model_path / '1_Pooling' / 'config.json').is_file()
    assert (model_path / model_store.COMPLETE_MARKER).is_file()
    assert not (model_path / 'v10').exists()
    assert client.list_calls == 1

    cached_path = model_store.resolve_model_path(
        'gs://bucket/models/v1', cache_dir=tmp_path
    )

    assert cached_path == model_path
    assert client.list_calls == 1


def test_download_requires_complete_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeStorageClient([FakeBlob('models/v1/modules.json', b'[]')])
    monkeypatch.setattr(model_store, '_storage_client', lambda: client)

    model_path = model_store.resolve_model_path(
        'gs://bucket/models/v1', cache_dir=tmp_path
    )
    (model_path / model_store.COMPLETE_MARKER).unlink()

    model_store.resolve_model_path('gs://bucket/models/v1', cache_dir=tmp_path)

    assert client.list_calls == 2


def test_download_requires_modules_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeStorageClient([FakeBlob('models/v1/weights.bin', b'weights')])
    monkeypatch.setattr(model_store, '_storage_client', lambda: client)

    with pytest.raises(model_store.ModelSourceError, match='modules.json'):
        model_store.resolve_model_path('gs://bucket/models/v1', cache_dir=tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_download_rejects_empty_prefix(tmp_path: Path) -> None:
    with pytest.raises(model_store.ModelSourceError, match='prefix'):
        model_store.resolve_model_path('gs://bucket', cache_dir=tmp_path)
