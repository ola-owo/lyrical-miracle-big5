from pathlib import Path

import pytest

from big5 import export


def _write_required_files(path: Path) -> None:
    (path / 'modules.json').write_text('[]', encoding='utf-8')
    (path / 'config.json').write_text('{}', encoding='utf-8')
    (path / 'model.safetensors').write_bytes(b'weights')


def test_assert_standalone_accepts_required_files(tmp_path: Path) -> None:
    _write_required_files(tmp_path)

    export._assert_standalone(tmp_path)


def test_assert_standalone_rejects_adapter_files(tmp_path: Path) -> None:
    _write_required_files(tmp_path)
    (tmp_path / 'adapter_config.json').write_text('{}', encoding='utf-8')

    with pytest.raises(RuntimeError, match='adapter'):
        export._assert_standalone(tmp_path)


def test_assert_standalone_requires_model_files(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match='missing'):
        export._assert_standalone(tmp_path)


def test_export_refuses_non_empty_output(tmp_path: Path) -> None:
    (tmp_path / 'stale-file').write_text('stale', encoding='utf-8')

    with pytest.raises(FileExistsError, match='not empty'):
        export.export_model('unused-source', tmp_path)
