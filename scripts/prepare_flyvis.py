#!/usr/bin/env python3
"""Install the pinned Flyvis model from a separately acquired official ZIP.

Uses only the standard library. It never downloads, runs model code, creates
symlinks, overwrites an existing model, or extracts arbitrary ZIP paths.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_SHA256 = '71c78d4070556a536b13b23ee3139cd2788aa2a9d07d430a223b4edead281db1'
ARCHIVE_URL = 'https://drive.google.com/uc?export=download&id=13cJr2nMn89j-jBAd5RduYRJpBcXwoNrC'
MODEL_NAME = 'flyvis_0000_000'
ARCHIVE_PREFIX = 'results/flow/0000/000/'
SELECTED_FILES = frozenset(('_meta.yaml', 'best_chkpt', 'chkpts/chkpt_00000',
                            'validation/loss.h5', 'validation_loss.h5'))
MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_MEMBER_BYTES = 2 * 1024 * 1024
MAX_EXPANDED_BYTES = 16 * 1024 * 1024
MAX_MEMBERS = 1000


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('Duplicate manifest key: ' + name)
        result[name] = value
    return result


def _manifest_files(root):
    manifest = root / 'models' / (MODEL_NAME + '.manifest.json')
    if manifest.is_symlink() or not manifest.is_file() or manifest.stat().st_size > 65536:
        raise ValueError('A regular portable Flyvis manifest is required')
    value = json.loads(manifest.read_text(encoding='utf-8'), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError('Invalid model manifest')
    files = value.get('files')
    if (value.get('model_dir') != MODEL_NAME or value.get('checkpoint') != 'best_chkpt'
            or not isinstance(files, dict) or set(files) != SELECTED_FILES
            or any(not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None
                   for digest in files.values())):
        raise ValueError('Manifest must pin the selected five files and portable model directory')
    return files


def _safe_member(info):
    # ZipInfo.filename may truncate a NUL-containing original filename.
    name = info.orig_filename
    path = name[:-1] if name.endswith('/') else name
    if (not path or name != info.filename or '\\' in name or ':' in name
            or name.startswith('/') or any(part in ('', '.', '..') for part in path.split('/'))):
        raise ValueError('Unsafe ZIP member path')
    mode = (info.external_attr >> 16) & 0xffff
    file_type = stat.S_IFMT(mode)
    if file_type not in (0, stat.S_IFREG, stat.S_IFDIR) or stat.S_ISLNK(mode):
        raise ValueError('ZIP symlinks and special files are unsupported')
    if (file_type == stat.S_IFDIR) != info.is_dir() and file_type != 0:
        raise ValueError('ZIP member type disagrees with its path')
    if info.flag_bits & 1:
        raise ValueError('Encrypted ZIP members are unsupported')
    if not 0 <= info.file_size <= MAX_MEMBER_BYTES or info.is_dir() and info.file_size != 0:
        raise ValueError('ZIP member exceeds the size limit')
    return name


def selected_payloads(data, expected):
    """Validate structure and selected member hashes; never extract ZIP paths."""
    payloads = {}
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        members = archive.infolist()
        if len(members) > MAX_MEMBERS or sum(i.file_size for i in members) > MAX_EXPANDED_BYTES:
            raise ValueError('ZIP contents exceed the bounded archive size')
        names = set()
        for info in members:
            name = _safe_member(info)
            # Also reject a file and directory sharing the same logical name.
            key = name.rstrip('/')
            if key in names:
                raise ValueError('Duplicate ZIP member path')
            names.add(key)
            if info.is_dir() or not name.startswith(ARCHIVE_PREFIX):
                continue
            relative = name[len(ARCHIVE_PREFIX):]
            if relative not in expected:
                raise ValueError('Unexpected file in the selected model')
            with archive.open(info) as source:
                payload = source.read(MAX_MEMBER_BYTES + 1)
            if len(payload) != info.file_size or len(payload) > MAX_MEMBER_BYTES:
                raise ValueError('ZIP member length mismatch')
            if hashlib.sha256(payload).hexdigest() != expected[relative]:
                raise ValueError('Selected model hash mismatch: ' + relative)
            payloads[relative] = payload
    if set(payloads) != set(expected):
        raise ValueError('Archive is missing selected model files')
    return payloads


def _write_exclusive(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(payload)


def prepare(archive_path, root=ROOT):
    """Validate all input before publishing; clean up our new target on failure.

    Staging uses a private temporary directory. The final directory is claimed
    with an exclusive mkdir, so even an empty existing target is never replaced.
    A failed publication removes only the target this invocation created.
    """
    root = Path(root).resolve()
    models = root / 'models'
    if models.is_symlink() or not models.is_dir():
        raise ValueError('The project models directory must be a regular directory')
    target = models / MODEL_NAME
    if os.path.lexists(target):
        raise FileExistsError('Model target already exists; refusing to overwrite it')
    expected = _manifest_files(root)
    source = Path(archive_path)
    if not source.is_file() or not stat.S_ISREG(source.stat().st_mode):
        raise ValueError('Provide a regular official ZIP archive')
    with source.open('rb') as handle:
        data = handle.read(MAX_ARCHIVE_BYTES + 1)
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError('Archive exceeds the size limit')
    digest = hashlib.sha256(data).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError('Official archive SHA256 mismatch; nothing was installed')
    payloads = selected_payloads(data, expected)
    with tempfile.TemporaryDirectory(prefix='.prepare-flyvis-', dir=models) as temporary:
        staged = Path(temporary)
        for name, payload in payloads.items():
            _write_exclusive(staged / name, payload)
        # Verify disk bytes before making the output model directory visible.
        for name, checksum in expected.items():
            if hashlib.sha256((staged / name).read_bytes()).hexdigest() != checksum:
                raise ValueError('Staged model hash mismatch: ' + name)
        target.mkdir()  # Exclusive, including when another invocation wins a race.
        try:
            for name in sorted(payloads):
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Hard links publish the already verified bytes without rewriting
                # them; link creation is exclusive and both dirs share a volume.
                os.link(staged / name, destination)
        except BaseException:
            shutil.rmtree(target)
            raise
    return dict(model_dir=str(target), archive_sha256=digest, files=expected,
                installed_files=len(payloads), inference_executed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True,
                        help='Official results_pretrained_models.zip acquired separately')
    args = parser.parse_args(argv)
    try:
        result = prepare(args.archive)
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        parser.exit(2, 'Flyvis setup failed: ' + str(exc) + '\n')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
