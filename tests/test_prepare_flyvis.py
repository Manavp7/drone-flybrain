"""Local synthetic ZIP checks; no model download or inference is performed."""
from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from scripts import prepare_flyvis as setup


class PrepareFlyvisTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.models = self.root / 'models'
        self.models.mkdir()
        self.payloads = {name: ('test bytes: ' + name).encode() for name in setup.SELECTED_FILES}
        self.expected = {name: hashlib.sha256(value).hexdigest() for name, value in self.payloads.items()}
        self.manifest = self.models / (setup.MODEL_NAME + '.manifest.json')
        self.manifest.write_text(json.dumps(dict(model_dir=setup.MODEL_NAME,
                                                checkpoint='best_chkpt', files=self.expected)))
        self.target = self.models / setup.MODEL_NAME
        self.archive = self.root / 'official.zip'

    def zip_bytes(self, *, omit=(), additions=(), replacements=None):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED) as archive:
            for name, payload in self.payloads.items():
                if name not in omit:
                    archive.writestr(setup.ARCHIVE_PREFIX + name,
                                     (replacements or {}).get(name, payload))
            archive.writestr('results/flow/0000/001/best_chkpt', b'other model')
            for name, payload in additions:
                archive.writestr(name, payload)
        return stream.getvalue()

    def install(self, data=None):
        data = self.zip_bytes() if data is None else data
        self.archive.write_bytes(data)
        # Synthetic fixtures exercise all post-authentication safeguards. The
        # production CLI has no checksum override and pins the official archive.
        with patch.object(setup, 'ARCHIVE_SHA256', hashlib.sha256(data).hexdigest()):
            return setup.prepare(self.archive, self.root)

    def assert_clean(self):
        self.assertFalse(self.target.exists())
        self.assertFalse(self.target.is_symlink())
        self.assertEqual(list(self.models.glob('.prepare-flyvis-*')), [])

    def test_selected_five_bytes_only_and_manifest_unchanged(self):
        original = self.manifest.read_bytes()
        result = self.install()
        actual = {p.relative_to(self.target).as_posix(): p.read_bytes()
                  for p in self.target.rglob('*') if p.is_file()}
        self.assertEqual(actual, self.payloads)
        self.assertFalse(any(p.is_symlink() for p in self.target.rglob('*')))
        self.assertEqual(self.manifest.read_bytes(), original)
        self.assertEqual(result['installed_files'], 5)
        self.assertFalse(result['inference_executed'])
        self.assertEqual(list(self.models.glob('.prepare-flyvis-*')), [])

    def test_wrong_full_archive_digest_rejected_before_zip_parse(self):
        self.archive.write_bytes(b'not the official archive')
        with patch.object(setup, 'selected_payloads') as select:
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                setup.prepare(self.archive, self.root)
            select.assert_not_called()
        self.assert_clean()

    def test_per_member_digest_rejects_changed_selected_bytes(self):
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.install(self.zip_bytes(replacements={'best_chkpt': b'changed checkpoint'}))
        self.assert_clean()

    def test_missing_selected_file_is_not_replaced_from_other_model(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.install(self.zip_bytes(omit=('best_chkpt',)))
        self.assert_clean()

    def test_unexpected_selected_file_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            self.install(self.zip_bytes(additions=[(setup.ARCHIVE_PREFIX + 'extra.py', b'code')]))
        self.assert_clean()

    def test_duplicate_zip_entries_are_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            data = self.zip_bytes(additions=[(setup.ARCHIVE_PREFIX + 'best_chkpt', self.payloads['best_chkpt'])])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.install(data)
        self.assert_clean()

    def test_path_traversal_absolute_and_windows_paths_anywhere_rejected(self):
        for name in ('../escape', '/absolute', 'C:/escape', 'results/../escape',
                     'results\\escape', 'results//escape', 'results/./escape'):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'Unsafe'):
                self.install(self.zip_bytes(additions=[(name, b'bad')]))
            self.assert_clean()
        self.assertFalse((self.root / 'escape').exists())

    def test_symlink_and_device_members_never_materialized(self):
        for mode in (stat.S_IFLNK | 0o777, stat.S_IFCHR | 0o600):
            info = zipfile.ZipInfo('results/unselected-link')
            info.create_system = 3
            info.external_attr = mode << 16
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'symlinks and special'):
                self.install(self.zip_bytes(additions=[(info, b'../../outside')]))
            self.assert_clean()

    def test_file_directory_type_confusion_rejected(self):
        info = zipfile.ZipInfo('results/pretend-directory/')
        info.create_system = 3
        info.external_attr = (stat.S_IFREG | 0o644) << 16
        with self.assertRaisesRegex(ValueError, 'type disagrees'):
            self.install(self.zip_bytes(additions=[(info, b'')]))
        self.assert_clean()

    def test_archive_member_and_total_size_limits(self):
        data = self.zip_bytes()
        self.archive.write_bytes(data)
        with patch.object(setup, 'MAX_ARCHIVE_BYTES', len(data) - 1):
            with self.assertRaisesRegex(ValueError, 'Archive exceeds'):
                setup.prepare(self.archive, self.root)
        for name, limit in (('MAX_MEMBER_BYTES', 5), ('MAX_EXPANDED_BYTES', 5), ('MAX_MEMBERS', 1)):
            with self.subTest(limit=name), patch.object(setup, name, limit):
                with self.assertRaisesRegex(ValueError, 'size'):
                    self.install(data)
            self.assert_clean()

    def test_manifest_path_or_member_changes_rejected(self):
        for edit in ({'model_dir': '../outside'}, {'checkpoint': 'different'}, {'files': {'../bad': 'a' * 64}}):
            value = dict(model_dir=setup.MODEL_NAME, checkpoint='best_chkpt', files=self.expected)
            value.update(edit)
            self.manifest.write_text(json.dumps(value))
            with self.subTest(edit=edit), self.assertRaisesRegex(ValueError, 'Manifest must pin'):
                self.install()
            self.assert_clean()

    def test_existing_file_directory_and_dangling_link_are_preserved(self):
        self.target.write_bytes(b'keep file')
        with self.assertRaises(FileExistsError): self.install()
        self.assertEqual(self.target.read_bytes(), b'keep file')
        self.target.unlink()
        self.target.mkdir()
        with self.assertRaises(FileExistsError): self.install()
        self.assertTrue(self.target.is_dir())
        self.target.rmdir()
        self.target.symlink_to(self.root / 'missing')
        with self.assertRaises(FileExistsError): self.install()
        self.assertTrue(self.target.is_symlink())

    def test_models_directory_symlink_is_rejected(self):
        self.models.rename(self.root / 'real_models')
        self.models.symlink_to(self.root / 'real_models', target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'regular directory'):
            self.install()
        self.assertFalse(self.target.exists())

    def test_failed_staging_leaves_no_partial_model(self):
        real_write = setup._write_exclusive
        calls = []
        def fail_second(path, payload):
            calls.append(path)
            if len(calls) == 2: raise OSError('simulated disk error')
            real_write(path, payload)
        with patch.object(setup, '_write_exclusive', side_effect=fail_second):
            with self.assertRaisesRegex(OSError, 'disk error'): self.install()
        self.assert_clean()

    def test_failed_publication_removes_only_our_new_target(self):
        real_link = setup.os.link
        calls = []
        def fail_second(source, destination):
            calls.append(destination)
            if len(calls) == 2: raise OSError('simulated publication error')
            real_link(source, destination)
        with patch.object(setup.os, 'link', side_effect=fail_second):
            with self.assertRaisesRegex(OSError, 'publication error'): self.install()
        self.assert_clean()
        self.assertTrue(self.manifest.is_file())

    def test_another_install_winning_destination_race_is_not_removed(self):
        real_write = setup._write_exclusive
        def competing_install(path, payload):
            real_write(path, payload)
            if not self.target.exists():
                self.target.mkdir()
                (self.target / 'owned-by-other').write_text('preserve')
        with patch.object(setup, '_write_exclusive', side_effect=competing_install):
            with self.assertRaises(FileExistsError): self.install()
        self.assertEqual((self.target / 'owned-by-other').read_text(), 'preserve')
        self.assertEqual(list(self.models.glob('.prepare-flyvis-*')), [])

    def test_cli_requires_explicit_local_archive(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as result:
            setup.main([])
        self.assertEqual(result.exception.code, 2)
        with patch.object(setup, 'prepare', return_value={'inference_executed': False}) as prepare:
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(setup.main(['--archive', 'user-acquired.zip']), 0)
            prepare.assert_called_once_with(Path('user-acquired.zip'))
            self.assertFalse(json.loads(output.getvalue())['inference_executed'])


if __name__ == '__main__':
    unittest.main()
