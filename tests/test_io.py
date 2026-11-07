import json
import tempfile
import unittest
from pathlib import Path
from m3_filament.io import write_json, write_jsonl, read_jsonl, sha256


class ArtifactIOTest(unittest.TestCase):
    def test_existing_artifact_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'record.json'
            write_json(p,{'version':1})
            with self.assertRaises(FileExistsError):write_json(p,{'version':2})
            self.assertEqual(json.loads(p.read_text()),{'version':1})

    def test_bad_serialization_does_not_leave_partial_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'records.jsonl'
            with self.assertRaises(TypeError):write_jsonl(p,[{'bad':object()}])
            self.assertFalse(p.exists())

    def test_jsonl_roundtrip_and_known_file_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'rows.jsonl';rows=[{'observation_id':'图1','instances':[]}]
            write_jsonl(p,rows)
            self.assertEqual(list(read_jsonl(p)),rows)
            q=Path(tmp)/'empty';q.write_bytes(b'')
            self.assertEqual(sha256(q),'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855')
