import sys
import json
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "daniel"))

from data_manifest import assert_disjoint_splits, structural_fjsp_sha256
from data_manifest import dataset_fingerprint
from checkpointing import file_sha256
from evaluation_protocol import verify_frozen_evaluation


STANDARD = "1 2 2\n1 2 1 3 2 5\n"
EXTENDED = "1 2 2\n1 2 50 1 3 7 2 5 9\n"
DISTINCT = "1 2 2\n1 2 1 4 2 6\n"


class DatasetManifestTests(unittest.TestCase):
    def test_annotations_do_not_change_structural_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            standard = root / "standard.fjs"
            extended = root / "extended.fjs"
            standard.write_text(STANDARD, encoding="utf-8")
            extended.write_text(EXTENDED, encoding="utf-8")
            self.assertEqual(structural_fjsp_sha256(standard), structural_fjsp_sha256(extended))

    def test_structural_overlap_across_splits_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train"
            validation = root / "validation"
            train.mkdir()
            validation.mkdir()
            (train / "a.fjs").write_text(STANDARD, encoding="utf-8")
            (validation / "b.fjs").write_text(EXTENDED, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "dataset leakage"):
                assert_disjoint_splits(train=train, validation=validation)

    def test_distinct_splits_pass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train = root / "train"
            validation = root / "validation"
            train.mkdir()
            validation.mkdir()
            (train / "a.fjs").write_text(STANDARD, encoding="utf-8")
            (validation / "b.fjs").write_text(DISTINCT, encoding="utf-8")
            result = assert_disjoint_splits(train=train, validation=validation)
            self.assertEqual(set(result), {"train", "validation"})

    def test_frozen_evaluation_rechecks_checkpoint_and_test_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            test = root / "test"
            test.mkdir()
            instance = test / "case.fjs"
            instance.write_text(STANDARD, encoding="utf-8")
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"checkpoint")
            manifest = root / "frozen.json"
            manifest.write_text(json.dumps({
                "status": "FROZEN_BEFORE_FINAL_TEST",
                "checkpoints": [{"sha256": file_sha256(checkpoint)}],
                "splits": {"test": dataset_fingerprint(test)},
            }), encoding="utf-8")
            verify_frozen_evaluation(manifest, checkpoint, test)
            instance.write_text(DISTINCT, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint changed"):
                verify_frozen_evaluation(manifest, checkpoint, test)


if __name__ == "__main__":
    unittest.main()
