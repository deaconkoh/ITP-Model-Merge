import sys
import unittest
from argparse import Namespace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "daniel"))

from checkpointing import unwrap_checkpoint, validate_checkpoint_schema
import numpy as np
from feature_schemas import validate_config_against_schema


class ContractTests(unittest.TestCase):
    def test_canonical_feature_schema(self):
        config = Namespace(
            feature_schema="canonical_f11_p9_v2",
            fea_j_input_dim=11,
            fea_m_input_dim=8,
            fea_pair_input_dim=9,
            enable_priority=True,
            enable_carbon=True,
            carbon_feature=True,
        )
        schema = validate_config_against_schema(config)
        self.assertEqual((schema.operation_dim, schema.machine_dim, schema.pair_dim), (11, 8, 9))

    def test_dimension_drift_is_rejected(self):
        config = Namespace(
            feature_schema="canonical_f11_p9_v2",
            fea_j_input_dim=11,
            fea_m_input_dim=8,
            fea_pair_input_dim=8,
            enable_priority=True,
            enable_carbon=True,
            carbon_feature=True,
        )
        with self.assertRaisesRegex(ValueError, "do not match schema"):
            validate_config_against_schema(config)

    def test_legacy_state_dict_is_not_silently_canonical(self):
        state_dict = {"weight": object()}
        unwrapped, metadata = unwrap_checkpoint(state_dict)
        self.assertIs(unwrapped, state_dict)
        self.assertTrue(metadata["legacy"])
        self.assertEqual(metadata["provenance"], "UNKNOWN")
        with self.assertRaisesRegex(ValueError, "explicit legacy"):
            validate_checkpoint_schema(metadata, "canonical_f11_p9_v2")

    def test_legacy_tensor_family_must_match_selected_schema(self):
        state_dict = {
            "feature_exact.op_attention_blocks.0.attention_0.W": np.zeros((11, 32)),
            "feature_exact.mch_attention_blocks.0.attention_0.W": np.zeros((8, 32)),
            "actor.linears.0.weight": np.zeros((64, 41)),
        }
        metadata = {"legacy": True}
        validate_checkpoint_schema(metadata, "legacy_f11_p9_v1", state_dict)
        with self.assertRaisesRegex(ValueError, "tensor family"):
            validate_checkpoint_schema(metadata, "legacy_f11_p8_v1", state_dict)


if __name__ == "__main__":
    unittest.main()
