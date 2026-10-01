"""Audit test 6: after loading each specialist for fine-tuning, every parameter of policy and policy_old equals the
checkpoint before the first update (no missing, unexpected or randomly initialised keys; critic included).

    python -m unittest tests.audit.test_warmstart -v
"""
import json, subprocess, sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CK = REPO / "daniel/trained_network/SD2"
CASES = [("10x5+carbon+priority+m_s111.pth", "v1"), ("10x5+carbon+priority+c_s111.pth", "v1"),
         ("10x5+carbon+priority+t_k125s060_s111.pth", "v1"),
         ("10x5+carbon+priority+simplex_m500c100t400_k125s060_s111.pth", "v1"),          # Step 4 merge init, seed 111
         ("10x5+carbon+priority+s4t_spec_m_k125s060_s111@u1000.pth", "v1"),
         ("10x5+carbon+priority+m_s111_obsv2.pth", "v2"), ("10x5+carbon+priority+t_k125s060_s111_v2ft.pth", "v2")]


class WarmStart(unittest.TestCase):
    def test_all_parameters_loaded(self):
        rows = []
        for name, obs in CASES:
            p = subprocess.run([sys.executable, str(Path(__file__).with_name("warmstart_probe.py")), str(CK / name), obs],
                               capture_output=True, text=True, timeout=600)
            line = next((l for l in p.stdout.splitlines() if l.startswith("PROBE ")), None)
            self.assertIsNotNone(line, f"{name}: probe failed\n{p.stderr[-2000:]}")
            rows.append(json.loads(line[6:]))
        out = REPO / "results/code_audit/tests"; out.mkdir(parents=True, exist_ok=True)
        (out / "t6_warmstart.json").write_text(json.dumps(rows, indent=1))
        for r in rows:
            for w in ("policy", "policy_old"):
                self.assertEqual(r[f"{w}_missing"], [], r["checkpoint"])
                self.assertEqual(r[f"{w}_unexpected"], [], r["checkpoint"])
                self.assertEqual(r[f"{w}_max_abs_diff"], 0.0, r["checkpoint"])
            self.assertTrue(r["critic_loaded"], r["checkpoint"])


if __name__ == "__main__":
    unittest.main()
