"""Audit test 6 helper: build Step 4's train.Trainer with --init_from <checkpoint> exactly as a warm-start run does,
then compare every parameter of policy and policy_old with the checkpoint BEFORE any update. One process per call
(DANIEL's config is module-global). Prints one JSON line.

    python tests/audit/warmstart_probe.py <checkpoint path> <v1|v2>
"""
import json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DAN = REPO / "daniel"
ck, obs = sys.argv[1], sys.argv[2]
work = DAN / "train_log/audit_warmstart_work"; work.mkdir(parents=True, exist_ok=True)
sys.argv = [str(DAN / "train.py"), "--config", str(REPO / "configs/canonical/mct.json"),
            "--train_data_path", str(DAN / "data/data_train/SD2/10x5+carbon+priority"),
            "--validation_data_path", str(DAN / "data/data_validation/SD2/10x5+carbon+priority"),
            "--test_data_path", str(DAN / "data/data_final_test/SD2/10x5+carbon+priority"),   # provenance only, never read
            "--n_j", "10", "--n_m", "5", "--data_suffix", "carbon+priority", "--carbon_reward_weight", "0.002384",
            "--tardiness_reward_weight", "5.815362", "--seed_train", "111", "--device", "cuda",
            "--model_suffix", "audit_warmstart_k125s060", "--init_from", ck]
if obs == "v2":
    sys.argv += ["--obs_version", "v2", "--reward_version", "v2"]
os.chdir(work)
sys.path.insert(0, str(DAN))
import torch                                              # noqa: E402
import train as T                                         # noqa: E402
from checkpointing import unwrap_checkpoint               # noqa: E402

trainer = T.Trainer(T.configs)
ref, _ = unwrap_checkpoint(torch.load(ck, map_location="cpu", weights_only=False))
res = {"checkpoint": Path(ck).name, "obs": obs}
for which, net in (("policy", trainer.ppo.policy), ("policy_old", trainer.ppo.policy_old)):
    sd = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    res[f"{which}_missing"] = sorted(set(sd) - set(ref))
    res[f"{which}_unexpected"] = sorted(set(ref) - set(sd))
    res[f"{which}_max_abs_diff"] = max(float((sd[k].float() - ref[k].float()).abs().max()) for k in sd if k in ref)
    res[f"{which}_n_tensors"] = len(sd)
res["critic_loaded"] = all(torch.equal(trainer.ppo.policy.state_dict()[k].cpu(), ref[k].cpu()) for k in ref if k.startswith("critic"))
res["optimizer_state_empty"] = len(trainer.ppo.optimizer.state) == 0
print("PROBE " + json.dumps(res))
