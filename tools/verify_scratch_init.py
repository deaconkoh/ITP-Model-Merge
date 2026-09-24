"""Sanity check for Step 4 arm (b): is scratch@u0 exactly a random initialisation?

Rebuilds the policy the way train.py does for the scratch arm of one seed (same arguments, same seed,
no --init_from) and compares its weights with the saved @u0 budget checkpoint. Identical weights prove
the budget-0 point is the untrained network. The Trainer writes a run manifest under a SANITYCHECK
suffix; nothing is trained.

    cd daniel && python ../tools/verify_scratch_init.py <seed>
"""
import sys
from pathlib import Path
seed = sys.argv[1]
sys.argv = ["train.py", "--config", "../configs/canonical/mct.json",
            "--train_data_path", "./data/data_train/SD2/10x5+carbon+priority",
            "--validation_data_path", "./data/data_validation/SD2/10x5+carbon+priority",
            "--test_data_path", "./data/data_final_test/SD2/10x5+carbon+priority",
            "--n_j", "10", "--n_m", "5", "--data_suffix", "carbon+priority",
            "--carbon_reward_weight", "0.002384", "--tardiness_reward_weight", "5.815362",
            "--max_updates", "1000", "--device", "cuda", "--budget_checkpoints", "0,50,100,250,500,1000",
            "--seed_train", seed, "--model_suffix", f"SANITYCHECK_s4t_scratch_k125s060_s{seed}"]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "daniel"))
import torch
import train
from common_utils import load_checkpoint_state_dict
t = train.Trainer(train.configs)
fresh = {k: v.detach().cpu() for k, v in t.ppo.policy.state_dict().items()}
saved = load_checkpoint_state_dict(f"./trained_network/SD2/10x5+carbon+priority+s4t_scratch_k125s060_s{seed}@u0.pth",
                                   map_location="cpu", expected_schema=train.configs.feature_schema)
same = fresh.keys() == saved.keys() and all(torch.equal(fresh[k], saved[k].cpu()) for k in fresh)
diff = max(float((fresh[k] - saved[k].cpu()).abs().max()) for k in fresh)
spec = load_checkpoint_state_dict(f"./trained_network/SD2/10x5+carbon+priority+c_s{seed}.pth",
                                  map_location="cpu", expected_schema=train.configs.feature_schema)
dspec = sum(float((fresh[k] - spec[k].cpu()).norm() ** 2) for k in fresh) ** 0.5
print(f"seed {seed}: scratch@u0 weights identical to a fresh seeded initialisation: {same} "
      f"(max |diff| {diff:.3g}); distance to the carbon specialist {dspec:.1f}")
