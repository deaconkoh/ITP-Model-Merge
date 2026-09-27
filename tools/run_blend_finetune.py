"""Fine-tune the three-specialist blend vs one specialist (pre-registered in results/blend_finetune.md, d05b2b6).

Step 4's own `train.Trainer` drives every run; only the policy module and the PPO update step are replaced.

    python tools/run_blend_finetune.py --arm {F1,F2,C2,C3,S4CHECK} --seed 111 [--max-updates 1000] [--eval-only]

Arms
  F1  blend of the seed's three specialists (alpha = B3 weights), all specialists trainable, fresh critic on the
      three pooled embeddings (stop-gradient), 50-update critic-only warm-up
  F2  same blend with the specialists FROZEN plus a zero-initialised residual r_theta(s, a), fresh critic, warm-up
  C2  makespan specialist, DANIEL critic re-initialised, warm-up (otherwise Step 4's makespan arm)
  C3  Step 4 merge composition, DANIEL critic re-initialised, warm-up (otherwise Step 4's merge arm)
  S4CHECK  makespan specialist with the REPLACED update step but no critic change and no warm-up: must reproduce
      Step 4's makespan arm (harness check; not an arm of the study)
After training, every snapshot is evaluated greedily on the 50 HELD-OUT instances of blend_zero_training.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from copy import deepcopy
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DAN = REPO / "daniel"
ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True, choices=["F1", "F2", "C2", "C3", "S4CHECK"])
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--max-updates", type=int, default=1000)
ap.add_argument("--eval-only", action="store_true")
ARGS = ap.parse_args()

TAG, WC, WT = "k125s060", 0.002384, 5.815362                      # Step 4's reward weights
BUDGETS = [0, 50, 100, 250, 500, 1000]
WARMUP = 50
B3 = {111: (100, 100, 800), 222: (300, 200, 500), 333: (600, 100, 300), 444: (200, 0, 800)}
SMOKE = os.environ.get("BFT_SMOKE") == "1"            # smoke runs use separate names and never touch real outputs
SUFFIX = f"bft_{ARGS.arm}_{TAG}_s{ARGS.seed}" + ("_SMOKE" if SMOKE else "")
NAME = f"10x5+carbon+priority+{SUFFIX}"
OUT = REPO / "results/blend_finetune"
merge_sel = json.loads((REPO / "results/step4/merge_init_selection.json").read_text())
INIT = {"C2": f"10x5+carbon+priority+m_s{ARGS.seed}", "S4CHECK": f"10x5+carbon+priority+m_s{ARGS.seed}",
        "C3": merge_sel[str(ARGS.seed)]["checkpoint"]}.get(ARGS.arm, "")
budgets = [b for b in BUDGETS if b <= ARGS.max_updates]

os.chdir(DAN)
sys.path.insert(0, str(DAN))
sys.argv = ["train.py", "--config", "../configs/canonical/mct.json",
            "--train_data_path", "./data/data_train/SD2/10x5+carbon+priority",
            "--validation_data_path", "./data/data_validation/SD2/10x5+carbon+priority",
            "--test_data_path", "./data/data_final_test/SD2/10x5+carbon+priority",
            "--n_j", "10", "--n_m", "5", "--data_suffix", "carbon+priority",
            "--carbon_reward_weight", str(WC), "--tardiness_reward_weight", str(WT),
            "--max_updates", str(ARGS.max_updates), "--device", "cuda",
            "--budget_checkpoints", ",".join(map(str, budgets)), "--seed_train", str(ARGS.seed),
            "--model_suffix", SUFFIX] + (["--init_from", INIT] if INIT else [])
import numpy as np                                              # noqa: E402
import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402
import torch.nn.functional as F                                 # noqa: E402
import train                                                    # noqa: E402  (parses the argv above)
from train import configs                                       # noqa: E402
from model.main_model import DANIEL                             # noqa: E402
from model.sub_layers import Critic                             # noqa: E402
from common_utils import eval_actions, load_checkpoint_state_dict, greedy_select_action   # noqa: E402
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums          # noqa: E402
from data_utils import load_priority_carbon_data_from_files     # noqa: E402
from due_dates import due_dates_for_directory, load_manifest    # noqa: E402

DEV = torch.device(configs.device)
CK = DAN / "trained_network/SD2"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------------------------------------- modules
def net_forward(net, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs):
    """DANIEL.forward_logits, line for line, also returning each candidate's actor input (action embedding)."""
    fea_j, fea_m, fea_j_global, fea_m_global = net.feature_exact(fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx)
    sz_b, M, _, J = comp_idx.size()
    d = fea_j.size(-1)
    candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
    candidate_idx = candidate_idx.type(torch.int64)
    Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)
    Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
    Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)
    Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
    Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
    fea_pairs = fea_pairs.reshape(sz_b, -1, net.pair_input_dim)
    candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input,
                                   Fea_Gm_input, fea_pairs), dim=-1)
    candidate_scores = net.actor(candidate_feature).squeeze(-1)
    candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')
    return candidate_scores, candidate_feature, torch.cat((fea_j_global, fea_m_global), dim=-1)


class BlendPolicy(nn.Module):
    """pi = softmax( sum_i alpha_i log p_i  [+ r_theta] ) over valid actions; value from a fresh critic on the
    three pooled embeddings (stop-gradient). The weighted sum uses blend_zero_training's B3 arithmetic."""

    def __init__(self, nets, alpha_permille, residual, critic):
        super().__init__()
        self.nets = nn.ModuleList(nets)
        self.coef = [float(torch.tensor(a / 1000.0, dtype=torch.float64).to(torch.float32)) for a in alpha_permille]
        self.critic = critic
        self.residual = residual

    def z(self, **kw):
        outs = [net_forward(n, **kw) for n in self.nets]
        lps = [F.log_softmax(s, dim=1) for s, _, _ in outs]
        invalid = torch.isinf(lps[0])
        for x in lps[1:]:
            invalid = invalid | torch.isinf(x)
        coef = [torch.tensor(c, dtype=torch.float32, device=lps[0].device) for c in self.coef]
        z = sum(coef[i] * torch.where(invalid, torch.zeros_like(x), x) for i, x in enumerate(lps))
        if self.residual is not None:
            inp = torch.cat([cf.detach() for _, cf, _ in outs]
                            + [torch.where(invalid, torch.zeros_like(x), x).detach().unsqueeze(-1) for x in lps], -1)
            z = z + self.residual(inp).squeeze(-1)
        z = torch.where(invalid, torch.full_like(z, float("-inf")), z)
        return z, torch.cat([g for _, _, g in outs], -1)

    def forward(self, **kw):
        z, pooled = self.z(**kw)
        return F.softmax(z, dim=1), self.critic(pooled.detach())


def residual_mlp(in_dim):
    m = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1))
    nn.init.zeros_(m[-1].weight); nn.init.zeros_(m[-1].bias)
    return m


def load_daniel(name):
    net = DANIEL(configs)
    net.load_state_dict(load_checkpoint_state_dict(CK / f"{name}.pth", map_location=configs.device,
                                                   expected_schema=configs.feature_schema))
    return net


def seeded(fn, salt):
    """Create a module under its own RNG so the global streams are untouched (train() reseeds anyway)."""
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(ARGS.seed * 1000 + salt)
        return fn()


def build_policy(trainer=None):
    """Returns (policy, critic_params, trainable_params, warmup)."""
    emb = 2 * configs.layer_fea_output_dim[-1]
    if ARGS.arm in ("F1", "F2"):
        nets = [load_daniel(f"10x5+carbon+priority+{p}") for p in (f"m_s{ARGS.seed}", f"c_s{ARGS.seed}",
                                                                 f"t_{TAG}_s{ARGS.seed}")]
        critic = seeded(lambda: Critic(configs.num_mlp_layers_critic, 3 * emb, configs.hidden_dim_critic, 1), 1)
        res = None
        if ARGS.arm == "F2":
            act_dim = 4 * configs.layer_fea_output_dim[-1] + configs.fea_pair_input_dim
            res = seeded(lambda: residual_mlp(3 * act_dim + 3), 2)
            for n in nets:
                n.requires_grad_(False)
        pol = BlendPolicy(nets, B3[ARGS.seed], res, critic).to(DEV)
        trainable = [p for p in pol.parameters() if p.requires_grad]
        return pol, list(pol.critic.parameters()), trainable, WARMUP
    pol = trainer.ppo.policy if trainer is not None else load_daniel(INIT)       # DANIEL loaded via --init_from
    if ARGS.arm in ("C2", "C3"):
        pol.critic = seeded(lambda: Critic(configs.num_mlp_layers_critic, emb, configs.hidden_dim_critic, 1), 1).to(DEV)
        return pol, list(pol.critic.parameters()), list(pol.parameters()), WARMUP
    return pol, list(pol.critic.parameters()), list(pol.parameters()), 0          # S4CHECK


class TunedPPO:
    """PPO.update copied line for line, plus a critic-only warm-up for the first `warmup` updates."""

    def __init__(self, policy, critic_params, trainable, warmup):
        c = configs
        self.lr, self.gamma, self.gae_lambda = c.lr, c.gamma, c.gae_lambda
        self.eps_clip, self.k_epochs, self.tau = c.eps_clip, c.k_epochs, c.tau
        self.ploss_coef, self.vloss_coef, self.entloss_coef = c.ploss_coef, c.vloss_coef, c.entloss_coef
        self.minibatch_size = c.minibatch_size
        self.policy = policy
        self.policy_old = deepcopy(policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.critic_params, self.trainable, self.warmup = critic_params, trainable, warmup
        cp = {id(p) for p in critic_params}
        self.non_critic = [p for p in trainable if id(p) not in cp]
        self.warm_opt = torch.optim.Adam(critic_params, lr=self.lr) if warmup else None
        self.optimizer = None if warmup else torch.optim.Adam(trainable, lr=self.lr)
        self.V_loss_2 = nn.MSELoss()
        self.n_updates = 0

    def update(self, memory):
        warm = self.n_updates < self.warmup
        if warm:
            for p in self.non_critic:
                p.requires_grad_(False)
            opt = self.warm_opt
        else:
            if self.optimizer is None:
                for p in self.non_critic:
                    p.requires_grad_(True)
                self.optimizer = torch.optim.Adam(self.trainable, lr=self.lr)
            opt = self.optimizer
        t_data = memory.transpose_data()
        t_advantage_seq, v_target_seq = memory.get_gae_advantages()
        full_batch_size = len(t_data[-1])
        num_batch = np.ceil(full_batch_size / self.minibatch_size)
        loss_epochs = 0
        v_loss_epochs = 0
        for _ in range(self.k_epochs):
            for i in range(int(num_batch)):
                if i + 1 < num_batch:
                    start_idx = i * self.minibatch_size
                    end_idx = (i + 1) * self.minibatch_size
                else:
                    start_idx = i * self.minibatch_size
                    end_idx = full_batch_size
                pis, vals = self.policy(fea_j=t_data[0][start_idx:end_idx],
                                        op_mask=t_data[1][start_idx:end_idx],
                                        candidate=t_data[6][start_idx:end_idx],
                                        fea_m=t_data[2][start_idx:end_idx],
                                        mch_mask=t_data[3][start_idx:end_idx],
                                        comp_idx=t_data[5][start_idx:end_idx],
                                        dynamic_pair_mask=t_data[4][start_idx:end_idx],
                                        fea_pairs=t_data[7][start_idx:end_idx])
                action_batch = t_data[8][start_idx: end_idx]
                logprobs, ent_loss = eval_actions(pis, action_batch)
                ratios = torch.exp(logprobs - t_data[12][start_idx: end_idx].detach())
                advantages = t_advantage_seq[start_idx: end_idx]
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals.squeeze(1), v_target_seq[start_idx: end_idx])
                p_loss = - torch.min(surr1, surr2)
                ent_loss = - ent_loss.clone()
                if warm:
                    loss = self.vloss_coef * v_loss                       # warm-up: value loss only
                else:
                    loss = self.vloss_coef * v_loss + self.ploss_coef * p_loss + self.entloss_coef * ent_loss
                opt.zero_grad()
                loss_epochs += loss.mean().detach()
                v_loss_epochs += v_loss.mean().detach()
                loss.mean().backward()
                opt.step()
        for policy_old_params, policy_params in zip(self.policy_old.parameters(), self.policy.parameters()):
            policy_old_params.data.copy_(self.tau * policy_old_params.data + (1 - self.tau) * policy_params.data)
        self.n_updates += 1
        return loss_epochs.item() / self.k_epochs, v_loss_epochs.item() / self.k_epochs


# ---------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(policy, data, due, idx):
    """Greedy, HELD-OUT instances batched by equal maximum processing time. Blends decode as B3 did."""
    policy.eval()
    ub = np.array([float(data[1][i][data[1][i] > 0].max()) for i in idx])
    out = np.zeros((3, len(idx)))
    for u in np.unique(ub):
        r = np.flatnonzero(ub == u); ii = [idx[k] for k in r]
        env = FJSPEnvForSameOpNums(n_j=data[0][0].shape[0], n_m=data[1][0].shape[1])
        state = env.set_initial_data([data[0][i] for i in ii], [data[1][i] for i in ii], [data[2][i] for i in ii],
                                     [data[3][i] for i in ii], due_date_list=[due[i] for i in ii])
        while True:
            kw = dict(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor, candidate=state.candidate_tensor,
                      fea_m=state.fea_m_tensor, mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                      dynamic_pair_mask=state.dynamic_pair_mask_tensor, fea_pairs=state.fea_pairs_tensor)
            if isinstance(policy, BlendPolicy):
                a = F.log_softmax(policy.z(**kw)[0], dim=1).argmax(1)
            else:
                a = greedy_select_action(policy(**kw)[0])
            state, _, done = env.step(actions=a.cpu().numpy())
            if np.all(done):
                break
        comp = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
        out[:, r] = env.current_makespan, env.total_carbon, np.maximum(0, comp - np.array([due[i] for i in ii])).sum(1)
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rawdir = OUT / ("raw_smoke" if SMOKE else "raw")
    rawdir.mkdir(parents=True, exist_ok=True)
    res_file = rawdir / f"{ARGS.arm}_s{ARGS.seed}.npz"
    t_train = 0.0
    if not ARGS.eval_only:
        trainer = train.Trainer(configs)
        policy, cparams, trainable, warmup = build_policy(trainer)
        trainer.ppo = TunedPPO(policy, cparams, trainable, warmup)
        n_train = sum(p.numel() for p in trainable)
        log(f"{ARGS.arm} s{ARGS.seed}: trainable parameters {n_train:,}; critic {sum(p.numel() for p in cparams):,}; "
            f"warm-up {warmup} updates")
        torch.cuda.synchronize(); t0 = time.time()
        trainer.train()
        torch.cuda.synchronize(); t_train = time.time() - t0
        log(f"TRAINING_DONE {ARGS.arm} s{ARGS.seed}: {t_train / 60:.1f} min")
    else:
        n_train = None
    # ---- evaluate every snapshot on HELD-OUT
    split = json.loads((REPO / "results/blend_zero_training/split.json").read_text())
    held = split["held_idx"]
    pool = DAN / "data/data_train_vali/SD2/10x5+carbon+priority"
    data = load_priority_carbon_data_from_files(str(pool))
    due = due_dates_for_directory(pool, load_manifest())
    t0 = time.time()
    res, shas = {}, {}
    for b in budgets:
        ck = CK / f"{NAME}@u{b}.pth"
        sd = load_checkpoint_state_dict(ck, map_location=configs.device, expected_schema=configs.feature_schema) \
            if ARGS.arm not in ("F1", "F2") else torch.load(ck, map_location=configs.device)["state_dict"]
        policy, *_ = build_policy(None)
        policy.load_state_dict(sd)
        res[f"u{b}"] = evaluate(policy.to(DEV), data, due, held)
        shas[b] = sha(ck)
        log(f"  eval @u{b}: makespan {res[f'u{b}'][0].mean():.1f} carbon {res[f'u{b}'][1].mean():.1f} "
            f"tardiness {res[f'u{b}'][2].mean():.1f}")
    t_eval = time.time() - t0
    # ---- pre-registered update-0 checks (the study stops if one fails)
    if ARGS.arm != "S4CHECK":
        z = np.load(REPO / "results/blend_zero_training/raw" / f"seed{ARGS.seed}.npz")
        if ARGS.arm in ("F1", "F2"):
            g = int(np.flatnonzero((z["grid"] == np.array(B3[ARGS.seed])).all(1))[0]); ref = z["G"][:, g, held]
        else:
            ref = z["R_spec_m@0" if ARGS.arm == "C2" else "R_merge@0"][:, held]
        if not np.array_equal(res["u0"], ref):
            log(f"UPDATE-0 CHECK FAILED for {ARGS.arm} s{ARGS.seed}: max|diff| {np.abs(res['u0'] - ref).max()}")
            raise SystemExit(3)
        log(f"update-0 check passed: {ARGS.arm}@0 reproduces its reference exactly")
    np.savez_compressed(res_file, **res)
    meta = {"arm": ARGS.arm, "seed": ARGS.seed, "model_name": NAME, "init_from": INIT or "three specialists",
            "b3_alpha_permille": B3[ARGS.seed] if ARGS.arm in ("F1", "F2") else None,
            "trainable_parameters": n_train, "train_seconds": round(t_train, 1), "eval_seconds": round(t_eval, 1),
            "snapshot_sha256": shas, "argv": sys.argv}
    prev = rawdir / f"{ARGS.arm}_s{ARGS.seed}.json"
    if ARGS.eval_only and prev.exists():
        old = json.loads(prev.read_text()); meta["train_seconds"] = old.get("train_seconds"); meta["trainable_parameters"] = old.get("trainable_parameters")
    prev.write_text(json.dumps(meta, indent=1))
    log(f"UNIT_DONE {ARGS.arm} s{ARGS.seed}: train {t_train / 60:.1f} min, eval {t_eval / 60:.2f} min")


if __name__ == "__main__":
    main()
