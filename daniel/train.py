from common_utils import *
from params import configs
from tqdm import tqdm
from data_utils import load_data_from_files, load_priority_carbon_data_from_files, CaseGenerator, SD2_instance_generator
from common_utils import strToSuffix, setup_seed
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
import os
import random
import time
import sys
import json
from pathlib import Path
from model.PPO import PPO_initialize
from model.PPO import Memory
from objectives import ObjectiveSpec, environment_metrics
from feature_schemas import validate_config_against_schema
from data_manifest import assert_disjoint_splits
from checkpointing import (file_sha256, json_safe_config, make_checkpoint_bundle,
                           unwrap_checkpoint, validate_checkpoint_schema, git_state,
                           runtime_environment)

str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)


class Trainer:
    def __init__(self, config):

        self.feature_schema = validate_config_against_schema(config)
        self.objective = ObjectiveSpec(
            config.goal,
            carbon_weight=config.carbon_reward_weight,
            priority_weight=config.priority_reward_weight,
            tardiness_weight=config.tardiness_reward_weight,
        )

        self.n_j = config.n_j
        self.n_m = config.n_m
        self.low = config.low
        self.high = config.high
        self.op_per_job_min = int(0.8 * self.n_m)
        self.op_per_job_max = int(1.2 * self.n_m)
        self.data_source = config.data_source
        self.data_root = config.data_root
        self.config = config
        self.max_updates = config.max_updates
        self.reset_env_timestep = config.reset_env_timestep
        self.validate_timestep = config.validate_timestep
        self.num_envs = config.num_envs

        if not os.path.exists(f'./trained_network/{self.data_source}'):
            os.makedirs(f'./trained_network/{self.data_source}')
        if not os.path.exists(f'./train_log/{self.data_source}'):
            os.makedirs(f'./train_log/{self.data_source}')

        if device.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if self.data_source == 'SD1':
            self.data_name = f'{self.n_j}x{self.n_m}'
        elif self.data_source == 'SD2':
            self.data_name = f'{self.n_j}x{self.n_m}{strToSuffix(config.data_suffix)}'

        # benchmark v3 (docs/benchmark_v3.md): plain SD2 files + frozen hashed parameter manifest
        self.v3 = config.obs_version == "v3"
        if self.v3:
            import benchmark_v3
            if self.data_source != 'SD2':
                raise ValueError("benchmark v3 is defined for SD2 only")
            v3_split = lambda split: str(benchmark_v3.ROOT / split / self.data_source / self.data_name)
        self.train_data_path = config.train_data_path or \
            (v3_split("train") if self.v3 else f'{self.data_root}/data_train/{self.data_source}/{self.data_name}')
        self.vali_data_path = config.validation_data_path or \
            (v3_split("validation") if self.v3 else
             f'{self.data_root}/data_validation/{self.data_source}/{self.data_name}')
        # Retained only as provenance. Training deliberately never reads final-test data.
        self.test_data_path = config.test_data_path or \
            f'{self.data_root}/data_final_test/{self.data_source}/{self.data_name}'
        self.model_name = f'{self.data_name}{strToSuffix(config.model_suffix)}'

        self.data_fingerprints = assert_disjoint_splits(
            train=self.train_data_path,
            validation=self.vali_data_path,
        )

        # seed
        self.seed_train = config.seed_train
        self.seed_test = config.seed_test
        setup_seed(self.seed_train)

        self.env = FJSPEnvForSameOpNums(self.n_j, self.n_m)
        self.uses_priority_carbon_files = (self.config.enable_priority or self.config.enable_carbon) and 'carbon+priority' in self.data_name
        if self.uses_priority_carbon_files:
            self.file_train_data = load_priority_carbon_data_from_files(self.train_data_path)
        else:
            self.file_train_data = load_data_from_files(self.train_data_path) if self.config.train_from_files else None
        # validation data set
        if self.uses_priority_carbon_files:
            vali_data = load_priority_carbon_data_from_files(self.vali_data_path)
        else:
            vali_data = load_data_from_files(self.vali_data_path)
        if not vali_data[0]:
            raise ValueError(f"No validation instances found at {self.vali_data_path}")

        if self.data_source == 'SD1':
            self.vali_env = FJSPEnvForVariousOpNums(self.n_j, self.n_m)
        elif self.data_source == 'SD2':
            self.vali_env = FJSPEnvForSameOpNums(self.n_j, self.n_m)

        # Due dates are READ from the frozen manifest (hash-checked), never derived here, and only
        # for tardiness goals -- every other goal trains exactly as before.
        self.train_due, vali_due, self.due_date_provenance = None, None, None
        # v2 observations need due dates for every goal (results/v2_fixes.md)
        if not self.v3 and (self.objective.uses_tardiness or config.obs_version == "v2"):
            from due_dates import (MANIFEST, load_manifest, due_dates_for_directory,
                                   manifest_tag, file_sha256)
            manifest_path = Path(config.due_date_manifest or MANIFEST)
            dd_manifest = load_manifest(manifest_path)
            tag = manifest_tag(self.train_data_path, dd_manifest)
            if manifest_tag(self.vali_data_path, dd_manifest) != tag:
                raise ValueError("training and validation sizes map to different due-date tags")
            if tag not in (config.model_suffix or ""):
                raise ValueError(f"tardiness checkpoints must carry the due-date tag {tag!r} in "
                                 f"--model_suffix (got {config.model_suffix!r})")
            self.train_due = due_dates_for_directory(self.train_data_path, dd_manifest)
            vali_due = due_dates_for_directory(self.vali_data_path, dd_manifest)
            self.due_date_provenance = {
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": file_sha256(manifest_path),
                "dd_tag": tag, "s": dd_manifest["s"],
                "k": dd_manifest["k_by_size"][Path(self.train_data_path).name.split("+")[0]],
            }

        # v2 reward: frozen, hash-checked references (results/v2_fixes.md, A3); None under v1
        self.train_refs, vali_refs, self.reward_ref_provenance = None, None, None
        if config.reward_version == "v2":
            import reward_refs as RR
            ref_path = Path(config.reward_ref_manifest or RR.MANIFEST)
            rr = RR.load_manifest(ref_path)
            ref_seed = config.reward_ref_seed if config.reward_ref_seed is not None else config.seed_train
            train_pairs, tbar = RR.refs_for_directory(self.train_data_path, rr, ref_seed)
            vali_pairs, _ = RR.refs_for_directory(self.vali_data_path, rr, ref_seed)
            self.train_refs = {"goal": config.goal, "tbar": tbar, "pairs": train_pairs}
            vali_refs = {"goal": config.goal, "tbar": tbar, "mk": [p[0] for p in vali_pairs],
                         "cb": [p[1] for p in vali_pairs]}
            self.reward_ref_provenance = {"manifest": str(ref_path.resolve()), "manifest_sha256": RR.file_sha256(ref_path),
                                          "seed": ref_seed, "tbar": tbar}

        self.benchmark_provenance = None
        if self.v3:
            self.init_benchmark_v3(config)
        elif self.uses_priority_carbon_files:
            self.vali_env.set_initial_data(vali_data[0], vali_data[1], vali_data[2], vali_data[3],
                                           due_date_list=vali_due, reward_refs=vali_refs)
        else:
            self.vali_env.set_initial_data(vali_data[0], vali_data[1])

        self.ppo = PPO_initialize()
        self.parent_checkpoints = []
        self.load_initial_checkpoint()
        self.save_run_manifest()
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)

    def init_benchmark_v3(self, config):
        """Benchmark v3 train/validation data with frozen carbon, due dates and score references (hash-checked)."""
        import benchmark_v3
        if config.reward_version != "v3":
            raise ValueError("benchmark v3 observations need --reward_version v3")
        if benchmark_v3.TAG not in (config.model_suffix or ""):
            raise ValueError(f"benchmark v3 checkpoints must carry the tag {benchmark_v3.TAG!r} in --model_suffix "
                             f"(got {config.model_suffix!r})")
        benchmark_v3.goal_weights(config.goal)                  # m, c, t combinations only
        manifest_path = Path(config.benchmark_manifest or benchmark_v3.MANIFEST)
        manifest = benchmark_v3.load_manifest(manifest_path)
        self.v3_train = benchmark_v3.load_directory(self.train_data_path, manifest)
        v3_vali = benchmark_v3.load_directory(self.vali_data_path, manifest)
        if self.v3_train.size != v3_vali.size:
            raise ValueError("benchmark v3 training and validation sizes differ")
        self.mo = bool(getattr(config, "preference_dim", 0))
        if self.mo:
            self.init_mo_validation(config, v3_vali)
        else:
            self.vali_env.set_initial_data(**v3_vali.env_kwargs(range(len(v3_vali)), config.goal))
        self.v3_generator = None
        if config.v3_train_source == "generator":
            # original DANIEL scheme: fresh SD2 instances, v3 parameters from the frozen code (training-data decision)
            from benchmark_v3_generator import GeneratorSource
            self.v3_generator = GeneratorSource(config, self.v3_train.size, manifest)
        self.benchmark_provenance = {
            "train_source": config.v3_train_source,
            "version": manifest["version"], "manifest": str(manifest_path.resolve()),
            "manifest_sha256": benchmark_v3.file_sha256(manifest_path), "size": self.v3_train.size,
            "due_k": manifest["sizes"][self.v3_train.size]["k"], "due_s": manifest["sizes"][self.v3_train.size]["s"],
            "t_ref": self.v3_train.t_ref, "n_train": len(self.v3_train), "n_validation": len(v3_vali),
        }

    def init_mo_validation(self, config, v3_vali):
        """MO-DANIEL validation (docs/evaluation_protocol_v3.md, Addendum A): every validation instance under every
        preference of the coarse simplex grid, greedy; metric = mean normalised hypervolume over instances."""
        import mo_metrics
        if not config.mo_val_norm:
            raise ValueError("MO validation needs --mo_val_norm (frozen validation ideal/nadir points)")
        norm_path = Path(config.mo_val_norm).resolve()               # relative to the working directory
        norm = json.loads(norm_path.read_text())["instances"]
        n = len(v3_vali) if config.mo_val_first <= 0 else min(config.mo_val_first, len(v3_vali))
        self.mo_grid = mo_metrics.simplex_grid(config.mo_val_grid_step)
        self.mo_val_keys = v3_vali.keys[:n]
        missing = [k for k in self.mo_val_keys if k not in norm]
        if missing:
            raise KeyError(f"{len(missing)} validation instances lack normalisation points in {norm_path}")
        self.mo_val_norm = [mo_metrics.ideal_nadir(norm[k]) for k in self.mo_val_keys]
        g = len(self.mo_grid)
        idx = [i for i in range(n) for _ in range(g)]                     # instance-major, preference-minor
        self.vali_env.set_initial_data(**v3_vali.env_kwargs(idx, config.goal))
        self.vali_env.set_preference(np.tile(self.mo_grid, (n, 1)))
        self.mo_provenance = {"validation_norm": str(norm_path), "validation_norm_sha256": file_sha256(norm_path),
                              "grid_step": config.mo_val_grid_step, "grid_size": g, "instances": n,
                              "pref_dirichlet": config.pref_dirichlet}

    def validate_mo(self):
        """Greedy rollouts of all (validation instance, grid preference) pairs; per instance the non-dominated set of its
        schedules, normalised hypervolume (reference 1.1); returns the mean and per-objective means."""
        import mo_metrics
        self.ppo.policy.eval()
        state = self.vali_env.reset()
        while True:
            with torch.no_grad():
                pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor, op_mask=state.op_mask_tensor,
                                        candidate=state.candidate_tensor, fea_m=state.fea_m_tensor,
                                        mch_mask=state.mch_mask_tensor, comp_idx=state.comp_idx_tensor,
                                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                        fea_pairs=state.fea_pairs_tensor, pref=state.pref_tensor)
            state, _, done = self.vali_env.step(greedy_select_action(pi).cpu().numpy())
            if done.all():
                break
        self.ppo.policy.train()
        env = self.vali_env
        comp = env.true_op_ct[env.env_job_idx, env.job_last_op_id]
        tard = np.maximum(0.0, comp - env.true_due_dates).sum(1)
        obj = np.stack([env.current_makespan, env.total_carbon, tard], 1).reshape(len(self.mo_val_keys),
                                                                                    len(self.mo_grid), 3)
        hv = [mo_metrics.instance_hv(obj[i], *self.mo_val_norm[i]) for i in range(len(obj))]
        w = env.pref_weights
        score = (w[:, 0] * env.current_makespan / env.v2_mk_ref + w[:, 1] * env.total_carbon / env.v2_cb_ref
                 + w[:, 2] * tard / env.reward_v2.tbar)
        return {"hv": float(np.mean(hv)), "hv_per_instance": [float(h) for h in hv],
                "makespan": float(obj[..., 0].mean()), "carbon": float(obj[..., 1].mean()),
                "tardiness": float(obj[..., 2].mean()), "score_pref_mean": float(score.mean())}

    def sample_preferences(self):
        """Training preferences: w ~ Dirichlet(alpha, alpha, alpha), one per environment, from a dedicated generator."""
        a = self.config.pref_dirichlet
        return self.pref_rng.dirichlet([a, a, a], size=self.num_envs)

    def save_run_manifest(self):
        manifest = {
            "schema_version": 2,
            "status": "INITIALIZED",
            "model_name": self.model_name,
            "config": json_safe_config(self.config),
            "data_fingerprints": self.data_fingerprints,
            "final_test_path_not_loaded": str(Path(self.test_data_path).resolve()),
            "due_dates": getattr(self, "due_date_provenance", None),
            "reward_refs": getattr(self, "reward_ref_provenance", None),
            "benchmark": getattr(self, "benchmark_provenance", None),
            "mo": getattr(self, "mo_provenance", None),
            "parents": self.parent_checkpoints,
            "git": git_state(Path(__file__).resolve().parents[1]),
            "runtime": runtime_environment(),
        }
        path = Path(f'./train_log/{self.data_source}/{self.model_name}.run.json')
        path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    def train(self):
        """
            train the model following the config
        """
        setup_seed(self.seed_train)
        self.log = []
        self.validation_log = []
        self.record = float('inf')
        self.best_update = None
        self.train_seconds = 0.0                 # wall clock inside the update loop, validation included
        self.run_status = "RUNNING"
        chunk = self.config.chunk_updates
        if chunk > 0 and (chunk % self.reset_env_timestep or chunk % self.validate_timestep):
            raise ValueError("--chunk_updates must be a multiple of --reset_env_timestep and --validate_timestep")
        if getattr(self, "mo", False):
            # dedicated stream: the instance generator's global numpy stream is the same as the specialists'
            self.pref_rng = np.random.default_rng([self.seed_train, 20261019])
        start_update = self.load_resume_state() if self.config.resume else 0

        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"train data :{self.train_data_path}")
        print(f"vali data :{self.vali_data_path}")
        print(f"final test data (not loaded) :{self.test_data_path}")
        print(f"goal :{self.config.goal}")
        print("\n")

        self.train_st = time.time()

        # RQ1 budget curve: snapshot the policy at fixed update counts so that one run
        # yields the whole quality-vs-budget curve. Independent of the validation-improvement
        # checkpoint, which is untouched.
        self.budget_checkpoints = sorted(
            {int(v) for v in str(self.config.budget_checkpoints).split(',') if v.strip()}
        )
        if 0 in self.budget_checkpoints:
            self.save_budget_checkpoint(0)

        if self.config.converge_window > 0 and start_update == 0:
            t0 = time.time()
            self.validate_and_select(0, t0)
            self.train_seconds += time.time() - t0
        stop_update = self.max_updates if chunk <= 0 else min(self.max_updates, start_update + chunk)
        next_update = start_update

        for i_update in tqdm(range(start_update, stop_update), file=sys.stdout, desc="progress", colour='blue'):
            if self.run_status == "CONVERGED":
                break
            ep_st = time.time()

            # resampling the training data
            if i_update % self.reset_env_timestep == 0 and self.v3 and self.v3_generator is not None:
                batch = self.v3_generator.sample(self.num_envs)
                state = self.env.set_initial_data(**batch.env_kwargs(range(self.num_envs), self.config.goal))
            elif i_update % self.reset_env_timestep == 0 and self.v3:
                idxs = [random.randrange(len(self.v3_train)) for _ in range(self.num_envs)]
                self.last_sample_idxs = idxs
                state = self.env.set_initial_data(**self.v3_train.env_kwargs(idxs, self.config.goal))
            elif i_update % self.reset_env_timestep == 0:
                sampled_data = self.sample_training_instances()
                if len(sampled_data) == 4:
                    dataset_job_length, dataset_op_pt, dataset_op_priority, dataset_op_carbon = sampled_data
                    due = ([self.train_due[i] for i in self.last_sample_idxs]
                           if self.train_due is not None else None)
                    refs = None
                    if self.train_refs is not None:
                        pairs = [self.train_refs["pairs"][i] for i in self.last_sample_idxs]
                        refs = {"goal": self.train_refs["goal"], "tbar": self.train_refs["tbar"],
                                "mk": [p[0] for p in pairs], "cb": [p[1] for p in pairs]}
                    state = self.env.set_initial_data(dataset_job_length, dataset_op_pt,
                                                      dataset_op_priority, dataset_op_carbon,
                                                      due_date_list=due, reward_refs=refs)
                elif len(sampled_data) == 3:
                    dataset_job_length, dataset_op_pt, dataset_op_priority = sampled_data
                    state = self.env.set_initial_data(dataset_job_length, dataset_op_pt, dataset_op_priority)
                else:
                    dataset_job_length, dataset_op_pt = sampled_data
                    state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
            else:
                state = self.env.reset()
            if getattr(self, "mo", False):
                prefs = self.sample_preferences()
                state = self.env.set_preference(prefs)

            # Log the reward actually supplied to PPO.  The previous makespan
            # offset made carbon/priority run logs contain an unrelated term.
            ep_rewards = np.zeros(self.env.number_of_envs, dtype=np.float64)

            while True:

                # state store
                self.memory.push(state)
                with torch.no_grad():

                    pi_envs, vals_envs = self.ppo.policy_old(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                                                             op_mask=state.op_mask_tensor,  # [sz_b, N, N]
                                                             candidate=state.candidate_tensor,  # [sz_b, J]
                                                             fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                                                             mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                                                             comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                                                             dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                                                             fea_pairs=state.fea_pairs_tensor,  # [sz_b, J, M]
                                                             pref=state.pref_tensor)  # MO-DANIEL only

                # sample the action
                action_envs, action_logprob_envs = sample_action(pi_envs)

                # state transition
                state, reward, done = self.env.step(actions=action_envs.cpu().numpy())
                ep_rewards += reward
                reward = torch.from_numpy(reward).to(device)

                # collect the transition
                self.memory.done_seq.append(torch.from_numpy(done).to(device))
                self.memory.reward_seq.append(reward)
                self.memory.action_seq.append(action_envs)
                self.memory.log_probs.append(action_logprob_envs)
                self.memory.val_seq.append(vals_envs.squeeze(1))

                if done.all():
                    break

            loss, v_loss = self.ppo.update(self.memory)
            self.memory.clear_memory()

            mean_rewards_all_env = np.mean(ep_rewards)
            mean_makespan_all_env = np.mean(self.env.current_makespan)
            mean_carbon_all_env = np.mean(self.env.total_carbon)

            # save the mean rewards of all instances in current training data
            entry = {"update": i_update, "mean_return": float(mean_rewards_all_env)}
            if self.ppo.last_stats is not None:
                entry["ppo"] = self.ppo.last_stats
            if getattr(self, "mo", False):
                entry["pref"] = {"mean": prefs.mean(0).tolist(), "min": prefs.min(0).tolist(),
                                 "max": prefs.max(0).tolist()}
                entry["objectives"] = {"makespan": float(np.mean(self.env.current_makespan)),
                                       "carbon": float(np.mean(self.env.total_carbon)),
                                       "tardiness": float(np.mean(np.maximum(
                                           0.0, self.env.true_op_ct[self.env.env_job_idx, self.env.job_last_op_id]
                                           - self.env.true_due_dates).sum(1)))}
            self.log.append(entry)

            if (i_update + 1) in self.budget_checkpoints:
                self.save_budget_checkpoint(i_update + 1)

            # validate the trained model
            if (i_update + 1) % self.validate_timestep == 0:
                self.validate_and_select(i_update + 1, ep_st)

            ep_et = time.time()
            self.train_seconds += ep_et - ep_st
            next_update = i_update + 1
            
            # print the reward, makespan, loss and training time of the current episode
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    i_update + 1, mean_rewards_all_env, mean_makespan_all_env, loss, ep_et - ep_st))
            if self.config.enable_carbon:
                tqdm.write(f'Episode {i_update + 1}\t carbon: {mean_carbon_all_env:.2f}')

        self.train_et = time.time()

        finished = self.run_status == "CONVERGED" or next_update >= self.max_updates
        if finished and self.config.converge_window > 0 and self.run_status != "CONVERGED":
            self.run_status = "NOT_CONVERGED"            # hit the cap: reported, never silently
        elif finished and self.run_status == "RUNNING":
            self.run_status = "FINISHED"
        if chunk > 0 or self.config.resume:
            self.save_resume_state(next_update)
        self.write_status(next_update)
        with open(f'./train_log/{self.data_source}/reward_{self.model_name}.json', 'w', encoding='utf-8') as handle:
            json.dump(self.log, handle, indent=2)
        if not finished:
            return                                       # chunk done; the next process resumes

        # log results
        self.save_training_log()

    def validate_and_select(self, update, ep_st):
        """Greedy validation, best-checkpoint selection and (if enabled) the convergence rule."""
        if getattr(self, "mo", False):
            validation_metrics = self.validate_mo()
        elif self.data_source == "SD1":
            validation_metrics = self.validate_envs_with_various_op_nums()
        else:
            validation_metrics = self.validate_envs_with_same_op_nums()

        if getattr(self, "mo", False):
            selection_metric = -validation_metrics["hv"]           # lower is better: minus the validation hypervolume
        elif self.config.reward_version in ("v2", "v3"):
            # mean normalised score (= -mean return): Step 4 score (v2) or benchmark v3 score
            selection_metric = validation_metrics[f"score_{self.config.reward_version}"]
        else:
            selection_metric = self.objective.selection_metric(validation_metrics)
        if selection_metric < self.record:
            self.save_model(update, validation_metrics, selection_metric)
            self.record = selection_metric
            self.best_update = update

        entry = {"update": update, **validation_metrics, "selection_metric": selection_metric,
                 "train_seconds": self.train_seconds + (time.time() - ep_st)}
        if self.config.converge_window > 0:
            from convergence import is_converged
            converged, rel = is_converged([(e["update"], e["selection_metric"]) for e in self.validation_log + [entry]],
                                          self.config.converge_window, self.config.converge_tol,
                                          self.config.min_updates)
            entry["rel_improvement_window"] = rel
            if converged:
                self.run_status = "CONVERGED"
        self.validation_log.append(entry)
        self.save_validation_log()
        self.write_status(update)
        third = (f"tardiness={validation_metrics['tardiness']:.6g}; " if self.v3 else
                 f"operation-priority={validation_metrics['priority']:.6g}; ")
        tqdm.write(
            f"Validation makespan={validation_metrics['makespan']:.6g}; "
            f"carbon={validation_metrics['carbon']:.6g}; "
            + third +
            f"selection={selection_metric:.6g} (best={self.record:.6g})"
        )

    def resume_path(self):
        return Path(f'./trained_network/{self.data_source}/{self.model_name}.resume.pth')

    def run_identity(self):
        c = self.config
        return {"model_name": self.model_name, "goal": c.goal, "seed_train": c.seed_train, "n_j": c.n_j, "n_m": c.n_m,
                "obs_version": c.obs_version, "reward_version": c.reward_version,
                "v3_train_source": c.v3_train_source, "num_envs": c.num_envs, "lr": c.lr,
                "reset_env_timestep": c.reset_env_timestep, "validate_timestep": c.validate_timestep,
                "converge_window": c.converge_window, "converge_tol": c.converge_tol, "min_updates": c.min_updates,
                "max_updates": c.max_updates,
                **({"init_from": c.init_from, "init_mode": c.init_mode} if c.init_from else {}),
                **({"preference_dim": c.preference_dim, "pref_dirichlet": c.pref_dirichlet,
                    "mo_val_grid_step": c.mo_val_grid_step, "mo_val_first": c.mo_val_first,
                    "mo_val_norm": c.mo_val_norm, "log_ppo_stats": c.log_ppo_stats}
                   if getattr(c, "preference_dim", 0) else {})}

    def write_status(self, update):
        last = self.validation_log[-1] if self.validation_log else {}
        status = {"model_name": self.model_name, "goal": self.config.goal, "update": update,
                  "max_updates": self.max_updates, "status": self.run_status, "best_value": self.record,
                  "best_update": self.best_update, "latest_value": last.get("selection_metric"),
                  "rel_improvement_window": last.get("rel_improvement_window"), "train_seconds": self.train_seconds,
                  "generator": self.v3_generator.provenance() if getattr(self, "v3_generator", None) else None,
                  "written": time.strftime("%Y-%m-%d %H:%M:%S")}
        Path(f'./train_log/{self.data_source}/{self.model_name}.status.json').write_text(json.dumps(status, indent=1))

    def save_resume_state(self, next_update):
        """Everything needed to continue bit-for-bit: weights, optimiser, RNG states, logs, best checkpoint."""
        best = Path(f'./trained_network/{self.data_source}/{self.model_name}.pth')
        gen = getattr(self, "v3_generator", None)
        state = {"identity": self.run_identity(), "next_update": next_update,
                 "policy": self.ppo.policy.state_dict(), "policy_old": self.ppo.policy_old.state_dict(),
                 "optimizer": self.ppo.optimizer.state_dict(),
                 "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                         "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                 "record": self.record, "best_update": self.best_update, "run_status": self.run_status,
                 "log": self.log, "validation_log": self.validation_log, "train_seconds": self.train_seconds,
                 "best_bundle": torch.load(best, map_location="cpu", weights_only=False) if best.exists() else None,
                 "generator": (gen.generated, gen.collisions) if gen else None,
                 "pref_rng": self.pref_rng.bit_generator.state if getattr(self, "mo", False) else None}
        path = self.resume_path(); tmp = path.with_suffix(".tmp")
        torch.save(state, tmp)
        os.replace(tmp, path)

    def load_resume_state(self):
        path = self.resume_path()
        if not path.exists():
            return 0
        state = torch.load(path, map_location=device, weights_only=False)
        if state["identity"] != self.run_identity():
            raise ValueError(f"resume state {path} belongs to a different run configuration")
        self.ppo.policy.load_state_dict(state["policy"])
        self.ppo.policy_old.load_state_dict(state["policy_old"])
        self.ppo.optimizer.load_state_dict(state["optimizer"])
        self.record, self.best_update = state["record"], state["best_update"]
        self.run_status, self.train_seconds = state["run_status"], state["train_seconds"]
        self.log, self.validation_log = state["log"], state["validation_log"]
        if state["best_bundle"] is not None:   # discard any best checkpoint written by an interrupted chunk
            torch.save(state["best_bundle"], f'./trained_network/{self.data_source}/{self.model_name}.pth')
        if state["generator"] is not None:
            self.v3_generator.generated, self.v3_generator.collisions = state["generator"]
        if state.get("pref_rng") is not None:
            self.pref_rng.bit_generator.state = state["pref_rng"]
        self.save_validation_log()
        rng = state["rng"]
        random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu() for r in rng["cuda"]])
        print(f"resumed {self.model_name} at update {state['next_update']} ({self.run_status})")
        return state["next_update"]

    def save_training_log(self):
        """
            save reward data & validation makespan data (during training) and the entire training time
        """
        with open(f'./train_log/{self.data_source}/reward_{self.model_name}.json', 'w', encoding='utf-8') as handle:
            json.dump(self.log, handle, indent=2)

        with open(f'./train_log/{self.data_source}/valiquality_{self.model_name}.json', 'w', encoding='utf-8') as handle:
            json.dump(self.validation_log, handle, indent=2)

        with open('./train_time.txt', 'a', encoding='utf-8') as handle:
            handle.write(
                f'model path: ./DANIEL_FJSP/trained_network/{self.data_source}/{self.model_name}\t\ttraining time: '
                f'{round((self.train_et - self.train_st), 2)}\t\t local time: {str_time}\n')

    def save_validation_log(self):
        """
            save the results of validation
        """
        with open(f'./train_log/{self.data_source}/valiquality_{self.model_name}.json', 'w', encoding='utf-8') as handle:
            json.dump(self.validation_log, handle, indent=2)

    def sample_training_instances(self):
        """
            sample training instances following the config, 
            the sampling process of SD1 data is imported from "songwenas12/fjsp-drl" 
        :return: new training instances
        """
        prepare_JobLength = [random.randint(self.op_per_job_min, self.op_per_job_max) for _ in range(self.n_j)]
        dataset_JobLength = []
        dataset_OpPT = []
        dataset_OpPriority = []
        if self.config.train_from_files:
            if self.file_train_data is None:
                self.file_train_data = load_data_from_files(self.train_data_path)
            source_count = len(self.file_train_data[0])
            if source_count == 0:
                raise ValueError(f"No file training data found at {self.train_data_path}")
            sample_idxs = [random.randrange(source_count) for _ in range(self.num_envs)]
            self.last_sample_idxs = sample_idxs
            for idx in sample_idxs:
                dataset_JobLength.append(self.file_train_data[0][idx])
                dataset_OpPT.append(self.file_train_data[1][idx])
                if len(self.file_train_data) >= 3:
                    dataset_OpPriority.append(self.file_train_data[2][idx])
            if self.uses_priority_carbon_files and len(self.file_train_data) >= 4:
                dataset_OpCarbon = [self.file_train_data[3][idx] for idx in sample_idxs]
                return dataset_JobLength, dataset_OpPT, dataset_OpPriority, dataset_OpCarbon
            if dataset_OpPriority:
                return dataset_JobLength, dataset_OpPT, dataset_OpPriority
            return dataset_JobLength, dataset_OpPT

        for i in range(self.num_envs):
            if self.data_source == 'SD1':
                case = CaseGenerator(self.n_j, self.n_m, self.op_per_job_min, self.op_per_job_max,
                                     nums_ope=prepare_JobLength, path='./test', flag_doc=False)
                JobLength, OpPT, _ = case.get_case(i)

            else:
                JobLength, OpPT, _ = SD2_instance_generator(config=self.config)
            dataset_JobLength.append(JobLength)
            dataset_OpPT.append(OpPT)

        return dataset_JobLength, dataset_OpPT

    def validate_envs_with_same_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have the same number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                                        op_mask=state.op_mask_tensor,
                                        candidate=state.candidate_tensor,  # [sz_b, J]
                                        fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                                        mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                                        comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                                        fea_pairs=state.fea_pairs_tensor)  # [sz_b, J, M]

            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        return environment_metrics(self.vali_env)

    def validate_envs_with_various_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have various number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                batch_idx = ~torch.from_numpy(self.vali_env.done_flag)
                pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
                                        op_mask=state.op_mask_tensor[batch_idx],
                                        candidate=state.candidate_tensor[batch_idx],  # [sz_b, J]
                                        fea_m=state.fea_m_tensor[batch_idx],  # [sz_b, M, 6]
                                        mch_mask=state.mch_mask_tensor[batch_idx],  # [sz_b, M, M]
                                        comp_idx=state.comp_idx_tensor[batch_idx],  # [sz_b, M, M, J]
                                        dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],  # [sz_b, J, M]
                                        fea_pairs=state.fea_pairs_tensor[batch_idx])  # [sz_b, J, M]

            action = greedy_select_action(pi)
            state, _, done = self.vali_env.step(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        return environment_metrics(self.vali_env)

    def save_model(self, update, validation_metrics, selection_metric):
        """
            save the model
        """
        bundle = make_checkpoint_bundle(
            self.ppo.policy.state_dict(),
            config=json_safe_config(self.config),
            data_fingerprints=self.data_fingerprints,
            update=update,
            validation_metrics={k: v for k, v in validation_metrics.items() if not isinstance(v, list)},
            selection_metric=selection_metric,
            parents=self.parent_checkpoints,
            repo_root=Path(__file__).resolve().parents[1],
        )
        torch.save(bundle, f'./trained_network/{self.data_source}/{self.model_name}.pth')

    def save_budget_checkpoint(self, update):
        """
            snapshot the policy at a fixed optimisation budget (RQ1 budget curves).
            Deliberately does NOT consult validation: the point is to record the policy
            after exactly `update` PPO updates, whatever its quality.
        """
        bundle = make_checkpoint_bundle(
            self.ppo.policy.state_dict(),
            config=json_safe_config(self.config),
            data_fingerprints=self.data_fingerprints,
            update=update,
            validation_metrics={},
            selection_metric=float('nan'),
            parents=self.parent_checkpoints,
            repo_root=Path(__file__).resolve().parents[1],
        )
        path = f'./trained_network/{self.data_source}/{self.model_name}@u{update}.pth'
        torch.save(bundle, path)
        print(f"budget checkpoint saved: {path}")

    def resolve_checkpoint_path(self, checkpoint):
        """
            resolve checkpoint names and paths for fine-tuning starts
        """
        if not checkpoint:
            return ''
        if checkpoint.endswith('.pth') or os.path.sep in checkpoint or '/' in checkpoint:
            return checkpoint
        return f'./trained_network/{self.data_source}/{checkpoint}.pth'

    def load_initial_checkpoint(self):
        """
            optionally start training from an existing merged/specialist checkpoint
        """
        init_path = self.resolve_checkpoint_path(self.config.init_from)
        if not init_path:
            return
        if not os.path.exists(init_path):
            raise FileNotFoundError(f'Initial checkpoint not found: {init_path}')
        checkpoint = torch.load(init_path, map_location=device)
        state_dict, metadata = unwrap_checkpoint(checkpoint)
        mode = getattr(self.config, "init_mode", "exact")
        report = None
        if mode == "exact":
            validate_checkpoint_schema(metadata, self.config.feature_schema, state_dict)
            self.ppo.policy.load_state_dict(state_dict)
        else:
            # warm-start arms (docs/warmstart_protocol_v3.md): transfer into the freshly initialised network, so every
            # part that is not copied keeps exactly the scratch arm's initialisation for this seed
            from warmstart import transfer
            from feature_schemas import SCHEMA_V3, SCHEMA_V3_MO
            if not getattr(self.config, "preference_dim", 0):
                raise ValueError("--init_mode full/backbone/no_critic targets a preference-conditioned DANIEL")
            validate_checkpoint_schema(metadata, SCHEMA_V3_MO if mode == "no_critic" else SCHEMA_V3, state_dict)
            report = transfer(state_dict, self.ppo.policy, mode)
            report = {k: len(v) for k, v in report.items()}
        self.ppo.policy_old.load_state_dict(self.ppo.policy.state_dict())
        self.parent_checkpoints = [{
            "path": str(Path(init_path).resolve()),
            "sha256": file_sha256(init_path),
            "schema_version": metadata.get("schema_version", 1),
            "init_mode": mode, "transfer": report,
        }]
        print(f"loaded initial checkpoint :{init_path}")

    def load_model(self):
        """
            load the trained model
        """
        model_path = f'./trained_network/{self.data_source}/{self.model_name}.pth'
        state_dict, metadata = unwrap_checkpoint(torch.load(model_path, map_location=device))
        validate_checkpoint_schema(metadata, self.config.feature_schema, state_dict)
        self.ppo.policy.load_state_dict(state_dict)


def main():
    trainer = Trainer(configs)
    trainer.train()


if __name__ == '__main__':
    main()
