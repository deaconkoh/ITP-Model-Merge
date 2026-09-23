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

        self.train_data_path = config.train_data_path or \
            f'{self.data_root}/data_train/{self.data_source}/{self.data_name}'
        self.vali_data_path = config.validation_data_path or \
            f'{self.data_root}/data_validation/{self.data_source}/{self.data_name}'
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
        if self.objective.uses_tardiness:
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

        if self.uses_priority_carbon_files:
            self.vali_env.set_initial_data(vali_data[0], vali_data[1], vali_data[2], vali_data[3],
                                           due_date_list=vali_due)
        else:
            self.vali_env.set_initial_data(vali_data[0], vali_data[1])

        self.ppo = PPO_initialize()
        self.parent_checkpoints = []
        self.load_initial_checkpoint()
        self.save_run_manifest()
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)

    def save_run_manifest(self):
        manifest = {
            "schema_version": 2,
            "status": "INITIALIZED",
            "model_name": self.model_name,
            "config": json_safe_config(self.config),
            "data_fingerprints": self.data_fingerprints,
            "final_test_path_not_loaded": str(Path(self.test_data_path).resolve()),
            "due_dates": getattr(self, "due_date_provenance", None),
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

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()

            # resampling the training data
            if i_update % self.reset_env_timestep == 0:
                sampled_data = self.sample_training_instances()
                if len(sampled_data) == 4:
                    dataset_job_length, dataset_op_pt, dataset_op_priority, dataset_op_carbon = sampled_data
                    due = ([self.train_due[i] for i in self.last_sample_idxs]
                           if self.train_due is not None else None)
                    state = self.env.set_initial_data(dataset_job_length, dataset_op_pt,
                                                      dataset_op_priority, dataset_op_carbon,
                                                      due_date_list=due)
                elif len(sampled_data) == 3:
                    dataset_job_length, dataset_op_pt, dataset_op_priority = sampled_data
                    state = self.env.set_initial_data(dataset_job_length, dataset_op_pt, dataset_op_priority)
                else:
                    dataset_job_length, dataset_op_pt = sampled_data
                    state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
            else:
                state = self.env.reset()

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
                                                             fea_pairs=state.fea_pairs_tensor)  # [sz_b, J, M]

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
            self.log.append({"update": i_update, "mean_return": float(mean_rewards_all_env)})

            if (i_update + 1) in self.budget_checkpoints:
                self.save_budget_checkpoint(i_update + 1)

            # validate the trained model
            if (i_update + 1) % self.validate_timestep == 0:
                if self.data_source == "SD1":
                    validation_metrics = self.validate_envs_with_various_op_nums()
                else:
                    validation_metrics = self.validate_envs_with_same_op_nums()

                selection_metric = self.objective.selection_metric(validation_metrics)
                if selection_metric < self.record:
                    self.save_model(i_update + 1, validation_metrics, selection_metric)
                    self.record = selection_metric

                self.validation_log.append({
                    "update": i_update + 1,
                    **validation_metrics,
                    "selection_metric": selection_metric,
                })
                self.save_validation_log()
                tqdm.write(
                    f"Validation makespan={validation_metrics['makespan']:.6g}; "
                    f"carbon={validation_metrics['carbon']:.6g}; "
                    f"operation-priority={validation_metrics['priority']:.6g}; "
                    f"selection={selection_metric:.6g} (best={self.record:.6g})"
                )

            ep_et = time.time()
            
            # print the reward, makespan, loss and training time of the current episode
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    i_update + 1, mean_rewards_all_env, mean_makespan_all_env, loss, ep_et - ep_st))
            if self.config.enable_carbon:
                tqdm.write(f'Episode {i_update + 1}\t carbon: {mean_carbon_all_env:.2f}')

        self.train_et = time.time()

        # log results
        self.save_training_log()

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
            validation_metrics=validation_metrics,
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
        validate_checkpoint_schema(metadata, self.config.feature_schema, state_dict)
        self.ppo.policy.load_state_dict(state_dict)
        self.ppo.policy_old.load_state_dict(state_dict)
        self.parent_checkpoints = [{
            "path": str(Path(init_path).resolve()),
            "sha256": file_sha256(init_path),
            "schema_version": metadata.get("schema_version", 1),
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
