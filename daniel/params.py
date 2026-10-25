import argparse
import json
from pathlib import Path


def str2bool(v):
    """
        transform string value to bool value
    :param v: a string input
    :return: the bool value
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser = argparse.ArgumentParser(description='Arguments for DANIEL_FJSP')
parser.add_argument('--config', type=str, default='',
                    help='JSON experiment configuration; explicit CLI arguments override file values')
# args for device
parser.add_argument('--device', type=str, default='cuda', help='Device name')
parser.add_argument('--device_id', type=str, default='0', help='Device id')

# args for file_name

parser.add_argument('--model_suffix', type=str, default='', help='Suffix of the model')
parser.add_argument('--data_suffix', type=str, default='mix', help='Suffix of the data')
parser.add_argument('--init_from', type=str, default='',
                    help='Optional checkpoint path or model name to continue/fine-tune from')

# args for AutoExperiment
parser.add_argument('--cover_flag', type=str2bool, default=True, help='Whether covering test results of the model')
parser.add_argument('--cover_data_flag', type=str2bool, default=False, help='Whether covering the generated data')
parser.add_argument('--cover_heu_flag', type=str2bool, default=False,
                    help='Whether covering test results of heuristics')
parser.add_argument('--cover_train_flag', type=str2bool, default=True, help='Whether covering the trained model')

# args for data load
parser.add_argument('--model_source', type=str, default='SD2', help='Suffix of the data that model trained on')
parser.add_argument('--data_source', type=str, default='SD2', help='Suffix of test data')
parser.add_argument('--data_root', type=str, default='./data',
                    help='Root directory containing SD1/SD2 data folders')
parser.add_argument('--train_data_path', type=str, default='',
                    help='Explicit training dataset directory (must differ from validation/test)')
parser.add_argument('--validation_data_path', type=str, default='',
                    help='Explicit validation dataset directory (must differ from training/test)')
parser.add_argument('--test_data_path', type=str, default='',
                    help='Explicit final-test dataset directory; training never loads this path')

# args for SD2 data generation
parser.add_argument('--op_per_job', type=float, default=0,
                    help='Number of operations per job, default 0, means the number equals m')
parser.add_argument('--op_per_mch_min', type=int, default=1,
                    help='Minimum number of compatible machines for each operation')
parser.add_argument('--op_per_mch_max', type=int, default=5,
                    help='Maximum number of compatible machines for each operation')
parser.add_argument('--data_size', type=int, default=100, help='The number of instances for data generation')
parser.add_argument('--data_type', type=str, default="train",
                    choices=['train', 'validation', 'final_test', 'legacy_test', 'legacy_vali'],
                    help='Generated split; legacy values reproduce inherited paths only')

# args for testData to excel
parser.add_argument('--sort_flag', type=str2bool, default=True,
                    help='Whether sorting the printed results by the makespan')

# args for or-tools
parser.add_argument('--max_solve_time', type=int, default=1800, help='The maximum solving time of OR-Tools')

# args for seed
parser.add_argument('--seed_datagen', type=int, default=200, help='Seed for data generation')
parser.add_argument('--seed_train_vali_datagen', type=int, default=100, help='Seed for generate validation data')
parser.add_argument('--seed_train_datagen', type=int, default=200, help='Seed for canonical training data')
parser.add_argument('--seed_validation_datagen', type=int, default=100, help='Seed for canonical validation data')
parser.add_argument('--seed_final_test_datagen', type=int, default=400,
                    help='Seed for untouched canonical final-test data')
parser.add_argument('--seed_train', type=int, default=300, help='Seed for training')
parser.add_argument('--seed_test', type=int, default=50, help='Seed for testing heuristics')
# args for tricks

# args for env
parser.add_argument('--n_j', type=int, default=10, help='Number of jobs of the instance')
parser.add_argument('--n_m', type=int, default=5, help='Number of machines of the instance')
parser.add_argument('--n_op', type=int, default=50, help='Number of operations of the instance')
parser.add_argument('--low', type=int, default=1, help='Lower Bound of processing time(PT)')
parser.add_argument('--high', type=int, default=99, help='Upper Bound of processing time')

# args for network
parser.add_argument('--fea_j_input_dim', type=int, default=11, help='Dimension of operation raw feature vectors')
parser.add_argument('--fea_m_input_dim', type=int, default=8, help='Dimension of machine raw feature vectors')
parser.add_argument('--fea_pair_input_dim', type=int, default=9, help='Dimension of operation-machine pair features')
parser.add_argument('--feature_schema', type=str, default='canonical_f11_p9_v2',
                    choices=['canonical_f11_p9_v2', 'legacy_f11_p9_v1',
                             'legacy_f11_p8_v1', 'legacy_f10_p8_v1'],
                    help='Versioned definition and ordering of model input features')

parser.add_argument('--dropout_prob', type=float, default=0.0, help='Dropout rate (1 - keep probability).')

parser.add_argument('--num_heads_OAB', nargs='+', type=int, default=[4, 4],
                    help='Number of attention head of operation message attention block')
parser.add_argument('--num_heads_MAB', nargs='+', type=int, default=[4, 4],
                    help='Number of attention head of machine message attention block')
parser.add_argument('--layer_fea_output_dim', nargs='+', type=int, default=[32, 8],
                    help='Output dimension of the DAN layers')

parser.add_argument('--num_mlp_layers_actor', type=int, default=3, help='Number of layers in Actor network')
parser.add_argument('--hidden_dim_actor', type=int, default=64, help='Hidden dimension of Actor network')
parser.add_argument('--num_mlp_layers_critic', type=int, default=3, help='Number of layers in Critic network')
parser.add_argument('--hidden_dim_critic', type=int, default=64, help='Hidden dimension of Critic network')

# args for priority-flexibility training
parser.add_argument('--enable_priority', type=str2bool, default=True,
                    help='Whether to add fixed job-priority information to the environment')
parser.add_argument('--urgent_jobs', type=int, default=0,
                    help='Number of urgent jobs per training/validation instance when priority is enabled')
parser.add_argument('--priority_weight', type=float, default=3.0,
                    help='Priority weight assigned to urgent jobs')
parser.add_argument('--priority_reward_weight', type=float, default=1.0,
                    help='Weight of the priority reward term added to the normal makespan reward')
parser.add_argument('--priority_seed', type=int, default=50,
                    help='Random seed used to assign urgent jobs')
parser.add_argument('--priority_scope', type=str, default='operation',
                    choices=['operation', 'legacy_job'],
                    help='Canonical runs require operation priorities; legacy_job is experimental only')
parser.add_argument('--train_from_files', type=str2bool, default=True,
                    help='Whether to sample training instances from files instead of generating new SD2 instances')
parser.add_argument('--enable_carbon', type=str2bool, default=True,
                    help='Whether to use operation-machine carbon values for carbon-aware training/evaluation')
parser.add_argument('--carbon_feature', type=str2bool, default=True,
                    help='Whether to add carbon as an extra pair feature; changes checkpoint architecture')
parser.add_argument('--due_date_manifest', type=str, default='',
                    help='Frozen due-date manifest (default: data/due_dates/due_date_manifest.json)')
parser.add_argument('--tardiness_reward_weight', type=float, default=1.0,
                    help='Weight of the total-tardiness reward term (goals containing t)')
parser.add_argument('--carbon_reward_weight', type=float, default=0.01,
                    help='Penalty weight for chosen carbon when carbon-aware reward is enabled')
parser.add_argument('--goal', type=str, default='m', choices=['m', 'c', 'p', 'mc', 'mp', 'mcp', 't', 'mt', 'ct', 'mct'],
                    help='Inherited objective: makespan/carbon/operation-priority and their combinations')

# args for PPO Algorithm
# v2 observation/reward (results/v2_fixes.md). v1 is the default and is untouched by these flags.
parser.add_argument('--obs_version', type=str, default='v1', choices=['v1', 'v2', 'v3'],
                    help='v2 adds due-date observations (2 operation + 1 pair feature) and scales carbon by 1/100; '
                         'v3 = benchmark v3 observations (docs/benchmark_v3.md)')
parser.add_argument('--reward_version', type=str, default='v1', choices=['v1', 'v2', 'v3'],
                    help='v2 normalises each reward term by the Step 4 score references (return = -score); '
                         'v3 = the same form with the benchmark v3 references')
parser.add_argument('--benchmark_manifest', type=str, default='',
                    help='Frozen benchmark v3 manifest (default: data/benchmark_v3/manifest_benchmark_v3.json)')
parser.add_argument('--reward_ref_manifest', type=str, default='',
                    help='Frozen v2 reward-reference manifest (default: data/reward_refs/reward_ref_manifest.json)')
parser.add_argument('--reward_ref_seed', type=int, default=None,
                    help='Seed whose centroid references normalise the v2 reward (default: --seed_train)')
# v3 baselines (docs/evaluation_protocol_v3.md); every default leaves earlier behaviour unchanged
parser.add_argument('--v3_train_source', type=str, default='pool', choices=['pool', 'generator'],
                    help='benchmark v3 training data: the fixed v3 train pool, or fresh SD2-generator instances '
                         '(the original DANIEL scheme) with v3 parameters from the frozen code')
parser.add_argument('--converge_window', type=int, default=0,
                    help='Stop when the best validation value improved by < --converge_tol over this many updates '
                         '(0 disables; validation also runs at update 0 when enabled)')
parser.add_argument('--converge_tol', type=float, default=0.005, help='Relative improvement threshold')
parser.add_argument('--min_updates', type=int, default=0, help='No convergence stop before this many updates')
parser.add_argument('--chunk_updates', type=int, default=0,
                    help='Run at most this many updates in this process, then save a resume state and exit '
                         '(0 disables; must be a multiple of --reset_env_timestep and --validate_timestep)')
parser.add_argument('--resume', type=str2bool, default=False,
                    help='Continue from <model>.resume.pth (optimiser and RNG states included) if it exists')
# v3 MO-DANIEL (docs/evaluation_protocol_v3.md, Addendum A); every default leaves earlier behaviour unchanged
parser.add_argument('--preference_dim', type=int, default=0, choices=[0, 3],
                    help='3 = preference-conditioned DANIEL: w = (w_m, w_c, w_t) appended to the actor and critic inputs, '
                         'reward -(w . (MK/MKref, C/Cref, T/Tref)); needs benchmark v3 and --goal mct')
parser.add_argument('--pref_dirichlet', type=float, default=1.0,
                    help='Training preferences: w ~ Dirichlet(alpha, alpha, alpha), drawn per environment every update')
parser.add_argument('--mo_val_grid_step', type=float, default=0.25,
                    help='MO validation: simplex grid step of the greedy validation preferences (0.25 -> 15 vectors)')
parser.add_argument('--mo_val_norm', type=str, default='',
                    help='MO validation: frozen per-instance ideal/nadir points of the validation split (JSON)')
parser.add_argument('--mo_val_first', type=int, default=0, help='MO validation: only the first N instances (smoke tests)')
parser.add_argument('--init_mode', type=str, default='exact', choices=['exact', 'full', 'backbone', 'no_critic'],
                    help='How --init_from is loaded: exact = same schema (default, unchanged); full / backbone = weight '
                         'transfer from a v3 specialist (or merge) into a preference-conditioned DANIEL; no_critic = '
                         'everything but the critic from a preference-conditioned student (docs/warmstart_protocol_v3.md)')
parser.add_argument('--log_ppo_stats', type=str2bool, default=False,
                    help='Log per-update PPO statistics (entropy, approx. KL, clip fraction, losses, explained variance, '
                         'gradient norm); observation only, the update is unchanged')
parser.add_argument('--num_envs', type=int, default=20, help='Batch size for training environments')
parser.add_argument('--max_updates', type=int, default=1000, help='No. of episodes of each env for training')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')

parser.add_argument('--gamma', type=float, default=1, help='Discount factor used in training')
parser.add_argument('--k_epochs', type=int, default=4, help='Update frequency of each episode')
parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip parameter')
parser.add_argument('--vloss_coef', type=float, default=0.5, help='Critic loss coefficient')
parser.add_argument('--ploss_coef', type=float, default=1, help='Policy loss coefficient')
parser.add_argument('--entloss_coef', type=float, default=0.01, help='Entropy loss coefficient')
parser.add_argument('--tau', type=float, default=0, help='Policy soft update coefficient')
parser.add_argument('--gae_lambda', type=float, default=0.98, help='GAE parameter')

# args for training
parser.add_argument('--train_size', type=str, default="10x5", help='Size of training instances')
parser.add_argument('--validate_timestep', type=int, default=10, help='Interval for validation and data log')
parser.add_argument('--budget_checkpoints', type=str, default='',
                    help='Comma-separated update counts at which to snapshot the policy for '
                         'RQ1 budget curves, e.g. "0,50,100,250,500,1000". Saved as '
                         '<model_name>@u<N>.pth alongside (not replacing) the usual '
                         'validation-improvement checkpoint. Empty disables the feature.')
parser.add_argument('--reset_env_timestep', type=int, default=20, help='Interval for reseting the environment')
parser.add_argument('--minibatch_size', type=int, default=1024, help='Batch size for computing the gradient')

# args for test
parser.add_argument('--test_data', nargs='+', default=['10x5+mix'], help='List of data for testing')
parser.add_argument('--test_mode', type=str2bool, default=False, help='Whether using the sampling strategy in testing')
parser.add_argument('--sample_times', type=int, default=100, help='Sampling times for the sampling strategy')
parser.add_argument('--test_model', nargs='+', default=['10x5+mix'], help='List of model for testing')
parser.add_argument('--test_method', nargs='+', default=[], help='List of heuristic methods for testing')

config_probe, _ = parser.parse_known_args()
if config_probe.config:
    config_path = Path(config_probe.config)
    values = json.loads(config_path.read_text(encoding='utf-8'))
    known_destinations = {action.dest for action in parser._actions}
    unknown = sorted(set(values) - known_destinations)
    if unknown:
        raise ValueError(f'Unknown keys in experiment config {config_path}: {unknown}')
    parser.set_defaults(**values)

configs = parser.parse_args()

if any('carbon+priority' in data_name for data_name in configs.test_data):
    configs.enable_priority = True
    configs.enable_carbon = True

if (configs.enable_priority or configs.enable_carbon) and configs.fea_j_input_dim == 10:
    configs.fea_j_input_dim = 11
if configs.enable_carbon and configs.carbon_feature and configs.fea_pair_input_dim == 8:
    configs.fea_pair_input_dim = 9
if configs.obs_version == 'v2':
    # v1 layout kept in place; 2 operation features and 1 pair feature appended (results/v2_fixes.md)
    configs.fea_j_input_dim, configs.fea_pair_input_dim = 13, 10
    configs.feature_schema = 'canonical_f13_p10_obsv2'
if configs.obs_version == 'v3' or configs.reward_version == 'v3':
    # benchmark v3 (docs/benchmark_v3.md): no priority slot; machine power; fixed carbon and time scales
    if configs.obs_version != configs.reward_version:
        raise ValueError('benchmark v3 needs both --obs_version v3 and --reward_version v3')
    configs.fea_j_input_dim, configs.fea_m_input_dim, configs.fea_pair_input_dim = 12, 9, 10
    configs.feature_schema = 'benchmark_v3_f12_m9_p10'
    configs.enable_priority, configs.enable_carbon, configs.carbon_feature = False, True, True
if configs.preference_dim:
    if configs.obs_version != 'v3':
        raise ValueError('--preference_dim needs benchmark v3 (--obs_version v3 --reward_version v3)')
    if configs.goal != 'mct':
        raise ValueError('a preference-conditioned model is trained with --goal mct (the preference sets the weights)')
    configs.feature_schema = 'benchmark_v3_f12_m9_p10_w3'
