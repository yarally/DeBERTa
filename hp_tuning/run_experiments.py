import json
import os
import subprocess
import sys
import random
import time
from warnings import catch_warnings, simplefilter

from scipy.stats import qmc, norm
from sklearn.gaussian_process import GaussianProcessRegressor

ROOT_DIR = ''.join(f'{p}/' for p in os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])[:-1]
sys.path.append(ROOT_DIR)
import DeBERTa.apps.run as DeBERTa

INITIAL_SEED = 123456789
MODEL_CONFIG_FILE = '/tmp/DeBERTa/tmp_model_config.json'
RESULT_FILE = '/tmp/DeBERTa/exps/CoLA/eval_results_dev_001068-1068.txt'
MODEL_PARAMS = ["attention_probs_dropout_prob",
                "hidden_act",
                "hidden_dropout_prob",
                "initializer_range",
                "layer_norm_eps",
                "max_relative_positions",
                "padding_idx",
                "pos_att_type",
                "position_biased_input",
                "relative_attention",
                "type_vocab_size"]


# HELPER FUNCTIONS #####################################################################################################
def __parse_config(args):
    with open(args.exp_config) as f:
        config = json.load(f)
    epochs = config['epochs']
    dim = len(config['experiments'])
    lower = [bounds[0] for _, bounds in config['experiments'].items()]
    upper = [bounds[1] for _, bounds in config['experiments'].items()]
    factors = config['experiments'].keys()
    constants = config['constants']
    return dim, epochs, lower, upper, factors, constants


def __run_deberta(args):
    DeBERTa.run(args)
    with open(RESULT_FILE, 'r') as res:
        lines = [line.strip() for line in res.readlines()]
        return float(lines[0].split("accuracy = ")[1]), float(lines[1].split("eval_loss = ")[1])


# OPTIMIZATION STRATEGIES ##############################################################################################
def run_custom_experiments(args):
    print('Applying a custom strategy...')
    with open(args.exp_config) as f:
        config = json.load(f)
    for i, experiment in enumerate(config['experiments']):
        set_variables(args, {**experiment, **config['constants']}, i)
        DeBERTa.run(args)


def run_random_experiments(args):
    print('Applying a random strategy...')
    random_function = random_functions[args.exp_rand]
    results = ['epoch,accuracy,eval_loss,config']
    dim, epochs, lower, upper, factors, constants = __parse_config(args)
    X = random_function(dim, epochs, lower, upper)
    y = []
    for i, x in enumerate(X):
        random.seed(INITIAL_SEED + i)
        # TODO: Make compatible for non-numeric values!
        factor_values = {key: value for key, value in zip(factors, x)}
        set_variables(args, {**factor_values, **constants}, i)
        acc, loss = __run_deberta(args)
        results.append(f'{i}\t{acc}\t{loss}\t{factor_values}')
        y.append(acc)
    print(f'Best accuracy found for random: {max(y)}')
    return results


def run_grid_experiments(args):
    print('Applying a grid search strategy...')
    pass


def run_bayesian_pi_experiments(args):
    print('Applying a bayesian strategy with PI-acquisition...')
    random_function = random_functions[args.exp_rand]
    acquisition_function = acquisition_functions[args.exp_acq]
    results = ['epoch,accuracy,eval_loss,config']
    dim, epochs, lower, upper, factors, constants = __parse_config(args)
    X = random_function(dim, 1, lower, upper)
    y = []
    model = GaussianProcessRegressor()
    for i in range(epochs):
        if i > 0:
            model.fit(X, y)
            candidate_samples = random_function(dim, 1000, lower, upper)
            X.append(acquisition_function(candidate_samples, model, max(y)))
        random.seed(INITIAL_SEED + i)
        # TODO: Make compatible for non-numeric values!
        factor_values = {key: value for key, value in zip(factors, X[i])}
        set_variables(args, {**factor_values, **constants}, i)
        acc, loss = __run_deberta(args)
        results.append(f'{i}\t{acc}\t{loss}\t{factor_values}')
        y.append(acc)
    print(f'Best accuracy found for bayesian: {max(y)}')
    return results


# RANDOM FUNCTIONS #####################################################################################################
def get_quasi_random_samples(dim, length, lower, upper, sampler=qmc.Halton):
    sampler = sampler(dim)
    sample = sampler.random(length)
    return qmc.scale(sample, lower, upper)


def get_random_samples(dim, length, lower, upper):
    return [[random.uniform(lower[i], upper[i]) for i in range(dim)] for _ in range(length)]


# ACQUISITION FUNCTIONS ################################################################################################
def best_probability_of_improvement(samples, model, y_best, maximize=True):
    best = -1
    x_next = None
    for x in samples:
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            mu_, std_ = model.predict([x], return_std=True)
            pi = norm.cdf((mu_ - y_best) / (std_ + 1e-9)) if maximize else norm.cdf((y_best - mu_) / (std_ + 1e-9))
            if pi > best:
                best = pi
                x_next = x
    return x_next


# Unchangeable...
# max_position_embeddings
# vocab_size
# hidden_size
# intermediate_size
# num_attention_heads
# num_hidden_layers

# regularlization param?
def set_variables(args, experiment_config, index):
    """
    ### DeBERTa Variables
    # max_seq_length
    # model_config
    # cls_drop_out
    # vocab_type
    # vat_lambda^
    # vat_learning_rate*
    # vat_init_perturbation
    # vat_loss_fn*

    ### Training variables
    # train_batch_size
    # accumulative_update
    # seed

    ### Optimizer variables
    # lookahead_k
    # lookahead_alpha
    # with_radam
    # opt_type
    # learning_rate*^ --> Weird behavior for lr >= 3e-5
    # scale_steps
    # loss_scale*
    # dump_interval
    # weight_decay
    # max_grad_norm*
    # adam_beta1*
    # adam_beta2*
    # lr_schedule_ends
    # epsilon
    # fp16*
    # warmup_proportion
    # lr_schedule
    """
    with open(MODEL_CONFIG_FILE, 'w') as outfile:
        print(f'Current experiment configuration: {experiment_config}')
        model_config = {param: value for param, value in experiment_config.items() if param in MODEL_PARAMS}
        json.dump(model_config, outfile)

    setattr(args, 'seed', INITIAL_SEED + index)
    for key, value in experiment_config.items():
        if key not in MODEL_PARAMS:
            setattr(args, key, value)


def set_constants(parser):
    # Dirty Hack to work around required command line arguments
    if '--exp_name' in sys.argv:
        task_name = sys.argv[sys.argv.index('--exp_name') + 1]
    else:
        task_name = parser.get_default('exp_name')

    print('Preparing for task:', task_name)
    sys.argv.extend(['--task_name', task_name, '--output_dir', f'/tmp/DeBERTa/exps/{task_name}'])

    parser.set_defaults(
        data_dir=f'/tmp/DeBERTa/glue_tasks/{task_name}',
        model_config=MODEL_CONFIG_FILE,
        do_train=True,
        tag='hp_tuning_',
        init_model='base',
        num_train_epochs=1,
        dump_interval=2048,
    )


def init_parser():
    parser = DeBERTa.build_argument_parser()
    parser.add_argument("--exp_strategy",
                        default='random',
                        type=str,
                        required=False,
                        choices=strategies.keys(),
                        help="The strategy for hyper parameter optimization")
    parser.add_argument("--exp_acq",
                        default='pi',
                        type=str,
                        required=False,
                        choices=acquisition_functions.keys(),
                        help="The acquisition function of the bayesian optimizer")
    parser.add_argument("--exp_rand",
                        default='random',
                        type=str,
                        required=False,
                        choices=random_functions.keys(),
                        help="The method for generating random samples")
    parser.add_argument("--exp_name",
                        default='CoLA',
                        type=str,
                        required=False,
                        choices=['CoLA', 'MNLI', 'MRPC', 'QNLI', 'QQP', 'RTE', 'SST-2', 'STS-B'],
                        help="The type of task to run")
    parser.add_argument("--exp_config",
                        default='config.json',
                        type=str,
                        required=False,
                        help="The config file of the experiments")
    set_constants(parser)
    parser.parse_known_args()
    return parser.parse_args()


strategies = {
    "random": run_random_experiments,
    "grid": run_grid_experiments,
    "bayesian": run_bayesian_pi_experiments,
    "custom": run_custom_experiments
}

random_functions = {
    "random": get_random_samples,
    "qmc": get_quasi_random_samples
}

acquisition_functions = {
    "pi": best_probability_of_improvement
}


def main(args):
    start_time = time.perf_counter()
    power_logger = subprocess.Popen(
        'nvidia-smi --query-gpu=power.draw --format=csv --loop-ms=1000 -f power_usage.csv'.split(' '))
    args.tag += args.exp_strategy
    random.seed(INITIAL_SEED)
    results = strategies[args.exp_strategy](args)
    power_logger.kill()
    with open("power_usage.csv", "a") as power_log:
        power_log.write(f'\n{str(start_time)},{str(time.perf_counter())}')
    print('Power usage logged to power_usage.csv')
    print('##### RESULTS #####')
    [print(res) for res in results]


if __name__ == '__main__':
    main(init_parser())
