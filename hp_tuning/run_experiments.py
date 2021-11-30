import json
import os
import sys
import random

ROOT_DIR = ''.join(f'{p}/' for p in os.path.dirname(os.path.abspath(__file__)).split('/')[:-1])[:-1]
sys.path.append(ROOT_DIR)
import DeBERTa.apps.run as DeBERTa

INITIAL_SEED = 42
MODEL_CONFIG_FILE = '/tmp/DeBERTa/tmp_model_config.json'
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


def run_custom_experiments(args):
    print('Applying a custom strategy...')
    with open(args.exp_config) as f:
        config = json.load(f)
    for i, experiment in enumerate(config['experiments']):
        set_variables(args, {**experiment, **config['constants']}, i)
        DeBERTa.run(args)


def run_random_experiments(args):
    print('Applying a random strategy...')
    with open(args.exp_config) as f:
        config = json.load(f)
        for i in range(config['epochs']):
            random.seed(INITIAL_SEED + i)
            factors = {key: random.uniform(value[0], value[1]) for key, value in config['experiments'].items()} # TODO: Make compatible for non-numeric values!
            set_variables(args, {**factors, **config['constants']}, i)
            DeBERTa.run(args)


def run_grid_experiments(args):
    print('Applying a grid search strategy...')
    pass


# Unchangeable...
# max_position_embeddings
# vocab_size
# hidden_size
# intermediate_size
# num_attention_heads
# num_hidden_layers
def set_variables(args, experiment_config, index):
    """
    ### DeBERTa Variables
    # max_seq_length
    # model_config
    # cls_drop_out
    # vocab_type
    # vat_lambda
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
    # learning_rate* --> Weird behavior for lr >= 3e-5
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
        tag='hp-tuning',
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
                        choices=['random', 'grid', 'custom'],
                        help="The strategy for hyper parameter optimization")
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


def main(args):
    strategies = {
        "random": run_random_experiments,
        "grid": run_grid_experiments,
        "custom": run_custom_experiments
    }
    return strategies[args.exp_strategy](args)


if __name__ == '__main__':
    main(init_parser())
