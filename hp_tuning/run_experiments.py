import json

import DeBERTa.apps.run as deberta
import sys

INITIAL_SEED = 42
MODEL_CONFIG_FILE = '/tmp/DeBERTa/tmp_model_config.json'
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
    # learning_rate*
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
        json.dump(experiment_config['model_config'], outfile)

    setattr(args, 'seed', INITIAL_SEED + index)
    for k, v in experiment_config['custom_config'].items():
        setattr(args, k, v)


def set_constants(parser, task_name):
    # Dirty Hack to work around required command line arguments
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


def main(task_name, experiment_file):
    parser = deberta.build_argument_parser()
    set_constants(parser, task_name)
    parser.parse_known_args()
    args = parser.parse_args()
    with open(experiment_file) as f:
        experiments = json.load(f)
    for i, experiment in enumerate(experiments):
        set_variables(args, experiment, i)
        deberta.run(args)


if __name__ == '__main__':
    main('CoLA', 'test_experiment_config.json')
