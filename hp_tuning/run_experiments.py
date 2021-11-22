import DeBERTa.apps.run as deberta


class ExperimentConfig(object):
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def set_variables(args):
    args['eval_batch_size'] = 128
    args['predict_batch_size'] = 128
    args['train_batch_size'] = 8
    args['scale_steps'] = 250
    args['loss_scale'] = 16384
    args['accumulative_update'] = 1
    args['num_train_epochs'] = 1
    args['warmup'] = 100
    args['learning_rate'] = 2e-5
    args['max_seq_length'] = 128
    args['cls_drop_out'] = 0.15
    args['vat_lambda'] = 0
    args['vat_learning_rate'] = 1e-4
    args['vat_init_perturbation'] = 1e-2
    args['vat_loss_fn'] = 'symmetric-kl'


def set_constants(args, task_name):
    args['seed'] = 42
    args['vocab_path'] = None
    args['vocab_type'] = 'gpt2'
    args['task_name'] = task_name
    args['data_dir'] = '/tmp/DeBERTa/glue_tasks/' + task_name
    args['output_dir'] = '/tmp/DeBERTa/exps/' + task_name
    args['do_train'] = True
    args['do_eval'] = False
    args['do_predict'] = False
    args['init_model'] = 'base'
    args['tag'] = 'hp-tuning'
    args['debug'] = False
    args['pre_trained'] = None
    args['n_gpu'] = 1
    args['workers'] = 1


def set_model_config(args):
    args['model_config'] = None


def set_optimizer_arguments(args):
    args['dump_interval'] = 10000
    args['weight_decay'] = 0.01
    args['max_grad_norm'] = 1
    args['adam_beta1'] = 0.9
    args['adam_beta2'] = 0.999
    args['lr_schedule_ends'] = 0
    args['epsilon'] = 1e-6
    args['fp16'] = False
    args['warmup_proportion'] = 0.1
    args['lr_schedule'] = 'warmup_linear'


def main():
    args = {}
    set_variables(args)
    set_constants(args, 'CoLA')
    set_model_config(args)
    set_optimizer_arguments(args)
    deberta.run(ExperimentConfig(args))


if __name__ == '__main__':
    # AFTER REBOOT:
    # cache_dir = /tmp/DeBERTa/
    # cd experiments/glue
    # ./download_data.sh  $cache_dir/glue_tasks
    main()


