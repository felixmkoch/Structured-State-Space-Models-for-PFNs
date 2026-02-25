import math

import hydrapfn.priors as priors
import hydrapfn.encoders as encoders
from hydrapfn.train import Losses
from hydrapfn.train import train
from hydrapfn.utils import get_uniform_single_eval_pos_sampler


def make_get_batch(model_proto, **extra_kwargs):
        
        def new_get_batch(batch_size, 
                          seq_len, 
                          num_features, 
                          hyperparameters, 
                          device, 
                          model_proto=model_proto, 
                          **kwargs):
            kwargs = {**extra_kwargs, **kwargs} # new args overwrite pre-specified args
            return model_proto.get_batch(
                batch_size=batch_size, 
                seq_len=seq_len, 
                device=device, 
                hyperparameters=hyperparameters, 
                num_features=num_features, 
                **kwargs)
        
        return new_get_batch


def get_mlp_prior_hyperparameters(config):
    from tabpfn.priors.utils import gamma_sampler_f
    config = {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}

    if 'random_feature_rotation' not in config:
        config['random_feature_rotation'] = True

    if "prior_sigma_gamma_k" in config:
        sigma_sampler = gamma_sampler_f(config["prior_sigma_gamma_k"], config["prior_sigma_gamma_theta"])
        config['init_std'] = sigma_sampler
    if "prior_noise_std_gamma_k" in config:
        noise_std_sampler = gamma_sampler_f(config["prior_noise_std_gamma_k"], config["prior_noise_std_gamma_theta"])
        config['noise_std'] = noise_std_sampler

    return config


def get_gp_prior_hyperparameters(config):
    return {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}

#----------------------------------------------------------------------
#               TRAIN MODEL FUNCITON
#----------------------------------------------------------------------
def train_model(
        config: dict,           # Config containing training information.
        evaluation_class = None,
        device: str = "cuda"    # Default device to be train on.
):
    
    #----------------------------------------------------------------------
    #                    Gradient Aggregation
    #----------------------------------------------------------------------
    
    if config.get('aggregate_k_gradients') is None:
        config['aggregate_k_gradients'] = math.ceil(
            config['batch_size']
            * (
                config['nlayers']
                * config['emsize']
                * config['bptt']
                * config['bptt']
                / 10824640000
            )
        )

    k = config['aggregate_k_gradients']

    config['num_steps'] = math.ceil(config['num_steps'] * k)
    config['batch_size'] = math.ceil(config['batch_size'] / k)
    config['recompute_attn'] = config.get('recompute_attn', False)


    #----------------------------------------------------------------------
    #                 Setup Prior Configuration
    #----------------------------------------------------------------------

    # Assume config sets prior_type to "prior_bag", which is the most efficient according to TabPFNv1.0
    get_batch_gp = make_get_batch(priors.fast_gp)
    get_batch_mlp = make_get_batch(priors.mlp)

    # This is for the config variable "flexible" == true. We assume that htis is the default here.
    get_batch_gp = make_get_batch(priors.flexible_categorical, **{'get_batch': get_batch_gp})
    get_batch_mlp = make_get_batch(priors.flexible_categorical, **{'get_batch': get_batch_mlp})

    prior_bag_hyperparameters = {'prior_bag_get_batch': (get_batch_gp, get_batch_mlp), 'prior_bag_exp_weights_1': 2.0}
    prior_hyperparameters = {**get_mlp_prior_hyperparameters(config), **get_gp_prior_hyperparameters(config), **prior_bag_hyperparameters}
    model_proto = priors.prior_bag

    # This is for the config variable "flexible" == true. We assume that htis is the default here.
    prior_hyperparameters['normalize_labels'] = True
    prior_hyperparameters['check_is_compatible'] = True
    prior_hyperparameters['prior_mlp_scale_weights_sqrt'] = config['prior_mlp_scale_weights_sqrt'] if 'prior_mlp_scale_weights_sqrt' in prior_hyperparameters else None
    prior_hyperparameters['rotate_normalized_labels'] = config['rotate_normalized_labels'] if 'rotate_normalized_labels' in prior_hyperparameters else True

    # We assume "differentiable" in the config is always true
    get_batch_base = make_get_batch(model_proto)
    extra_kwargs = {'get_batch': get_batch_base, 'differentiable_hyperparameters': config['differentiable_hyperparameters']}
    model_proto = priors.differentiable_prior
    use_style = True

    # Because config['nan_prob_unknown_reason'] > 0.0
    encoder = encoders.NanHandlingEncoder

    # We assume that the maximum foir the number of classes is 10.
    loss = Losses.ce(config['max_num_classes'])

    config.setdefault('multiclass_type', 'rank')
    config.setdefault('mix_activations', False)
    config.setdefault('bptt_extra_samples', None)
    config['eval_positions'] = [int(config['bptt'] * 0.95)] if config['bptt_extra_samples'] is None else [int(config['bptt'])]


    #----------------------------------------------------------------------
    #                 Actual Model Train Function
    #----------------------------------------------------------------------

    single_eval_pos_generator = get_uniform_single_eval_pos_sampler(config.get('max_eval_pos', config['bptt']), min_len=config.get('min_eval_pos', 0))

    extra_prior_kwargs_dict={
        'num_features': config['num_features'],
        'hyperparameters': prior_hyperparameters,
        'batch_size_per_gp_sample': config.get('batch_size_per_gp_sample', None),
        **extra_kwargs
    }

    _, _, model, optimizer, _ = train(
        priordataloader_class = model_proto.DataLoader,
        criterion = loss,
        encoder_generator = encoder,
        y_encoder_generator = encoders.Linear,
        nhid=config['emsize'] * config['nhid_factor'],
        warmup_epochs = 20,
        steps_per_epoch = config['num_steps'],
        single_eval_pos_gen = single_eval_pos_generator,
        extra_prior_kwargs_dict = extra_prior_kwargs_dict,
        evaluation_class = evaluation_class,
        weight_decay = 0.0,
        config = config,
        device=device,
        **config
    )

    return model, optimizer