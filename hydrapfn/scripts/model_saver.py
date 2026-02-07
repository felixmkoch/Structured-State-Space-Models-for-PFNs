import os
import torch

def save_model(
        model, 
        path, 
        filename, 
        config_sample
        ):
    
    config_sample = {**config_sample}

    def make_serializable(config_sample):
        if isinstance(config_sample, dict):
            config_sample = {k: make_serializable(config_sample[k]) for k in config_sample}
        if isinstance(config_sample, list):
            config_sample = [make_serializable(v) for v in config_sample]
        if callable(config_sample):
            config_sample = str(config_sample)
        return config_sample

    #if 'num_features_used' in config_sample:
    #    del config_sample['num_features_used']

    #config_sample['num_classes_as_str'] = str(config_sample['num_classes'])
    #del config_sample['num_classes']

    config_sample = make_serializable(config_sample)

    torch.save((model.state_dict(), None, config_sample), os.path.join(path, filename))