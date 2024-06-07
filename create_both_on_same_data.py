#------------------------------------------------------------------------------------------------
#                                        IMPORTS
#------------------------------------------------------------------------------------------------
import os
print(f"currently in {os.getcwd()}")
import time
from datetime import datetime

import torch
# Limit PyTorch CUDA use
#torch.cuda.set_per_process_memory_fraction(0.6)
import json

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from tabpfn.scripts.model_builder import get_model, save_model
from tabpfn.scripts.model_builder_mamba import get_model_mamba 
from tabpfn.scripts.model_builder_both import get_model_both
from tabpfn.scripts.model_configs import *

from tabpfn.priors.utils import plot_features
from tabpfn.priors.utils import uniform_int_sampler_f

#------------------------------------------------------------------------------------------------
#                                       END IMPORTS
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                     HELPER FUNCTIONS
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                   END HELPER FUNCTIONS
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                        PARAMETER
#------------------------------------------------------------------------------------------------

# Mandatory Parameter
device = "cuda" if torch.cuda.is_available() else "cpu"

# Other Parameters
maximum_runtime = 10000
base_path = os.path.join("tabpfn")
print(f"Base Path is: {base_path}")
max_features = 100
large_datasets = True
max_samples = 10000 if large_datasets else 5000
bptt = 10000 if large_datasets else 3000
suite='cc'

# Others
json_file_path = "config.json"

#------------------------------------------------------------------------------------------------
#                                      END PARAMETER
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                         CONFIG
#------------------------------------------------------------------------------------------------

with open(json_file_path, "r") as f:
    config = json.load(f)
    
# Fill in stuff that could not be loaded properly into the config.json
uniform_int_sampler_f = (lambda a, b : lambda : round(np.random.uniform(a, b)))
choice_values = [
    torch.nn.modules.activation.Tanh, 
    torch.nn.modules.linear.Identity,
    torch.nn.modules.activation.ReLU
    ]

config["differentiable_hyperparameters"]["prior_mlp_activations"]["choice_values"] = choice_values
config["num_classes"] = uniform_int_sampler_f(2, config['max_num_classes']) # Wrong Function
config["num_features_used"] = uniform_int_sampler_f(1, max_features)

config['batch_size'] = 256 # just because we did this in the other config. Would be 64 default
config['emsize'] = 128 # Default was on 512, just to save some GPU mem.
config["epochs"] = 1

mamba_autocast = False
num_mamba_layers=2

#------------------------------------------------------------------------------------------------
#                                         END CONFIG
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                           MODELS
#------------------------------------------------------------------------------------------------

# Get the model 
#model = get_model(config, device, should_train=True, verbose=0) # , state_dict=model[2].state_dict()
transformer_model, mamba_model = get_model_both(config, device, should_train=True, verbose=0, mamba_autocast=mamba_autocast,num_mamba_layers=num_mamba_layers) # , state_dict=model[2].state_dict()

(transformer_hp_embedding, transformer_data, _), transformer_targets, transformer_single_eval_pos = next(iter(transformer_model[3]))
(mamba_hp_embedding, mamba_data, _), mamba_targets, mamba_single_eval_pos = next(iter(mamba_model[3]))

add_name = "mamba_custom"
i = 2
config['epoch_in_training'] = config["epochs"]

save_model(mamba_model[2], 
           base_path, 
           f'models_diff/prior_diff_real_checkpoint{add_name}_epoch_{i}.cpkt',
           config
           )

#------------------------------------------------------------------------------------------------
#                                         END MODELS
#------------------------------------------------------------------------------------------------

print("works")