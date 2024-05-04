#------------------------------------------------------------------------------------------------
#                                        IMPORTS
#------------------------------------------------------------------------------------------------
import os
print(f"currently in {os.getcwd()}")
import time
from datetime import datetime

import torch
import json

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt
from tabpfn.scripts.model_builder import get_model, save_model
from tabpfn.scripts.model_builder_mamba import get_model_mamba 
from tabpfn.scripts.model_configs import *

from tabpfn.priors.utils import plot_features
from tabpfn.priors.utils import uniform_int_sampler_f

#------------------------------------------------------------------------------------------------
#                                       END IMPORTS
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                        PARAMETER
#------------------------------------------------------------------------------------------------

# Mandatory Parameter
device = "cuda" if torch.cuda.is_available() else "cpu"

# Other Parameters
maximum_runtime = 10000
base_path = '.'
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

config['batch_size'] = 64 # just because we did this in the other config. Would be 64 default

#------------------------------------------------------------------------------------------------
#                                         END CONFIG
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                           MODEL
#------------------------------------------------------------------------------------------------

# Get the model 
model = get_model_mamba(config, device, should_train=True, verbose=2) # , state_dict=model[2].state_dict()

(hp_embedding, data, _), targets, single_eval_pos = next(iter(model[3]))


#------------------------------------------------------------------------------------------------
#                                         END MODEL
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                         VISUALIZE
#------------------------------------------------------------------------------------------------


from tabpfn.utils import normalize_data
fig = plt.figure(figsize=(8, 8))
N = 100

print(f"# Data Entries: {len(data)}")
print(f"Type of the  Data Entries: {type(data)}")
print(f"# Target Entries: {len(data)}")
print(f"Type of the  Target Entries: {type(data)}")

print(f"Torch Data Entries look like: ")
print(data[:50])

d = np.concatenate([data[:, 0, :].T, np.expand_dims(targets[:, 0], -1).T])
d[np.isnan(d)] = 0
c = np.corrcoef(d)
plt.matshow(np.abs(c), vmin=0, vmax=1)
plt.show()

#------------------------------------------------------------------------------------------------
#                                       END VISUALIZE
#------------------------------------------------------------------------------------------------



print("works")