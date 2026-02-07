#------------------------------------------------------------------------------------------------
#                                        IMPORTS
#------------------------------------------------------------------------------------------------
import os
print(f"currently in {os.getcwd()}")

import json
import torch
import wandb
import numpy as np

from hydrapfn.scripts.model_trainer import train_model
from hydrapfn.scripts.model_saver import save_model

from hydra_evaluation_helper import EvalHelper

#------------------------------------------------------------------------------------------------
#                                    DEFAULT SETTINGS
#------------------------------------------------------------------------------------------------
base_path = '.'
json_file_path = "tabpfn_original_config.json"
max_features = 100

with open(json_file_path, "r") as f: 
    config = json.load(f)

uniform_int_sampler_f = (lambda a, b : lambda : round(np.random.uniform(a, b)))
choice_values = [
    torch.nn.modules.activation.Tanh, 
    torch.nn.modules.linear.Identity,
    torch.nn.modules.activation.ReLU
    ]

config["differentiable_hyperparameters"]["prior_mlp_activations"]["choice_values"] = choice_values
config["num_classes"] = uniform_int_sampler_f(2, config['max_num_classes']) # Wrong Function
config["num_features_used"] = uniform_int_sampler_f(1, max_features)

device = "cuda:0"

#------------------------------------------------------------------------------------------------
#                                           WANDB
#------------------------------------------------------------------------------------------------

wandb_project = "void"
wandb_job_type = f"test"
wandb_run_name = f"test"

wandb_config= config

wandb_run = wandb.init(project=wandb_project,job_type=wandb_job_type,config=wandb_config, name=wandb_run_name, group="DDP")

eval_class = EvalHelper()

#------------------------------------------------------------------------------------------------
#                                          CUSTOM
#------------------------------------------------------------------------------------------------

config['batch_size'] = 64 
config['emsize'] = 64 
config["epochs"] = 1
config["bptt"] = 100
config["max_eval_pos"] = 90       

config["num_steps"] = 16

config["nlayers"] = 4
config["enable_autocast"] = True

device = "cuda:0"

#------------------------------------------------------------------------------------------------
#                                           MODEL
#------------------------------------------------------------------------------------------------

model = train_model(
    config=config,
    evaluation_class=eval_class
)

save_model(
    model = model,
    path = ".",
    filename = "hydrapfn/trained_models/test_model.cpkt",
    config_sample = config

)


print("worked")