import os
import torch
import torch.nn as nn

from hydrapfn.hydra_context import HydraModel

from hydrapfn.scripts.model_trainer import train_model


def load_hydrapfn_model(checkpoint_path, device="cuda"):
    """
    Load a single HydraPFN / TabPFN checkpoint from a given full path.

    :param checkpoint_path: Full path to .cpkt file
    :param device: Device to load model onto ('cpu' or 'cuda')
    :param verbose: Verbosity flag passed to get_model
    :return: (model, config_sample)
    """

    print("!! Warning: GPyTorch must be installed !!")
    print(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model_state = checkpoint["model_state_dict"]
    optimizer_state = checkpoint["optimizer"]
    config_sample = checkpoint["config"]

    # Fix activation config if needed
    if (
        "differentiable_hyperparameters" in config_sample
        and "prior_mlp_activations"
        in config_sample["differentiable_hyperparameters"]
    ):
        config_sample["differentiable_hyperparameters"][
            "prior_mlp_activations"
        ]["choice_values_used"] = config_sample[
            "differentiable_hyperparameters"
        ]["prior_mlp_activations"]["choice_values"]

        config_sample["differentiable_hyperparameters"][
            "prior_mlp_activations"
        ]["choice_values"] = [
            torch.nn.Tanh
            for _ in config_sample["differentiable_hyperparameters"][
                "prior_mlp_activations"
            ]["choice_values"]
        ]

    # Override training-specific settings for safe inference
    config_sample["categorical_features_sampler"] = lambda: lambda x: ([], [], [])
    config_sample["num_features_used_in_training"] = config_sample["num_features_used"]
    config_sample["num_features_used"] = lambda: config_sample["num_features"]
    config_sample["num_classes_in_training"] = config_sample["num_classes"]
    config_sample["num_classes"] = 2
    config_sample["batch_size_in_training"] = config_sample["batch_size"]
    config_sample["batch_size"] = 1
    config_sample["bptt_in_training"] = config_sample["bptt"]
    config_sample["bptt"] = 10
    config_sample["bptt_extra_samples_in_training"] = config_sample[
        "bptt_extra_samples"
    ]
    config_sample["bptt_extra_samples"] = None

    config_sample["epochs"] = 0

    # Build model
    model, optimizer = train_model(config_sample, evaluation_class=None, device=device)

    # Remove potential DataParallel prefix
    model_state = {k.replace("module.", ""): v for k, v in model_state.items()}

    # Load weights
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    optimizer.load_state_dict(optimizer_state)

    return model, optimizer, config_sample