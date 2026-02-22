import torch
from torch import nn
import random
import numpy as np

from hydrapfn.scripts.tabular_baselines import get_scoring_string
from hydrapfn.scripts import tabular_metrics
from hydrapfn.scripts.hydra_prediction_interface import hydra_predict
from hydrapfn.utils import torch_nanmean


def generate_valid_split(X, y, bptt, eval_position, is_classification, split_number=1):
    """Generates a deteministic train-(test/valid) split. Both splits must contain the same classes and all classes in
    the entire datasets. If no such split can be sampled in 7 passes, returns None.

    :param X: torch tensor, feature values
    :param y: torch tensor, class values
    :param bptt: Number of samples in train + test
    :param eval_position: Number of samples in train, i.e. from which index values are in test
    :param split_number: The split id
    :return:
    """
    done, seed = False, 13

    torch.manual_seed(split_number)
    perm = torch.randperm(X.shape[0]) if split_number > 1 else torch.arange(0, X.shape[0])
    X, y = X[perm], y[perm]
    while not done:
        if seed > 20:
            return None, None # No split could be generated in 7 passes, return None
        random.seed(seed)
        i = random.randint(0, len(X) - bptt) if len(X) - bptt > 0 else 0
        y_ = y[i:i + bptt]

        if is_classification:
            # Checks if all classes from dataset are contained and classes in train and test are equal (contain same
            # classes) and
            done = len(torch.unique(y_)) == len(torch.unique(y))
            done = done and torch.all(torch.unique(y_) == torch.unique(y))
            done = done and len(torch.unique(y_[:eval_position])) == len(torch.unique(y_[eval_position:]))
            done = done and torch.all(torch.unique(y_[:eval_position]) == torch.unique(y_[eval_position:]))
            seed = seed + 1
        else:
            done = True

    eval_xs = torch.stack([X[i:i + bptt].clone()], 1)
    eval_ys = torch.stack([y[i:i + bptt].clone()], 1)

    return eval_xs, eval_ys


def evaluate_position(X, 
                      y, 
                      categorical_feats, 
                      model, 
                      bptt, 
                      eval_position, 
                      ds_name, 
                      metric_used=None, 
                      device='cuda', 
                      **kwargs):
    
    eval_xs, eval_ys = generate_valid_split(
        X, 
        y, 
        bptt, 
        eval_position, 
        is_classification=tabular_metrics.is_classification(metric_used), 
        split_number=1)
    
    if eval_xs is None:
        print(f"No dataset could be generated {ds_name} {bptt}")
        return None
    
    eval_ys = (eval_ys > torch.unique(eval_ys).unsqueeze(0)).sum(axis=1).unsqueeze(-1)

    if isinstance(model, nn.Module):
        model = model.to(device)
        eval_xs = eval_xs.to(device)
        eval_ys = eval_ys.to(device)

    outputs, inference_time = hydra_predict(
        model,
        eval_xs,
        eval_ys,
        eval_position,
        metric_used = metric_used,
        categorical_feats = categorical_feats,
        inference_mode = True,
        device = device,
        extend_features = True,
        **kwargs
    )

    eval_ys = eval_ys[eval_position:]


    if outputs is None:
        print('Execution failed', ds_name)
        return None

    if torch.is_tensor(outputs): # Transfers data to cpu for saving
        outputs = outputs.cpu()
        eval_ys = eval_ys.cpu()

    ds_result = None, outputs, eval_ys, inference_time

    return ds_result
    


def evaluate(
        datasets,
        bptt,
        eval_positions,
        metric_used,
        model,
        device='cuda',
        **kwargs
):
    
    overall_result = {'metric_used': get_scoring_string(metric_used), 'bptt': bptt, 'eval_positions': eval_positions}

    aggregated_metric_datasets, num_datasets = torch.tensor(0.0), 0

    for [ds_name, X, y, categorical_feats, _, _] in datasets:
        dataset_bptt = min(len(X), bptt)

        aggregated_metric, num = torch.tensor(0.0), 0
        ds_result = {}
    
        for eval_position in eval_positions:
            ys = None

            eval_position_real = int(dataset_bptt * 0.5) if 2 * eval_position > dataset_bptt else eval_position
            eval_position_bptt = int(eval_position_real * 2.0)

            # Result: None, outputs, eval_ys, inference_time
            r = evaluate_position(
                X, 
                y, 
                model=model, 
                num_classes=len(torch.unique(y)), 
                categorical_feats = categorical_feats,
                bptt = eval_position_bptt, 
                ds_name=ds_name, 
                eval_position = eval_position_real, 
                metric_used = metric_used, 
                device=device, 
                **kwargs)
            
            if r is None:
                    print('Execution failed', ds_name)
                    continue
            
            _, outputs, ys, time_used = r

            if torch.is_tensor(outputs):
                outputs = outputs.to(outputs.device)
                ys = ys.to(outputs.device)


            # WARNING: This leaks information on the scaling of the labels
            if isinstance(model, nn.Module) and "BarDistribution" in str(type(model.criterion)):
                ys = (ys - torch.min(ys, axis=0)[0]) / (torch.max(ys, axis=0)[0] - torch.min(ys, axis=0)[0])

            # If we use the bar distribution and the metric_used is r2 -> convert buckets
            #  metric used is prob -> keep
            if isinstance(model, nn.Module) and "BarDistribution" in str(type(model.criterion)) and (
                    metric_used == tabular_metrics.r2_metric or metric_used == tabular_metrics.root_mean_squared_error_metric):
                ds_result[f'{ds_name}_bar_dist_at_{eval_position}'] = outputs
                outputs = model.criterion.mean(outputs)

            ys = ys.T
            ds_result[f'{ds_name}_outputs_at_{eval_position}'] = outputs
            ds_result[f'{ds_name}_ys_at_{eval_position}'] = ys
            ds_result[f'{ds_name}_time_at_{eval_position}'] = time_used

            ds_result["last_outputs"] = outputs

            new_metric = torch_nanmean(torch.stack([metric_used(ys[i], outputs[i]) for i in range(ys.shape[0])]))

            make_scalar = lambda x: float(x.detach().cpu().numpy()) if (torch.is_tensor(x) and (len(x.shape) == 0)) else x
            new_metric = make_scalar(new_metric)
            ds_result = {k: make_scalar(ds_result[k]) for k in ds_result.keys()}

            lib = np
            if not lib.isnan(new_metric).any():
                aggregated_metric, num = aggregated_metric + new_metric, num + 1


        overall_result.update(ds_result)
        if num > 0:
            aggregated_metric_datasets, num_datasets = (aggregated_metric_datasets + (aggregated_metric / num)), num_datasets + 1

    overall_result['mean_metric'] = aggregated_metric_datasets / num_datasets

    return overall_result

            