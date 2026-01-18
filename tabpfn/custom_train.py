import os
import itertools
import argparse
import time
import datetime
import yaml
from contextlib import nullcontext
from evaluation_helper import EvalHelper


import torch
from torch import nn

import tabpfn.utils as utils
from tabpfn.transformer import TransformerModel
from tabpfn.utils import get_cosine_schedule_with_warmup, get_openai_lr, StoreDictKeyPair, get_weighted_single_eval_pos_sampler, get_uniform_single_eval_pos_sampler
import tabpfn.priors as priors
import tabpfn.encoders as encoders
import tabpfn.positional_encodings as positional_encodings
from tabpfn.utils import init_dist
from torch.amp import autocast, GradScaler
from torch import nn
import wandb
from tabpfn.mamba import MambaModel
from tabpfn.hydra import HydraModel
from tabpfn.scripts import tabular_metrics
import numpy as np

class Losses():
    gaussian = nn.GaussianNLLLoss(full=True, reduction='none')
    mse = nn.MSELoss(reduction='none')
    def ce(num_classes):
        num_classes = num_classes.shape[0] if torch.is_tensor(num_classes) else num_classes
        return nn.CrossEntropyLoss(reduction='none', weight=torch.ones(num_classes))
    bce = nn.BCEWithLogitsLoss(reduction='none')


def sample_train(x, y, eval_position, bag_size):
    generator = torch.Generator()
    rng = np.random.RandomState()
    gen_seed = rng.randint(1e7)
    generator.manual_seed(gen_seed)

    n = eval_position
    m = bag_size

    first_part_x1 = x[1][:n]
    first_part_x2 = x[2][:n]
    first_part_y = y[:n]

    # Start by including all examples at least once
    indices = torch.arange(n)

    # Fill up the rest of the places with sampled examples with replacement.
    if m >= n:
        extra = torch.randint(0, n, (m - n,), generator=generator)
        indices = torch.cat((indices, extra))

    indices = indices[torch.randperm(len(indices), generator=generator)]

    # Select samples
    x1_first_sampled = first_part_x1[indices]
    x2_first_sampled = first_part_x2[indices]
    y_first_sampled = first_part_y[indices]

    # Concatenate with the eval/test part
    x1_return = torch.cat((x1_first_sampled, x[1][n:]), dim=0)
    x2_return = torch.cat((x2_first_sampled, x[2][n:]), dim=0)
    targets_return = torch.cat((y_first_sampled, y[n:]), dim=0)

    data_return = (None, x1_return, x2_return)
    return data_return, targets_return


def permute_data(data, targets, eval_position, device="cuda:0"):

    import numpy as np

    generator = torch.Generator()
    rng = np.random.RandomState()
    gen_seed = rng.randint(1e7)
    generator.manual_seed(gen_seed)

    # Onto the other device
    data[1] = data[1].to(device)
    data[2] = data[2].to(device)
    targets = targets.to(device)

    first_part_x1 = data[1][:eval_position]
    first_part_x2 = data[2][:eval_position]
    first_part_y = targets[:eval_position]

    permutation = permutation = torch.randperm(eval_position, generator=generator)

    x1_first_shuffled = first_part_x1[permutation]
    x2_first_shuffled = first_part_x2[permutation]
    y_first_shuffled = first_part_y[permutation]

    x1_return = torch.cat((x1_first_shuffled, data[1][eval_position:]), dim=0)
    x2_return = torch.cat((x2_first_shuffled, data[2][eval_position:]), dim=0)
    targets_return = torch.cat((y_first_shuffled, targets[eval_position:]), dim=0)

    data_return = (None, x1_return, x2_return)

    data_return[1].to(device)
    data_return[2].to(device)
    data[1].to(device)
    data[2].to(device)
    targets_return.to(device)

    return data_return, targets_return

def train(priordataloader_class, 
          criterion, 
          encoder_generator, 
          emsize=200, 
          nhid=200, 
          nlayers=6,
          nhead=2, 
          dropout=0.0,
          epochs=10, 
          steps_per_epoch=100, 
          batch_size=200, 
          bptt=10, 
          lr=None, 
          weight_decay=0.0, 
          warmup_epochs=10, 
          input_normalization=False,
          y_encoder_generator=None, 
          pos_encoder_generator=None, 
          decoder=None, 
          extra_prior_kwargs_dict={}, 
          scheduler=get_cosine_schedule_with_warmup,
          load_weights_from_this_state_dict=None, 
          validation_period=10, 
          single_eval_pos_gen=None, 
          bptt_extra_samples=None, 
          gpu_device='cuda:0',
          aggregate_k_gradients=1, 
          verbose=True, 
          style_encoder_generator=None, 
          epoch_callback=None,
          initializer=None, 
          initialize_with_model=None, 
          train_mixed_precision=False, 
          efficient_eval_masking=True,
          evaluation_class: EvalHelper=None, 
          enable_autocast=True,
          permutation_repeat=0,
          bootstrap_samples=0,
          enable_data_parallel=False,
          config={},
          model_type="",    # mamba/transformer/hydra
          transformer_full_attn = False,
          curriculum_cfg={},
          **model_extra_args
          ):
    device = gpu_device if torch.cuda.is_available() else 'cpu:0'
    print(f'Using {device} device')
    using_dist, rank, device = init_dist(device)
    single_eval_pos_gen = single_eval_pos_gen if callable(single_eval_pos_gen) else lambda: single_eval_pos_gen

    def eval_pos_seq_len_sampler():
        single_eval_pos = single_eval_pos_gen()
        if bptt_extra_samples:
            return single_eval_pos, single_eval_pos + bptt_extra_samples
        else:
            return single_eval_pos, bptt
        

    curriculum_dls = {}
    if curriculum_cfg:    
        curriculum_dls = {k: 
                      priordataloader_class(
                          num_steps=steps_per_epoch, 
                          batch_size=batch_size, 
                          eval_pos_seq_len_sampler=v[1], 
                          seq_len_maximum=v[0][0]+(bptt_extra_samples if bptt_extra_samples else 0), 
                          device=device, 
                          **extra_prior_kwargs_dict)
                        for k,v in curriculum_cfg.items()}
    
    dl = priordataloader_class(num_steps=steps_per_epoch, batch_size=batch_size, eval_pos_seq_len_sampler=eval_pos_seq_len_sampler, seq_len_maximum=bptt+(bptt_extra_samples if bptt_extra_samples else 0), device=device, **extra_prior_kwargs_dict)

    encoder = encoder_generator(dl.num_features, emsize)
    #style_def = dl.get_test_batch()[0][0] # the style in batch of the form ((style, x, y), target, single_eval_pos)
    style_def = None
    #print(f'Style definition of first 3 examples: {style_def[:3] if style_def is not None else None}')
    style_encoder = style_encoder_generator(style_def.shape[1], emsize) if (style_def is not None) else None
    if isinstance(criterion, nn.GaussianNLLLoss):
        n_out = 2
    elif isinstance(criterion, nn.CrossEntropyLoss):
        n_out = criterion.weight.shape[0]
    else:
        n_out = 1

    if model_type == "transformer":
        model = TransformerModel(encoder, 
                                n_out, 
                                emsize, 
                                nhead, 
                                nhid, 
                                nlayers, 
                                dropout, 
                                style_encoder=style_encoder,
                                y_encoder=y_encoder_generator(1, emsize), 
                                input_normalization=input_normalization,
                                pos_encoder=(pos_encoder_generator or positional_encodings.NoPositionalEncoding)(emsize, bptt*2),
                                decoder=decoder, 
                                init_method=initializer, 
                                efficient_eval_masking=efficient_eval_masking, 
                                full_attention=transformer_full_attn
                                **model_extra_args
                                )
    elif model_type == "mamba":
        model = MambaModel(
            encoder=encoder,
            n_out=n_out,
            ninp=emsize,
            nhid=nhid,
            y_encoder=y_encoder_generator(1, emsize),
            num_layers=nlayers,
            device=device,
        )

    elif model_type == "hydra":
        model = HydraModel(
            encoder=encoder,
            n_out=n_out,
            ninp=emsize,
            nhid=nhid,
            y_encoder=y_encoder_generator(1, emsize),
            num_layers=nlayers,
            device=device
        )
        

    model.criterion = criterion
    if load_weights_from_this_state_dict is not None:
        model.load_state_dict(load_weights_from_this_state_dict)
    if initialize_with_model is not None:
        model.init_from_small_model(initialize_with_model)

    print(f"Using a {model_type} model with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters")

    try:
        for (k, v), (k2, v2) in zip(model.state_dict().items(), initialize_with_model.state_dict().items()):
            print(k, ((v - v2) / v).abs().mean(), v.shape)
    except Exception:
        pass

    model.to(device)
    if using_dist:
        print("Distributed training")
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    dl.model = model
    if curriculum_cfg:
        for _, dl in curriculum_dls.items():
            dl.model = model

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler(optimizer, warmup_epochs, epochs if epochs is not None else 100) # when training for fixed time lr schedule takes 100 steps

    scaler = GradScaler("cuda") if train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    def train_epoch(dl):
        model.train()  # Turn on the train mode
        total_loss = 0.
        total_positional_losses = 0.
        total_positional_losses_recorded = 0
        nan_steps = 0
        #ignore_steps = 0
        before_get_batch = time.time()
        assert len(dl) % aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'
        
        for batch, (data, targets, single_eval_pos) in enumerate(dl):

            for repeat in range(permutation_repeat + 1):

                # And single_eval_pos is a fix because the sample_train will else throw an error.
                if bootstrap_samples and single_eval_pos > 0:

                    print(f"Bootstrap samples to a context of length {bootstrap_samples}.")

                    data, targets = sample_train(data, targets, single_eval_pos, bootstrap_samples)
                    single_eval_pos = bootstrap_samples

                targets_original = targets

                if repeat > 0:   # Then shuffle
                    data, targets = permute_data(data, targets_original, eval_position=single_eval_pos)

                if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1):
                    cm = model.no_sync()
                else:
                    cm = nullcontext()
                with cm:
                    time_to_get_batch = time.time() - before_get_batch
                    before_forward = time.time()
                    if bptt_extra_samples is None:
                        single_eval_pos = single_eval_pos_gen() if callable(single_eval_pos_gen) else single_eval_pos_gen
                    else:
                        single_eval_pos = targets.shape[0] - bptt_extra_samples

                    with autocast("cuda", enabled=scaler is not None):
                        # If style is set to None, it should not be transferred to device
                        output = model(
                            tuple(
                                e.to(device) if torch.is_tensor(e) else e 
                                for e in data
                                ) 
                                if isinstance(data, tuple)

                            else data.to(device), 
                            single_eval_pos=single_eval_pos)

                        forward_time = time.time() - before_forward

                        if single_eval_pos is not None:
                            targets = targets[single_eval_pos:]
                        if isinstance(criterion, nn.GaussianNLLLoss):
                            assert output.shape[-1] == 2, \
                                'need to write a little bit of code to handle multiple regression targets at once'

                            mean_pred = output[..., 0]
                            var_pred = output[..., 1].abs()
                            losses = criterion(mean_pred.flatten(), targets.to(device).flatten(), var=var_pred.flatten())
                        elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                            losses = criterion(output.flatten(), targets.to(device).flatten())
                        elif isinstance(criterion, nn.CrossEntropyLoss):
                            # Original: losses = criterion(output.reshape(-1, n_out), targets.to(device).long().flatten())
                            # Done with single_eval_pos -> TODO
                            losses = criterion(output.reshape(-1, n_out), targets.to(device).long().flatten())
                        else:
                            losses = criterion(output, targets)
                        losses = losses.view(*output.shape[0:2])
                        #time.sleep(10)
                        loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
                        loss = loss / aggregate_k_gradients

                    if scaler: loss = scaler.scale(loss)
                    loss.backward()

                    if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                        if scaler: scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        try:
                            if scaler:
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                        except:
                            print("Invalid optimization step encountered")
                        optimizer.zero_grad()

                    step_time = time.time() - before_forward

                    if not torch.isnan(loss):
                        total_loss += losses.mean().cpu().detach().item()
                        total_positional_losses += losses.mean(1).cpu().detach() if single_eval_pos is None else \
                            nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)*\
                            losses[:bptt-single_eval_pos].mean().cpu().detach()

                        total_positional_losses_recorded += torch.ones(bptt) if single_eval_pos is None else \
                            nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)
                    nan_steps += nan_share
                    #ignore_steps += (targets == -100).float().mean()

                before_get_batch = time.time()

        return total_loss / (steps_per_epoch * (permutation_repeat + 1)), \
                [], \
                time_to_get_batch, \
                forward_time, \
                step_time, \
                nan_steps.cpu().item()/(batch+1),\
                0

    total_loss = float('inf')
    total_positional_losses = float('inf')

    print("Beginning the Training process")
    print(f"Total number of epochs: {epochs}")

    total_loss = float('inf')
    total_positional_losses = float('inf')
    try:
        for epoch in (range(1, epochs + 1) if epochs is not None else itertools.count(1)):

            # Check if curriculum learning requires new dataloader object -> update hte dataloader.
            if curriculum_cfg and epoch in curriculum_dls.keys():
                dl = curriculum_dls[epoch]

            epoch_start_time = time.time()
            total_loss, total_positional_losses, time_to_get_batch, forward_time, step_time, nan_share, ignore_share =\
                train_epoch(dl)
            if hasattr(dl, 'validate') and epoch % validation_period == 0:
                with torch.no_grad():
                    val_score = dl.validate(model)
            else:
                val_score = None

            if verbose:
                print('-' * 89)
                print(
                    f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.2f} | '
                    f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                    #f"lr {scheduler.get_last_lr()[0]}"
                    f' data time {time_to_get_batch:5.2f} step time {step_time:5.2f}'
                    f' forward time {forward_time:5.2f}' 
                    f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}'
                    + (f'val score {val_score}' if val_score is not None else ''))
                print('-' * 89)

            #
            # Wandb Logging
            #
            wandb_dict = {}
            wandb_dict[f"train/{model_type}_loss"] = total_loss
            wandb_dict["extras/nan_share"] = nan_share
            

            # Do other evaluations as well.
            if evaluation_class:
                metric_used = tabular_metrics.auc_metric
                eval_positions = [1000]
                eval_result = evaluation_class.do_evaluation(model=model, 
                                                             bptt=bptt,
                                                             eval_positions=eval_positions,
                                                             metric=metric_used, 
                                                             device=device, 
                                                             method_name="mamba")
                
                wandb_dict[f"test/{model_type}_mean_acc"] = eval_result

            wandb.log(wandb_dict)

            # stepping with wallclock time based scheduler
            if epoch_callback is not None and rank == 0:
                epoch_callback(model, epoch, config, model_type)
            scheduler.step()
    except KeyboardInterrupt:
        pass
    
    #
    # END Training Process
    #
    
    return total_loss, total_positional_losses, model.to('cpu'), dl
