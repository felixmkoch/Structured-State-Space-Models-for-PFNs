#------------------------------------------------------------------------------------------------
#                                      IMPORTS
#------------------------------------------------------------------------------------------------

import os
import itertools
import argparse
import time
import datetime
import yaml
from contextlib import nullcontext


import torch
from torch import nn

import tabpfn.utils as utils
from tabpfn.transformer import TransformerModel
from mamba import MambaModel
from tabpfn.utils import get_cosine_schedule_with_warmup, get_openai_lr, StoreDictKeyPair, get_weighted_single_eval_pos_sampler, get_uniform_single_eval_pos_sampler
import tabpfn.priors as priors
import tabpfn.encoders as encoders
import tabpfn.positional_encodings as positional_encodings
from tabpfn.utils import init_dist
from torch.cuda.amp import autocast, GradScaler
from torch import nn

#------------------------------------------------------------------------------------------------
#                                    END IMPORTS
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                    CLASS LOSSES
#------------------------------------------------------------------------------------------------


class Losses():
    gaussian = nn.GaussianNLLLoss(full=True, reduction='none')
    mse = nn.MSELoss(reduction='none')
    def ce(num_classes):
        num_classes = num_classes.shape[0] if torch.is_tensor(num_classes) else num_classes
        return nn.CrossEntropyLoss(reduction='none', weight=torch.ones(num_classes))
    bce = nn.BCEWithLogitsLoss(reduction='none')
    
#------------------------------------------------------------------------------------------------
#                                  END CLASS LOSSES
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                MAMBA TRAIN FUNCTION
#------------------------------------------------------------------------------------------------

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
    
    #
    # DataLoader Initialization
    #
    dl = priordataloader_class(
        num_steps=steps_per_epoch, 
        batch_size=batch_size, 
        eval_pos_seq_len_sampler=eval_pos_seq_len_sampler, 
        seq_len_maximum=bptt+(bptt_extra_samples if bptt_extra_samples else 0), 
        device=device, 
        **extra_prior_kwargs_dict
    )

    # Encoder
    encoder = encoder_generator(dl.num_features, emsize)
    if isinstance(criterion, nn.GaussianNLLLoss): n_out = 2
    elif isinstance(criterion, nn.CrossEntropyLoss): n_out = criterion.weight.shape[0]
    else: n_out = 1

    #
    # Transformer Model
    #
    model = TransformerModel(encoder, 
                             n_out, 
                             emsize, 
                             nhead, 
                             nhid, 
                             nlayers, 
                             dropout, 
                             style_encoder=None,
                             y_encoder=y_encoder_generator(1, emsize), 
                             input_normalization=input_normalization,
                             pos_encoder=(pos_encoder_generator or positional_encodings.NoPositionalEncoding)(emsize, bptt*2),
                             decoder=decoder, 
                             init_method=initializer, 
                             efficient_eval_masking=efficient_eval_masking, 
                             **model_extra_args
                             )
    
    #
    # END Transformer Model
    #
    
    #
    # MAMBA Model
    #

    # mamba_model = MambaModel()
    
    
    
    #
    # END MAMBA Model
    #
    
    # Check if model should be loaded from state dict
    model.criterion = criterion

    print(f"Using a Transformer with {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters")

    try:
        for (k, v), (k2, v2) in zip(model.state_dict().items(), initialize_with_model.state_dict().items()):
            print(k, ((v - v2) / v).abs().mean(), v.shape)
    except Exception:
        pass

    # Specify whether model should be trained on CPU or GPU
    model.to(device)
    
    # Init Data Loader and Optimizer
    dl.model = model
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler(optimizer, warmup_epochs, epochs if epochs is not None else 100) # when training for fixed time lr schedule takes 100 steps
    scaler = GradScaler() if train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    #
    # Train Function
    #
    
    def train_epoch():
        model.train()  # Turn on the train mode
        total_loss = 0.
        total_positional_losses = 0.
        total_positional_losses_recorded = 0
        nan_steps = 0
        ignore_steps = 0
        before_get_batch = time.time()
        assert len(dl) % aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'
        
        # Batch [int] is just a counter from 0 to num_batches.
        # Data [tuple] Of length 3 [BPTT, batch_size/aggregate_k_gradients, num_features] [BPTT, batch_size/aggregate_k_gradients]
        # Targets [Tensor] is a tensor of 1. and 0.. [BPTT, batch_size/aggregate_k_gradients]
        # Note: Targets and Data[3] seem to be the same.
        # Single_eval_pos idk what this does.
        
        for batch, (data, targets, single_eval_pos) in enumerate(dl):
            
            #print(f"Data type: {type(data)}")
            #print(f"Len of data tuple: {len(data)}")
            #print(f"Shape of data: {data[1].size()} and {data[2].size()}")
            #print(f"Data Value: {data}")
            #print(f"Targets Value: {targets}")
            #print(f"Shape of target: {targets.size()}")

            print(f"Currently in batch {batch + 1} out of {len(dl)} batches")
            
            
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

                with autocast(enabled=scaler is not None):
                    # If style is set to None, it should not be transferred to device
                    output = model(tuple(e.to(device) if torch.is_tensor(e) else e for e in data) if isinstance(data, tuple) else data.to(device)
                                   , single_eval_pos=single_eval_pos)

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
                        losses = criterion(output.reshape(-1, n_out), targets.to(device).long().flatten())
                    else:
                        losses = criterion(output, targets)
                    losses = losses.view(*output.shape[0:2])
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
                ignore_steps += (targets == -100).float().mean()

            before_get_batch = time.time()
            
            
        return total_loss / steps_per_epoch, \
                (total_positional_losses / total_positional_losses_recorded).tolist(), \
                time_to_get_batch, \
                forward_time, \
                step_time, \
                nan_steps.cpu().item()/(batch+1),\
                ignore_steps.cpu().item()/(batch+1)
               
    #
    # END Train Function
    #
    
    #
    # Training Process
    #
    
    print("BEGIN TRAINIG PROCESS")
    print(f"NUM EPOCHS: {epochs}")

    total_loss = float('inf')
    total_positional_losses = float('inf')
    try:
        for epoch in (range(1, epochs + 1) if epochs is not None else itertools.count(1)):
            
            print("----------------- EPOCH START ---------------------")

            epoch_start_time = time.time()
            total_loss, total_positional_losses, time_to_get_batch, forward_time, step_time, nan_share, ignore_share =\
                train_epoch()
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
                    f' data time {time_to_get_batch:5.2f} step time {step_time:5.2f}'
                    f' forward time {forward_time:5.2f}' 
                    f' nan share {nan_share:5.2f} ignore share (for classification tasks) {ignore_share:5.4f}'
                    + (f'val score {val_score}' if val_score is not None else ''))
                print('-' * 89)
                
            print("------------------ EPOCH END ----------------------")

            # stepping with wallclock time based scheduler
            if epoch_callback is not None and rank == 0:
                epoch_callback(model, epoch / epochs)
            scheduler.step()
    except KeyboardInterrupt:
        pass
    
    #
    # END Training Process
    #
    
    return total_loss, total_positional_losses, model.to('cpu'), dl
    
    
#------------------------------------------------------------------------------------------------
#                              END MAMBA TRAIN FUNCTION
#------------------------------------------------------------------------------------------------