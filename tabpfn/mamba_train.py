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
import wandb


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

from tabpfn.scripts import tabular_metrics

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
#                                  HELPER FUNCTIONS
#------------------------------------------------------------------------------------------------

def permute_data(data, targets):

    # Get the permutation of indices
    permutation = torch.randperm(data[1].size(0))

    data_return = (None, data[1][permutation], data[2][permutation])
    targets_return = targets[permutation]

    return data_return, targets_return

#------------------------------------------------------------------------------------------------
#                                 END HELPER FUNCTIONS
#------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------
#                                MAMBA TRAIN FUNCTION
#------------------------------------------------------------------------------------------------

def train_mamba(priordataloader_class, 
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
          enable_autocast=True,
          num_mamba_layers=2,
          evaluation_class=None,
          permutation_repeat=0,     # For each data-target-pair, we repeat the training permutation_repeat times with different data positions.
          enable_data_parallel=False,
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
    #print(f"Number of Dataloader Features: {dl.num_features}")
    encoder = encoder_generator(dl.num_features, emsize)
    if isinstance(criterion, nn.GaussianNLLLoss): n_out = 2
    elif isinstance(criterion, nn.CrossEntropyLoss): n_out = criterion.weight.shape[0]
    else: n_out = 1

    #
    # MAMBA Model
    #

    #print(f"Emsize: {emsize}")

    mamba_model = MambaModel(
        encoder=encoder,
        n_out=n_out,
        ninp=emsize,
        nhid=nhid,
        y_encoder=y_encoder_generator(1, emsize),
        num_layers=num_mamba_layers,
        device=device,
    )
    
    #
    # END MAMBA Model
    #
    
    # Check if model should be loaded from state dict
    mamba_model.criterion = criterion

    print(f"Using Mamba with {sum(p.numel() for p in mamba_model.parameters())/1000/1000:.{2}f} M parameters")

    #try:
    #    for (k, v), (k2, v2) in zip(model.state_dict().items(), initialize_with_model.state_dict().items()):
    #        print(k, ((v - v2) / v).abs().mean(), v.shape)
    #except Exception:
    #    pass

    # Specify whether model should be trained on CPU or GPU
    mamba_model.to(device)
    
    # Init Data Loader and Optimizer
    dl.model = mamba_model

    # Distributed Mode
    if enable_data_parallel:
        print("Distributed Training")
        mamba_model = torch.nn.DataParallel(mamba_model)

    optimizer = torch.optim.AdamW(mamba_model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler(optimizer, warmup_epochs, epochs if epochs is not None else 100) # when training for fixed time lr schedule takes 100 steps
    scaler = GradScaler() if enable_autocast else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    #
    # Train Function
    #
    
    def train_epoch():
        mamba_model.train()  # Turn on the train mode
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

            targets_original = targets

            for repeat in range(permutation_repeat + 1):

                if repeat > 0:   # Then shuffle
                    data, targets = permute_data(data, targets_original)

                if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1):
                    cm = mamba_model.no_sync()
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
                        output = mamba_model(
                            tuple(
                                e.to(device) if torch.is_tensor(e) else e 
                                for e in data
                                ) 
                                if isinstance(data, tuple)

                            else data.to(device), 
                            single_eval_pos=single_eval_pos)

                        forward_time = time.time() - before_forward

                        #print(f"Out shape: {output.size()}")
                        #print(f"Targets shape: {targets.size()}")

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
                        #print(f"Loss is: {losses}")
                        #time.sleep(10)
                        loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
                        loss = loss / aggregate_k_gradients
                        #print(f"Loss afterwards is: {loss}")

                    if scaler: loss = scaler.scale(loss)
                    #print(f"Loss inverted is: {loss}")
                    loss.backward()

                    if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                        if scaler: scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(mamba_model.parameters(), 1.)
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
                        
                        #print(f"Total Positional Losses: {total_positional_losses}")
                        #print(f"Total Positional Losses Recorded: {total_positional_losses_recorded}")
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
    
    print("Beginning the Training process")
    print(f"Total number of epochs: {epochs}")

    total_loss = float('inf')
    total_positional_losses = float('inf')
    try:
        for epoch in (range(1, epochs + 1) if epochs is not None else itertools.count(1)):
            
            #print("----------------- EPOCH START ---------------------")

            epoch_start_time = time.time()
            total_loss, total_positional_losses, time_to_get_batch, forward_time, step_time, nan_share, ignore_share =\
                train_epoch()
            if hasattr(dl, 'validate') and epoch % validation_period == 0:
                with torch.no_grad():
                    val_score = dl.validate(mamba_model)
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
                
            #print("------------------ EPOCH END ----------------------")

            #
            # Wandb Logging
            #
            wandb_dict = {}
            wandb_dict["train/mamba_loss"] = total_loss
            wandb_dict["extras/nan_share"] = nan_share
            

            # Do other evaluations as well.
            if evaluation_class:
                metric_used = tabular_metrics.auc_metric
                eval_positions = [1000]
                eval_result = evaluation_class.do_evaluation(model=mamba_model, 
                                                             bptt=bptt,
                                                             eval_positions=eval_positions,
                                                             metric=metric_used, 
                                                             device="cuda", 
                                                             method_name="mamba")
                
                wandb_dict["test/mamba_mean_acc"] = eval_result

            wandb.log(wandb_dict)

            # stepping with wallclock time based scheduler
            #if epoch_callback is not None and rank == 0:
            #    epoch_callback(model, epoch / epochs)
            scheduler.step()
    except KeyboardInterrupt:
        pass
    
    #
    # END Training Process
    #
    
    return total_loss, total_positional_losses, mamba_model.to('cpu'), dl
    
    
#------------------------------------------------------------------------------------------------
#                              END MAMBA TRAIN FUNCTION
#------------------------------------------------------------------------------------------------
