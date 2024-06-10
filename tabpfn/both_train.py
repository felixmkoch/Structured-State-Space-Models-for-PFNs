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

class Losses():
    gaussian = nn.GaussianNLLLoss(full=True, reduction='none')
    mse = nn.MSELoss(reduction='none')
    def ce(num_classes):
        num_classes = num_classes.shape[0] if torch.is_tensor(num_classes) else num_classes
        return nn.CrossEntropyLoss(reduction='none', weight=torch.ones(num_classes))
    bce = nn.BCEWithLogitsLoss(reduction='none')



def train_both_models(priordataloader_class, 
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
          enable_autocast=True,
          num_mamba_layers=1, 
          **model_extra_args
          ):
    device = gpu_device if torch.cuda.is_available() else 'cpu:0'

    print(f'Using {device} device')

    using_dist, rank, device = init_dist(device)
    single_eval_pos_gen = single_eval_pos_gen if callable(single_eval_pos_gen) else lambda: single_eval_pos_gen

    def eval_pos_seq_len_sampler():
        single_eval_pos = single_eval_pos_gen()
        if bptt_extra_samples: return single_eval_pos, single_eval_pos + bptt_extra_samples
        else: return single_eval_pos, bptt
        

    dl = priordataloader_class(
        num_steps=steps_per_epoch, 
        batch_size=batch_size, 
        eval_pos_seq_len_sampler=eval_pos_seq_len_sampler, 
        seq_len_maximum=bptt+(bptt_extra_samples if bptt_extra_samples else 0), 
        device=device, 
        **extra_prior_kwargs_dict
        )

    encoder = encoder_generator(dl.num_features, emsize)
    style_def = None

    style_encoder = style_encoder_generator(style_def.shape[1], emsize) if (style_def is not None) else None
    if isinstance(criterion, nn.GaussianNLLLoss): n_out = 2
    elif isinstance(criterion, nn.CrossEntropyLoss): n_out = criterion.weight.shape[0]
    else: n_out = 1


    #
    # Transformer Model
    #
    transformer_model = TransformerModel(encoder, 
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
                             **model_extra_args
                             )
    
    #
    # MAMBA Model
    #

    mamba_model = MambaModel(
        encoder=encoder,
        n_out=n_out,
        ninp=emsize,
        nhid=nhid,
        y_encoder=y_encoder_generator(1, emsize),
        num_layers=num_mamba_layers,
        device=device,
    )

    transformer_model.criterion = criterion
    mamba_model.criterion = criterion

    print(f"Using a Transformer with {sum(p.numel() for p in transformer_model.parameters())/1000/1000:.{2}f} M parameters")

    transformer_model.to(device)
    mamba_model.to(device)

    if using_dist:
        print("Distributed training")
        model = torch.nn.parallel.DistributedDataParallel(transformer_model, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    dl.model = transformer_model

    transformer_optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=lr, weight_decay=weight_decay)
    transformer_scheduler = scheduler(transformer_optimizer, warmup_epochs, epochs if epochs is not None else 100) # when training for fixed time lr schedule takes 100 steps

    mamba_optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=lr, weight_decay=weight_decay)
    mamba_scheduler = scheduler(mamba_optimizer, warmup_epochs, epochs if epochs is not None else 100) # when training for fixed time lr schedule takes 100 steps

    transformer_scaler = GradScaler() if train_mixed_precision else None
    mamba_scaler = GradScaler() if train_mixed_precision else None

    # check that everything uses up-to-date APIs
    utils.check_compatibility(dl)

    def train_epoch():

        transformer_model.train()  # Turn on the train mode

        assert len(dl) % aggregate_k_gradients == 0, 'Please set the number of steps per epoch s.t. `aggregate_k_gradients` divides it.'

        # For Transformer
        transformer_total_loss = 0.
        transformer_total_positional_losses = 0.
        transformer_total_positional_losses_recorded = 0
        transformer_nan_steps = 0
        transformer_ignore_steps = 0
        transformer_before_get_batch = time.time()
        

        # For MAMBA
        mamba_total_loss = 0.
        mamba_total_positional_losses = 0.
        mamba_total_positional_losses_recorded = 0
        mamba_nan_steps = 0
        mamba_ignore_steps = 0
        mamba_before_get_batch = time.time()

        
        for batch, (data, targets, single_eval_pos) in enumerate(dl):

            #print(f"Currently in batch {batch + 1} out of {len(dl)} batches")

            # Others

            if bptt_extra_samples is None: single_eval_pos = single_eval_pos_gen() if callable(single_eval_pos_gen) else single_eval_pos_gen
            else: single_eval_pos = targets.shape[0] - bptt_extra_samples

            if single_eval_pos is not None: targets = targets[single_eval_pos:]

            #
            # For the transformer:
            #

            if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1): cm = transformer_model.no_sync()
            else: cm = nullcontext()
            
            with cm:
                transformer_time_to_get_batch = time.time() - transformer_before_get_batch
                before_forward = time.time()

                with autocast(enabled=transformer_scaler is not None):
                    # If style is set to None, it should not be transferred to device
                    transformer_output = transformer_model(
                        tuple(
                            e.to(device) if torch.is_tensor(e) 
                            else e 
                            for e in data
                            ) if isinstance(data, tuple) 
                            else data.to(device)
                                   , 
                                   single_eval_pos=single_eval_pos)

                    transformer_forward_time = time.time() - before_forward

                    if isinstance(criterion, nn.GaussianNLLLoss):
                        assert transformer_output.shape[-1] == 2, \
                            'need to write a little bit of code to handle multiple regression targets at once'

                        mean_pred = transformer_output[..., 0]
                        var_pred = transformer_output[..., 1].abs()
                        transformer_losses = criterion(mean_pred.flatten(), targets.to(device).flatten(), var=var_pred.flatten())
                    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                        transformer_losses = criterion(transformer_output.flatten(), targets.to(device).flatten())
                    elif isinstance(criterion, nn.CrossEntropyLoss):
                        transformer_losses = criterion(transformer_output.reshape(-1, n_out), targets.to(device).long().flatten())
                    else:
                        transformer_losses = criterion(transformer_output, targets)

                    #print(f"Size transformer after flatten {targets.to(device).long().flatten()}")
                    transformer_losses = transformer_losses.view(*transformer_output.shape[0:2])
                    transformer_loss, nan_share = utils.torch_nanmean(transformer_losses.mean(0), return_nanshare=True)
                    transformer_loss = transformer_loss / aggregate_k_gradients

                if transformer_scaler: transformer_loss = transformer_scaler.scale(transformer_loss)

                transformer_loss.backward()

                if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                    if transformer_scaler: transformer_scaler.unscale_(transformer_optimizer)
                    torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), 1.)
                    try:
                        if transformer_scaler:
                            transformer_scaler.step(transformer_optimizer)
                            transformer_scaler.update()
                        else:
                            transformer_optimizer.step()
                    except:
                        print("Invalid optimization step encountered")
                    transformer_optimizer.zero_grad()

                step_time = time.time() - before_forward

                if not torch.isnan(transformer_loss):
                    transformer_total_loss += transformer_losses.mean().cpu().detach().item()
                    transformer_total_positional_losses += transformer_losses.mean(1).cpu().detach() if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)*\
                        transformer_losses[:bptt-single_eval_pos].mean().cpu().detach()

                    transformer_total_positional_losses_recorded += torch.ones(bptt) if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)
                transformer_nan_steps += nan_share
                transformer_ignore_steps += (targets == -100).float().mean()


            transformer_before_get_batch = time.time()

            #
            # MAMBA Model 
            #

            if using_dist and not (batch % aggregate_k_gradients == aggregate_k_gradients - 1): cm = mamba_model.no_sync()
            else: cm = nullcontext()

            with cm:
                mamba_time_to_get_batch = time.time() - mamba_before_get_batch
                mamba_before_forward = time.time()

                #print(f"Autocast is {enable_autocast}")

                with autocast(enabled=enable_autocast):
                    # If style is set to None, it should not be transferred to device
                    mamba_output = mamba_model(
                        tuple(
                            e.to(device) if torch.is_tensor(e) else e 
                            for e in data
                            ) 
                            if isinstance(data, tuple)

                        else data.to(device), 
                        single_eval_pos=single_eval_pos)

                    mamba_forward_time = time.time() - before_forward

                    if isinstance(criterion, nn.GaussianNLLLoss):
                        assert mamba_output.shape[-1] == 2, \
                            'need to write a little bit of code to handle multiple regression targets at once'

                        mean_pred = mamba_output[..., 0]
                        var_pred = mamba_output[..., 1].abs()
                        mamba_losses = criterion(mean_pred.flatten(), targets.to(device).flatten(), var=var_pred.flatten())
                    elif isinstance(criterion, (nn.MSELoss, nn.BCEWithLogitsLoss)):
                        mamba_losses = criterion(mamba_output.flatten(), targets.to(device).flatten())
                    elif isinstance(criterion, nn.CrossEntropyLoss):
                        # Original: losses = criterion(output.reshape(-1, n_out), targets.to(device).long().flatten())
                        # Done with single_eval_pos -> TODO
                        # Also Long did note work.
                        mamba_losses = criterion(mamba_output.reshape(-1, n_out), targets.to(device).long().flatten())
                    else:
                        mamba_losses = criterion(mamba_output, targets)
                    mamba_losses = mamba_losses.view(*mamba_output.shape[0:2])
                    mamba_loss, mamba_nan_share = utils.torch_nanmean(mamba_losses.mean(0), return_nanshare=True)
                    mamba_loss = mamba_loss / aggregate_k_gradients

                if mamba_scaler: mamba_loss = mamba_scaler.scale(mamba_loss)
                mamba_loss.backward()

                if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                    if mamba_scaler: mamba_scaler.unscale_(mamba_optimizer)
                    torch.nn.utils.clip_grad_norm_(mamba_model.parameters(), 1.)
                    try:
                        if mamba_scaler:
                            mamba_scaler.step(mamba_optimizer)
                            mamba_scaler.update()
                        else:
                            mamba_optimizer.step()
                    except:
                        print("Invalid optimization step encountered")
                    mamba_optimizer.zero_grad()

                mamba_step_time = time.time() - mamba_before_forward

                if not torch.isnan(mamba_loss):
                    mamba_total_loss += mamba_losses.mean().cpu().detach().item()
                    mamba_total_positional_losses += mamba_losses.mean(1).cpu().detach() if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)*\
                        mamba_losses[:bptt-single_eval_pos].mean().cpu().detach()

                    mamba_total_positional_losses_recorded += torch.ones(bptt) if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)
                mamba_nan_steps += nan_share
                mamba_ignore_steps += (targets == -100).float().mean()

            mamba_before_get_batch = time.time()


            #
            # Return
            # 

        return \
            transformer_total_loss / steps_per_epoch, \
            (transformer_total_positional_losses / transformer_total_positional_losses_recorded).tolist(),\
            transformer_time_to_get_batch, \
            transformer_forward_time, \
            step_time, \
            transformer_nan_steps.cpu().item()/(batch+1), \
            transformer_ignore_steps.cpu().item()/(batch+1), \
            mamba_total_loss / steps_per_epoch, \
            (mamba_total_positional_losses / mamba_total_positional_losses_recorded).tolist(),\
            mamba_time_to_get_batch, \
            mamba_forward_time, \
            step_time, \
            mamba_nan_steps.cpu().item()/(batch+1), \
            mamba_ignore_steps.cpu().item()/(batch+1)

    transformer_total_loss = float('inf')
    mamba_total_loss = float('inf')
    transformer_total_positional_losses = float('inf')
    mamba_total_positional_losses = float('inf')
    try:
        cur_epoch = 0
        for epoch in (range(1, epochs + 1) if epochs is not None else itertools.count(1)):
            
            epoch_start_time = time.time()
            cur_epoch += 1
            print(f"{epoch_start_time}: Start epoch {cur_epoch}")

            transformer_total_loss, \
            transformer_total_positional_losses, \
            transformer_time_to_get_batch,\
            transformer_forward_time, \
            transformer_step_time, \
            transformer_nan_share, \
            transformer_ignore_share, \
            mamba_total_loss, \
            mamba_total_positional_losses, \
            mamba_time_to_get_batch,\
            mamba_forward_time, \
            mamba_step_time, \
            mamba_nan_share, \
            mamba_ignore_share =\
                train_epoch()
            if hasattr(dl, 'validate') and epoch % validation_period == 0:
                with torch.no_grad():
                    val_score = dl.validate(model)
            else:
                val_score = None

            if verbose:
                print('-' * 89)
                print("TRANSFORMER VERBOSE:")
                print('-' * 89)
                print(
                    f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {transformer_total_loss:5.2f} | '
                    f"pos losses {','.join([f'{l:5.2f}' for l in transformer_total_positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                    f' data time {transformer_time_to_get_batch:5.2f} step time {transformer_step_time:5.2f}'
                    f' forward time {transformer_forward_time:5.2f}' 
                    f' nan share {transformer_nan_share:5.2f} ignore share (for classification tasks) {transformer_ignore_share:5.4f}'
                    + (f'val score {val_score}' if val_score is not None else ''))
                print('-' * 89)
                print("MAMBA VERBOSE:")
                print('-' * 89)
                print(
                    f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {mamba_total_loss:5.2f} | '
                    f"pos losses {','.join([f'{l:5.2f}' for l in mamba_total_positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                    f' data time {mamba_time_to_get_batch:5.2f} step time {mamba_step_time:5.2f}'
                    f' forward time {mamba_forward_time:5.2f}' 
                    f' nan share {mamba_nan_share:5.2f} ignore share (for classification tasks) {mamba_ignore_share:5.4f}'
                    + (f'val score {val_score}' if val_score is not None else ''))
                print('-' * 89)


            # stepping with wallclock time based scheduler
            if epoch_callback is not None and rank == 0:
                epoch_callback(model, epoch / epochs)
            transformer_scheduler.step()
            mamba_scheduler.step()

    except KeyboardInterrupt:
        pass

    if rank == 0: # trivially true for non-parallel training
        #if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        #    model = model.module
        #    dl = None
        return \
            (transformer_total_loss, transformer_total_positional_losses, transformer_model.to('cpu'), dl), \
            (mamba_total_loss, mamba_total_positional_losses, mamba_model.to('cpu'), dl)
    



def _parse_args(config_parser, parser):
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text
