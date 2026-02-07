from torch import nn
import torch
import time
import wandb
from torch.amp import GradScaler, autocast
from contextlib import nullcontext

from hydrapfn.utils import get_cosine_schedule_with_warmup
from hydrapfn.utils import init_dist
import hydrapfn.utils as utils
from tabpfn.scripts import tabular_metrics
from hydrapfn.hydra import HydraModel
from hydra_evaluation_helper import EvalHelper


class Losses():
    gaussian = nn.GaussianNLLLoss(full=True, reduction='none')
    mse = nn.MSELoss(reduction='none')
    def ce(num_classes):
        num_classes = num_classes.shape[0] if torch.is_tensor(num_classes) else num_classes
        return nn.CrossEntropyLoss(reduction='none', weight=torch.ones(num_classes))
    bce = nn.BCEWithLogitsLoss(reduction='none')


def train(
        priordataloader_class,
        criterion,
        encoder_generator,
        emsize: int = 128,
        nhid: int = 128,
        nlayers: int = 8,
        epochs: int = 10,
        steps_per_epoch: int = 100,
        batch_size: int = 128,
        bptt: int = 128,
        lr: float = 0.0001,
        weight_decay: float = 0.0,
        warmup_epochs = 10,
        y_encoder_generator = None,
        extra_prior_kwargs_dict={}, 
        scheduler=get_cosine_schedule_with_warmup,
        single_eval_pos_gen=None,
        device: str = 'cuda:0',
        aggregate_k_gradients=1,
        train_mixed_precision=False, 
        evaluation_class: EvalHelper=None, 
        config={},
        **model_extra_args
):
    
    #
    print(f'Using device {device}')
    using_dist, rank, device = init_dist(device)

    #-----------------------------------------------------------------------------
    #                      Initialize Datloader et al
    #-----------------------------------------------------------------------------
    single_eval_pos_gen = single_eval_pos_gen if callable(single_eval_pos_gen) else lambda: single_eval_pos_gen
    def eval_pos_seq_len_sampler():
        single_eval_pos = single_eval_pos_gen()
        return single_eval_pos, bptt
    
    dl = priordataloader_class(num_steps=steps_per_epoch, batch_size=batch_size, eval_pos_seq_len_sampler=eval_pos_seq_len_sampler, seq_len_maximum=bptt, device=device, **extra_prior_kwargs_dict)

    encoder = encoder_generator(dl.num_features, emsize)

    if isinstance(criterion, nn.GaussianNLLLoss): n_out = 2
    elif isinstance(criterion, nn.CrossEntropyLoss): n_out = criterion.weight.shape[0]
    else: n_out = 1

    #-----------------------------------------------------------------------------
    #                            Model Definition
    #-----------------------------------------------------------------------------

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

    print(f"Numer of Parameter in model {sum(p.numel() for p in model.parameters())/1000/1000:.{2}f} M parameters")

    model.to(device)
    dl.model = model    # Model attatched to dataloader as well.

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = scheduler(optimizer, warmup_epochs, epochs)
    scaler = GradScaler("cuda") if train_mixed_precision else None

    #-----------------------------------------------------------------------------
    #                  Definition of the training for one epoch
    #-----------------------------------------------------------------------------

    def train_epoch():
        model.train()
        total_loss = 0.
        total_positional_losses = 0.
        total_positional_losses_recorded = 0
        nan_steps = 0

        for batch, (data, targets, single_eval_pos) in enumerate(dl):
            cm = nullcontext()
            with cm:
                single_eval_pos = single_eval_pos_gen() if callable(single_eval_pos_gen) else single_eval_pos_gen
                with autocast("cuda", enabled=scaler is not None):
                    output = model(
                        tuple(
                            e.to(device) if torch.is_tensor(e) else e 
                            for e in data
                            ) 
                            if isinstance(data, tuple)

                        else data.to(device), 
                        single_eval_pos=single_eval_pos
                    )

                    if single_eval_pos is not None:
                        targets = targets[single_eval_pos:]

                    # Because we use the CrossEntropyLoss here. If this changes, you also need to change this one here.
                    losses = criterion(output.reshape(-1, n_out), targets.to(device).long().flatten())
                    losses = losses.view(*output.shape[0:2])
                    loss, nan_share = utils.torch_nanmean(losses.mean(0), return_nanshare=True)
                    loss = loss / aggregate_k_gradients

                if scaler: 
                    loss = scaler.scale(loss)
                loss.backward()

                if batch % aggregate_k_gradients == aggregate_k_gradients - 1:
                    if scaler: 
                        scaler.unscale_(optimizer)
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
                
                if not torch.isnan(loss):
                    total_loss += losses.mean().cpu().detach().item()
                    total_positional_losses += losses.mean(1).cpu().detach() if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)*\
                        losses[:bptt-single_eval_pos].mean().cpu().detach()

                    total_positional_losses_recorded += torch.ones(bptt) if single_eval_pos is None else \
                        nn.functional.one_hot(torch.tensor(single_eval_pos), bptt)
                nan_steps += nan_share

        return total_loss / (steps_per_epoch), total_positional_losses, nan_steps.cpu().item()/(batch+1)
    

    #-----------------------------------------------------------------------------
    #                       Prepare Training Loops
    #-----------------------------------------------------------------------------

    print(f"Total number of epochs: {epochs}")
    total_loss = float('inf')
    total_positional_losses = [float('inf')]

    try:
        for epoch in (range(1, epochs + 1)):

            epoch_start_time = time.time()
            total_loss, total_positional_losses, nan_share = train_epoch()

            print('-' * 89)
            print(
                f'| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s | mean loss {total_loss:5.2f} | '
                f"pos losses {','.join([f'{l:5.2f}' for l in total_positional_losses])}, lr {scheduler.get_last_lr()[0]}"
                f' nan share {nan_share:5.2f}')
            print('-' * 89)

            # Wandb Logging
            wandb_dict = {}
            wandb_dict[f"train/loss"] = total_loss
            wandb_dict["extras/nan_share"] = nan_share
            

            # Do other evaluations as well.
            if evaluation_class:
                metric_used = tabular_metrics.auc_metric
                eval_result = evaluation_class.do_evaluation(model=model, 
                                                             bptt=bptt,
                                                             eval_positions=[1000],
                                                             metric=metric_used, 
                                                             device=device, 
                                                             method_name="hydra")
                
                wandb_dict[f"test/mean_acc"] = eval_result

            wandb.log(wandb_dict)

            scheduler.step()
    except KeyboardInterrupt:
        pass

    return total_loss, total_positional_losses, model.to('cpu'), dl
