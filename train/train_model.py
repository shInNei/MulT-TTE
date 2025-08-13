import copy
import time
from typing import Dict
import gc

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.metric import calculate_metrics
from utils.util import save_model, to_var

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag
        
def train_model(model: nn.Module, data_loaders: Dict[str, DataLoader],
                loss_func: callable, optimizer_R: optim, optimizer_D: optim,
                model_folder: str, args, start_epoch=-1, **kwargs):
    num_epochs = args.epochs
    beta = args.beta
    theta = args.theta
    since = time.perf_counter()

    if "train" not in data_loaders:
        raise KeyError("train loader is missing from data_loaders")
    print(f"train loader found with {len(data_loaders['train'])} batches")
        
    with open(model_folder + "/output.txt", "a") as f:
        f.write(str(model))
        f.write("\n\n")

    save_dict = {
        'R_state_dict': copy.deepcopy(model.regressor.state_dict()),
        'D_state_dict': copy.deepcopy(model.discriminator.state_dict()), 
        'epoch': 0
    }
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_R, mode='min', factor=.2, patience=2,
        threshold=1e-2, threshold_mode='rel', min_lr=1e-7, verbose=True
    )

    try:
        for epoch in range(start_epoch + 1, num_epochs):
            running_loss = 0.0
            train_discriminator = (epoch % 2 == 0)

            set_requires_grad(model.regressor, True)
            set_requires_grad(model.discriminator, train_discriminator)

            steps, predictions, targets = 0, [], []
            tqdm_loader = tqdm(data_loaders["train"], mininterval=3)

            for step, (features, truth_data) in enumerate(tqdm_loader):
                steps += truth_data.size(0)
                features = to_var(features, args.device)
                
                targets.append(truth_data.numpy())
                truth_data = to_var(truth_data, args.device)
                args.real_time = truth_data

                ### Train regressor
                set_requires_grad(model.regressor, True)
                set_requires_grad(model.discriminator, False)
                
                optimizer_R.zero_grad()
                outputs, loss_1, _, loss_fake_R = model(features, args)
                loss_2 = loss_func(truth=truth_data, predict=outputs)
                loss = (1 - beta) * loss_1 / (loss_1 / loss_2 + 1e-4).detach() \
                       + beta * loss_2 + theta * loss_fake_R
                
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad.clip_grad_norm_(model.regressor.parameters(), 50)
                optimizer_R.step()
                
                ### Train discriminator (every other epoch)
                if train_discriminator:
                    set_requires_grad(model.regressor, False)
                    set_requires_grad(model.discriminator, True)
                    
                    optimizer_D.zero_grad()
                    _, _, loss_D, _ = model(features, args)
                    loss_D.backward()
                    torch.nn.utils.clip_grad.clip_grad_norm_(model.discriminator.parameters(), 50)
                    optimizer_D.step()

                tqdm_loader.set_description(
                    f'train epoch: {epoch}, train loss: {(running_loss / steps) :.8f}, '
                    f'loss1: {loss_1.item()}, loss2: {loss_2.item()}'
                )

                predictions.append(outputs.cpu().detach().numpy())
                running_loss += loss.item() * truth_data.size(0)

                if step % 1000 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            torch.cuda.empty_cache()
            scheduler.step(running_loss / steps)

            # Save after each epoch
            save_dict.update(
                R_state_dict=copy.deepcopy(model.regressor.state_dict()),
                D_state_dict=copy.deepcopy(model.discriminator.state_dict()),
                epoch=epoch,
                optimizer_R_state_dict=copy.deepcopy(optimizer_R.state_dict()),
                optimizer_D_state_dict=copy.deepcopy(optimizer_D.state_dict())
            )
            save_model(f"{model_folder}/latest_model.pkl", **save_dict)

    finally:
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")
        save_model(f"{model_folder}/final_model.pkl", **save_dict)
