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
from utils.util import save_model, to_var, W1Distance
from utils.prepare import create_main_loss
def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag
        
def train_model(R_model: nn.Module,D_model: nn.Module, data_loaders: Dict[str, DataLoader],
                R_loss_func: callable, D_loss_func: callable, optimizer_R: torch.optim, optimizer_D: torch.optim,
                model_folder: str, args, start_epoch=-1, **kwargs):
    num_epochs = args.epochs
    n_critic = getattr(args, "n_critic", 1)
    z_dim = getattr(args, "z_dim", 8)
    phases = [
        'train',
        'val',
        'test'
        ]
    w1 = W1Distance()
    since = time.perf_counter()
    for phase in phases:
        if phase not in data_loaders:
            raise KeyError(f"{phase} loader is missing from data_loaders")
        print(f"{phase} loader found with {len(data_loaders[phase])} batches")
        
    with open(model_folder + "/output.txt", "a") as f:
        f.write("REGRESSION MODEL:\n")
        f.write(str(R_model))
        f.write("\n\n")
        f.write("DISCRIMINATOR MODEL:\n")
        f.write(str(D_model))
        f.write("\n\n")

    R_save_dict, best_mae = {'state_dict': copy.deepcopy(R_model.state_dict()),
                           'epoch': 0
                           }, 10000
    D_save_dict = {'state_dict': copy.deepcopy(D_model.state_dict())}
    
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_R, mode='min', factor=.2, patience=2,
    #                                                  threshold=1e-2, threshold_mode='rel', min_lr=1e-7)

    try:
        patiance = 0
        for epoch in range(start_epoch + 1, num_epochs):
            running_loss_R = {phase: 0.0 for phase in phases}
            running_loss_D = 0.0
            running_loss_R_from_D = 0.0
            msg = []
            # training/val/test loop
            for phase in phases:
                if phase == 'train':
                    R_model.train()
                    D_model.train()
                else:
                    R_model.eval()
                    D_model.eval()
                    
                steps, predictions, targets = 0, list(), list()
                
                tqdm_loader = tqdm(data_loaders[phase],mininterval=3)
                for step, (features, truth_data) in enumerate(tqdm_loader):
                    steps += truth_data.size(0)
                    
                    features = to_var(features, args.device)
                    lens = features['lens']
                    
                    targets.append(truth_data.numpy())
                    truth_data = to_var(truth_data, args.device)
                    B = truth_data.size(0)
                    if phase == 'train':
                        # why noise???? --> so can model random event
                        # TODO: work on putting the noise into the generator/ regressor. However might be optional if BERT is already installed
                        # --- train critic ---
                        set_requires_grad(D_model, True)
                        set_requires_grad(R_model, False)
                        
                        for _ in range(n_critic):
                            with torch.no_grad():        
                                # z = torch.randn(B,z_dim).to(args.device)
                                fake_times,spatio_temporal_features,_,_ = R_model(features,args)
                                
                            D_fake, fake_imgs = D_model(spatio_temporal_features, fake_times, lens)
                            D_real, real_imgs = D_model(spatio_temporal_features, truth_data.unsqueeze(-1), lens)
                            gp = W1Distance.calculate_gp_v2(D_model,real_imgs,fake_imgs,lens,args.device)
                            loss_D,_ = w1(D_fake,D_real,gp)
                            
                            optimizer_D.zero_grad()
                            loss_D.backward()
                            optimizer_D.step()
                        # --- train regressor ---
                        set_requires_grad(D_model, False)
                        set_requires_grad(R_model, True)
                        
                        # z = torch.rand(B,z_dim).to(device=args.device)
                        fake_times, spatio_temporal_features,_, loss_1 = R_model(features,args)
                        
                        D_fake,_ = D_model(spatio_temporal_features, fake_times, lens)
                        loss_R_from_D,_ = w1(D_fake)
                        
                        # if getattr(args, "punish_only", False):    
                        #     loss_R_from_D = torch.relu(loss_R_from_D)
                        
                        loss_2 = R_loss_func(truth=truth_data, predict=fake_times)
                        
                        loss_R = create_main_loss(loss_1,loss_2,loss_R_from_D,args)
                        
                        optimizer_R.zero_grad()
                        loss_R.backward()
                        optimizer_R.step()
                        
                    else:
                        # --- TEST AND VALIDATION PHASE ---
                        with torch.no_grad():
                            fake_times, spatio_temporal_features, _, loss_1 = R_model(features,args)
                            
                            D_fake, fake_imgs = D_model(spatio_temporal_features, fake_times, lens)
                            D_real, real_imgs = D_model(spatio_temporal_features, truth_data.unsqueeze(-1), lens)
                                                        
                            loss_2 = R_loss_func(truth=truth_data, predict=fake_times)    
                            
                            loss_R = create_main_loss(loss_1,loss_2,None,args)

                    if phase == 'train':   
                        d_loss_str = f"D loss: {running_loss_D / steps :.8f}"
                        loss_from_D_str = f"lossRfromD: {running_loss_R_from_D / steps :.8f}"
                    desc = f"loss1: {loss_1.item()}, loss2: {loss_2.item()}, {(d_loss_str + loss_from_D_str) if phase == 'train' else ''}"
                    tqdm_loader.set_description(
                        f'{phase} epoch: {epoch}, {phase} loss: {(running_loss_R[phase] / steps) :.8f}, '
                        + desc
                    )
                        
                    with torch.no_grad():
                        predictions.append(fake_times.cpu().detach().numpy())

                    running_loss_R[phase] += loss_R.item() * truth_data.size(0)
                    if phase == 'train':  
                        running_loss_D += loss_D.item() * truth_data.size(0)
                        running_loss_R_from_D += loss_R_from_D.item() * truth_data.size(0)
                    if step % 1000 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

                torch.cuda.empty_cache()

                predictions = np.concatenate(predictions).copy()
                targets = np.concatenate(targets).copy()
                
                # assert predictions[0].shape == targets[0].shape, f'{predictions.shape}, {targets.shape}'
                
                scores = calculate_metrics(predictions.reshape(predictions.shape[0], -1),
                                           targets.reshape(targets.shape[0], -1), args, plot=epoch % 5 == 0, **kwargs)
                with open(model_folder+"/output.txt", "a") as f:
                    if phase == 'train':
                        f.write(f'{phase} epoch: {epoch}, {phase} loss: {running_loss_R[phase] / steps}, {phase} discriminator loss: {running_loss_D / steps}\n')
                        f.write(f'lossRfromD: {running_loss_R_from_D / steps}\n')
                    else:
                        f.write(f'{phase} epoch: {epoch}, {phase} loss: {running_loss_R[phase] / steps}\n')
                    f.write(str(scores))
                    f.write('\n')
                    f.write(str(time.time()))
                    f.write("\n\n")
                print(scores)
                if phase == 'train':
                    msg.append(f"{phase} epoch: {epoch}, {phase} loss: {running_loss_R[phase] / steps}, {phase} discriminator loss: {running_loss_D / steps}\n {scores}\n")
                else:
                    msg.append(f"{phase} epoch: {epoch}, {phase} loss: {running_loss_R[phase] / steps}\n {scores}\n")
                if phase == 'val':
                    if scores['MAE'] < best_mae:
                        best_mae = scores['MAE']
                        R_save_dict.update(
                            state_dict=copy.deepcopy(R_model.state_dict()),
                            epoch=epoch,
                            optimizer_state_dict=copy.deepcopy(optimizer_R.state_dict())
                        )
                        save_model(f"{model_folder}/best_R_model.pkl", **R_save_dict)
                        
                        D_save_dict.update(
                            state_dict=copy.deepcopy(D_model.state_dict()),
                            epoch=epoch,
                            optimizer_state_dict=copy.deepcopy(optimizer_D.state_dict())
                        )
                        save_model(f"{model_folder}/best_D_model.pkl", **D_save_dict)
                        
                        patiance = 0
                    else:
                        patiance += 1
                        print(f"Current MAE {scores['MAE']} more than best MAE {best_mae}, patience: {patiance}")

            if patiance >= args.patience:
                print(f"Early stop! best MAE: {best_mae}")
                break
            # scheduler.step(running_loss_R['val'])

    finally:
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")

        save_model(f"{model_folder}/best_R_model.pkl", **R_save_dict)
        save_model(f"{model_folder}/final_R_model.pkl",
                   **{'state_dict': copy.deepcopy(R_model.state_dict()),
                      'epoch': epoch,
                      'optimizer_state_dict': copy.deepcopy(optimizer_R.state_dict()),
                      })
        
        save_model(f"{model_folder}/best_D_model.pkl", **D_save_dict)
        save_model(f"{model_folder}/final_D_model.pkl",
                   **{'state_dict': copy.deepcopy(D_model.state_dict()),
                      'epoch': epoch,
                      'optimizer_state_dict': copy.deepcopy(optimizer_D.state_dict())
                      })
