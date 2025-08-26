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
        
def train_model(R_model: nn.Module,D_model: nn.Module, data_loaders: Dict[str, DataLoader],
                R_loss_func: callable, D_loss_func: callable, optimizer_R: optim, optimizer_D: optim,
                model_folder: str, args, start_epoch=-1, **kwargs):
    num_epochs = args.epochs
    beta = args.beta
    theta = args.theta
    phases = ['train','val', 'test']
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
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_R, mode='min', factor=.2, patience=2,
                                                     threshold=1e-2, threshold_mode='rel', min_lr=1e-7)

    try:
        patiance = 0
        for epoch in range(start_epoch + 1, num_epochs):
            running_loss_R = {phase: 0.0 for phase in phases}
            running_loss_D = {phase: 0.0 for phase in phases} 
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
                    
                    targets.append(truth_data.numpy())
                    truth_data = to_var(truth_data, args.device)
                    
                    if phase == 'train':
                        ### train regressor
                        set_requires_grad(R_model, True)
                        set_requires_grad(D_model, False)
                        
                        optimizer_R.zero_grad()
                        outputs, spatio_temporal_features, loss_1 = R_model(features,args)
                        
                        lens = features['lens']
                        
                        D_output_fake, fake_imgs = D_model(spatio_temporal_features, outputs, lens)
                        D_output_real, real_imgs = D_model(spatio_temporal_features, truth_data.unsqueeze(-1), lens)
                         
                        loss_2 = R_loss_func(truth=truth_data, predict=outputs)
                        loss_R = D_loss_func(D_output_fake,D_output_real,D_loss_func.calculate_gradient_penalty(real_imgs, fake_imgs))
                        
                        loss = (1 - beta) * loss_1 / (loss_1 / loss_2 + 1e-4).detach() + beta * loss_2 + theta * loss_R
                        
                        loss.backward(retain_graph=True)
                        torch.nn.utils.clip_grad.clip_grad_norm_(R_model.regressor.parameters(), 50)  # after 50  # 20效果不佳，无法达到最优
                        optimizer_R.step()
                        
                        if epoch % args.epoch_cycle == 0:
                        # if True:
                            set_requires_grad(R_model, False)
                            set_requires_grad(D_model, True)
                            
                            optimizer_D.zero_grad()
                            
                            outputs, spatio_temporal_features, loss_1 = R_model(features, args)
                            D_output_fake = D_model(spatio_temporal_features, outputs, lens)
                            loss_D = D_loss_func(D_output_fake)

                            loss_D.backward()
                            torch.nn.utils.clip_grad.clip_grad_norm_(R_model.discriminator.parameters(), 50)  # after 50  # 20效果不佳，无法达到最优
                            optimizer_D.step()
                    else:
                        with torch.no_grad():
                            outputs, spatio_temporal_features, loss_1 = R_model(features,args)
                                                    
                            lens = features['lens']
                                                    
                            D_output_fake = D_model(spatio_temporal_features, outputs, lens)
                            D_output_real = D_model(spatio_temporal_features, truth_data, lens)      
                             
                            loss_2 = R_loss_func(truth=truth_data, predict=outputs)
                            loss_R = D_loss_func(D_output_fake,D_output_real)
                            loss_D = D_loss_func(D_output_fake)
                            
                            loss = (1 - beta) * loss_1 / (loss_1 / loss_2 + 1e-4).detach() + beta * loss_2 + theta * loss_R

                            
                    d_loss_str = f"D loss: {loss_D.item()}"
                    desc = f"loss1: {loss_1.item()}, loss2: {loss_2.item()}, {d_loss_str}"
                    tqdm_loader.set_description(
                        f'{phase} epoch: {epoch}, {phase} loss: {(running_loss_R[phase] / steps) :.8f}, '
                        + desc
                    )
                        
                        
                    # with torch.set_grad_enabled(phase == 'train'):
                    #     outputs, loss_1, loss_D, loss_fake_R = model(features, args)
                        
                    #     loss_2 = loss_func(truth=truth_data, predict=outputs)
                    #     loss = (1 - beta) * loss_1 / (loss_1 / loss_2 + 1e-4).detach() + beta * loss_2 + theta * loss_fake_R

                    #     tqdm_loader.set_description(
                    #         f'{phase} epoch: {epoch}, {phase} loss: {(running_loss[phase] / steps) :.8f}, '
                    #         f'loss1: {loss_1.item()}, loss2: {loss_2.item()}, discriminator loss: {loss_D.item()}')

                    #     if phase == 'train':
                    #         # train regressor
                    #         optimizer_R.zero_grad()
                    #         loss.backward(retain_graph=train_discriminator)
                    #         torch.nn.utils.clip_grad.clip_grad_norm_(model.regressor.parameters(), 50)  # after 50  # 20效果不佳，无法达到最优
                    #         optimizer_R.step()
                            
                    #         # train discriminator
                    #         if train_discriminator:
                    #             optimizer_D.zero_grad()
                    #             loss_D.backward()
                    #             torch.nn.utils.clip_grad.clip_grad_norm_(model.discriminator.parameters(), 50)
                    #             optimizer_D.step()

                    with torch.no_grad():
                        predictions.append(outputs.cpu().detach().numpy())

                    running_loss_R[phase] += loss.item() * truth_data.size(0)
                    running_loss_D[phase] += loss_D.item() * truth_data.size(0)
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
                    f.write(f'{phase} epoch: {epoch}, {phase} loss: {running_loss_R[phase] / steps}, {phase} discriminator loss: {running_loss_D[phase] / steps}\n')
                    f.write(str(scores))
                    f.write('\n')
                    f.write(str(time.time()))
                    f.write("\n\n")
                print(scores)
                msg.append(f"{phase} epoch: {epoch}, {phase} loss: {running_loss_R[phase] / steps}, {phase} discriminator loss: {running_loss_D[phase] / steps}\n {scores}\n")
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
            scheduler.step(running_loss_R['val'])

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
