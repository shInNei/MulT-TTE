import os
import sys
import shutil
from tqdm import tqdm

import torch
from torch import optim
import numpy as np

from train.train_model import train_model
from utils.prepare import create_model, create_loss
from utils.prepare import load_datadict, load_datadoct_pre
from utils.metric import calculate_metrics
from utils.util import to_var, W1Distance
import time


def test_model(model, data_loader, args):
    model.eval()
    predictions = list()
    targets = list()
    inds = list()
    tqdm_loader = tqdm(data_loader)
    for step, (features, truth_data) in enumerate(tqdm_loader):
        if isinstance(features, dict) and 'inds' in features.keys():
            inds.append(features['inds'])
        features = to_var(features, args.device)
        truth_data = to_var(truth_data, args.device)

        outputs, _ = model(features, args)

        targets.append(truth_data.cpu().numpy())
        predictions.append(outputs.cpu().detach().numpy())
    pre2 = np.concatenate(predictions).squeeze()
    tar2 = np.concatenate(targets)
    if len(inds) > 0:
        print(f"test size: {len(inds)}")
        print(f"test traj ids of a batch: {inds[0]}")
        inds = np.concatenate(inds)
    else:
        inds = None
    metric = calculate_metrics(pre2, tar2, args, plot=True, inds=inds)
    print(metric)
    with open(f'{args.absPath}/data/result_{args.model}.txt', 'a') as f:
        f.write(time.strftime("%m/%d %H:%M:%S",time.localtime(time.time())))
        f.write(f"epoch:{args.epochs} lr:{args.lr}\ndataset:{args.dataset} identify:{args.identify}\nloss:{args.loss}\n")
        f.write(f"{args.model_config}\n")
        f.write(f"{args.data_config}\n")
        f.write(f"{metric}\n\n")

    np.save(os.path.join(args.model_folder, "result.npy"), np.asarray([pre2, inds]))


def train_main(args):
    if args.model == 'None':
        print('No chosen model')
        sys.exit(0)
    print(f"{args.mode} {args.model}_{args.identify} on {args.dataset}")
    
    load_datadoct_pre(args)
    data_loaders, scaler = load_datadict(args)
    args.scaler = scaler
    
    R_model, D_model = create_model(args)
    
    R_loss_func = create_loss(args)
    D_loss_func = W1Distance()
    
    model_folder = f'{args.absPath}/data/save_models/{args.model}_{args.identify}_{args.dataset}'
    args.model_folder = model_folder
    R_model = R_model.to(args.device)
    D_model = D_model.to(args.device)
    
    print(f'R loss function: {args.loss}')
    print(f'model config: {args.model_config}')
    print(f'data config: {args.data_config}')
    print(f'arg: {args}')
    # print(R_model)
    # 训练
    if args.mode == 'train':
        if os.path.exists(model_folder):
            shutil.rmtree(model_folder, ignore_errors=True)
        os.makedirs(model_folder, exist_ok=True)

        if args.optim == "Adam":
            optimizer_R = optim.Adam(R_model.parameters(), lr=args.lr)
            optimizer_D = optim.Adam(D_model.parameters(), lr=args.lr)
        elif args.optim == "AdamW":
            optimizer_R = optim.AdamW(R_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer_D = optim.AdamW(D_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError()

        train_model(R_model=R_model,D_model=D_model, data_loaders=data_loaders,
                    R_loss_func=R_loss_func,D_loss_func=D_loss_func, optimizer_R=optimizer_R, optimizer_D=optimizer_D,
                    model_folder=model_folder, args=args)
        
    elif args.mode == 'resume':
        if args.optim == "Adam":
            optimizer_R = optim.Adam(R_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer_D = optim.Adam(D_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optim == "AdamW":
            optimizer_R = optim.AdamW(R_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer_D = optim.AdamW(D_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError()
        
        final_R_model = torch.load(os.path.join(model_folder, 'final_R_model.pkl'), map_location=args.device)
        final_D_model = torch.load(os.path.join(model_folder, 'final_D_model.pkl'), map_location=args.device)
        start_epoch = final_R_model['epoch']
        R_model.load_state_dict(final_R_model['state_dict'], strict=False)
        D_model.load_state_dict(final_D_model['state_dict'], strict=False)
        optimizer_R.load_state_dict(final_R_model['optimizer_state_dict'])
        optimizer_D.load_state_dict(final_D_model['optimizer_state_dict'])
        train_model(R_model=R_model,D_model=D_model, data_loaders=data_loaders,
                    R_loss_func=R_loss_func,D_loss_func=D_loss_func, optimizer_R=optimizer_R, optimizer_D=optimizer_D,
                    model_folder=model_folder, args=args)
    # print(args.loss)
    # print(args.model_config)
    # print(args.data_config)
    # print(model_folder)
    
    # model.regressor.load_state_dict(torch.load(os.path.join(model_folder, 'best_model.pkl'), map_location=args.device)['R_state_dict'], strict=False)
    # model.discriminator.load_state_dict(torch.load(os.path.join(model_folder, 'best_model.pkl'), map_location=args.device)['D_state_dict'], strict=False)
    
    # test_model(model, data_loaders['test'], args)

