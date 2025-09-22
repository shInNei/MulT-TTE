import os
import torch
from torch.autograd import Variable


class StandardScaler2:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def save_model(path: str, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict, path)

def to_var(var, device=0):
    if torch.is_tensor(var):
        var = Variable(var)
        if torch.cuda.is_available():
            var = var.to(device)
        return var
    if isinstance(var, int) or isinstance(var, float):
        return var
    if isinstance(var, dict):
        for key in var:
            var[key] = to_var(var[key], device)
        return var
    if isinstance(var, list):
        var = map(lambda x: to_var(x, device), var)
        return var



class W1Distance:
    @staticmethod
    def calculate_gp_v2(model: torch.nn.Module, real_images: torch.Tensor, fake_images: torch.Tensor, lens, device, lambda_gp=10.0):
        B, _, _ = real_images.shape
        epsilon = torch.rand(B, 1,1, device=device).expand_as(real_images) 
        
        interpolated = epsilon * real_images + (1 - epsilon) * fake_images
        interpolated.requires_grad_(True)
        model_interpolated,_ = model(interpolated, None, lens)
        
        grads = torch.autograd.grad(
            outputs=model_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(model_interpolated, device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        grad_norm = grads.view(B,-1).norm(2,dim=1)
        gp = lambda_gp * ((grad_norm - 1) ** 2).mean()
        
        return gp
     
    def __call__(self, D_fake, D_real=None,GP=None):
        wasserstein_est = None
        if D_real is not None and D_fake is not None:
            # for discriminator
            wasserstein_est = D_real.mean().item() - D_fake.mean().item()
            loss =  - wasserstein_est + GP
        else:
            # for generator
            loss = -torch.mean(D_fake)
        return loss, wasserstein_est
