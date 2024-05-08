import torch
import torch.nn as nn
import torch.nn.functional as F
from UNet import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        # Config
        self.loss = nn.MSELoss()
        self.unet = Unet(3)
        self.T = 70
        self.beta_1 = 0.05
        self.beta_T = 0.1
        t_s = torch.arange(1, self.T+1, device=device)

        # Scheduler
        self.beta_t = ((self.beta_T - self.beta_1)/(self.T-1)) * (t_s-1) + self.beta_1
        self.sqrt_beta_t = self.beta_t ** 0.5
        self.alpha_t = 1 - self.beta_t
        alpha_t_bar_list = [self.alpha_t[0]]
        alpha_t_bar = self.alpha_t[0]
        for t in range(1, self.T):
            alpha_t_bar = alpha_t_bar * self.alpha_t[t]
            alpha_t_bar_list.append(alpha_t_bar)
        self.alpha_t_bar = torch.tensor(alpha_t_bar_list, device=device)
        self.sqrt_alpha_t_bar = self.alpha_t_bar ** 0.5
        self.sqrt_one_minus_alpha_t_bar = (1-self.alpha_t_bar) ** 0.5
        self.c1 = 1/(self.alpha_t ** 0.5)
        self.c2 = (1 - self.alpha_t)/self.sqrt_one_minus_alpha_t_bar
        
    def forward(self, images):
        T = self.T
        B = images.shape[0]
        t = torch.randint(1, T+1, (B, 1), device=device)
        t_normal = (t-1)/(T-1)
        eps = torch.randn_like(images, device=device)
        x_t = (self.sqrt_alpha_t_bar[t-1].reshape(-1, 1, 1, 1) * 
               images + 
               self.sqrt_one_minus_alpha_t_bar[t-1].reshape(-1, 1, 1, 1) * eps)
        predicted_eps = self.unet(x_t, t_normal)
        l = self.loss(predicted_eps, eps)
        return l

    def sample(self, B):
        T = self.T
        with torch.no_grad():
            X_t = torch.randn(B, 3, 64, 64, device=device)
            for t in range(T, 0, -1):
                t_normal = torch.full((B, 1), (t-1)/(T-1), device=device)
                z = torch.randn_like(X_t, device=device)
                var = (self.sqrt_beta_t[t-1]) * z
                if t <= 1:
                    var = 0
                X_t = (self.c1[t-1] * (X_t - self.c2[t-1] * self.unet(X_t, t_normal)) + var)
        return X_t.clamp(0,1)
