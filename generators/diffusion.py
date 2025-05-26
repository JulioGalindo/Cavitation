import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from generators.base import GenerativeModel
from utils.timeseries import TimeSeries

# Global optimizations for Apple MPS / CUDA / CPU
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')
torch.set_num_threads(os.cpu_count())

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, timesteps):
        device = timesteps.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(np.log(10000) / (half_dim - 1)))
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.SiLU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch)
        )
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, t_emb):
        h = self.block(x)
        time_emb = self.time_mlp(t_emb)[:, :, None]
        h = h + time_emb
        return nn.SiLU()(h + self.res_conv(x))

class SimpleUNet1D(nn.Module):
    def __init__(self, channels, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
        )
        self.down1 = ResidualBlock1D(1, channels, time_emb_dim)
        self.down2 = ResidualBlock1D(channels, channels, time_emb_dim)
        self.up1 = ResidualBlock1D(channels * 2, channels, time_emb_dim)
        self.up2 = ResidualBlock1D(channels, channels, time_emb_dim)
        self.final = nn.Conv1d(channels, 1, 1)
        self.pool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool(x1), t_emb)
        x3 = self.up1(torch.cat([self.upsample(x2), x1], dim=1), t_emb)
        x4 = self.up2(x3, t_emb)
        return self.final(x4)

class DiffusionGenerator(GenerativeModel):
    """DDPM-based diffusion generator for 1D cavitation signals."""
    def __init__(self, config: dict):
        self.sample_rate = config['sample_rate']
        self.duration = config['duration_seconds']
        self.signal_len = int(self.sample_rate * self.duration)
        self.batch_size = config.get('batch_size', 16)
        self.epochs = config.get('epochs', 50)
        self.lr = config.get('lr', 1e-4)
        self.timesteps = config.get('timesteps', 1000)
        self.beta_start = config.get('beta_start', 1e-4)
        self.beta_end = config.get('beta_end', 0.02)
        self.device = get_device()
        self.model = SimpleUNet1D(
            channels=config.get('unet_channels', 64),
            time_emb_dim=config.get('time_emb_dim', 128)
        ).to(self.device)
        if hasattr(torch, 'compile') and self.device.type != 'mps':
            self.model = torch.compile(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # Precompute betas and alphas
        betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        alphas = 1 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        # Register as buffers
        self.betas = betas.to(self.device)
        self.alphas_cum = alphas_cum.to(self.device)

    def train(self, dataset, **kwargs):
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=kwargs.get('num_workers', 4),
            pin_memory=kwargs.get('pin_memory', False)
        )
        self.model.train()
        for _ in range(self.epochs):
            for x, _ in loader:
                x = x.unsqueeze(1).to(self.device)
                t = torch.randint(0, self.timesteps, (x.size(0),), device=self.device)
                noise = torch.randn_like(x)
                sqrt_alpha_cum = self.alphas_cum[t]**0.5
                sqrt_one_minus = (1 - self.alphas_cum[t])**0.5
                noisy = sqrt_alpha_cum[:, None, None] * x + sqrt_one_minus[:, None, None] * noise
                pred = self.model(noisy, t)
                loss = nn.MSELoss()(pred, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return None

    def generate(self, state: int):
        x = torch.randn(1, 1, self.signal_len, device=self.device)
        for t in reversed(range(self.timesteps)):
            t_batch = torch.tensor([t], device=self.device)
            pred_noise = self.model(x, t_batch)
            beta = self.betas[t]
            alpha = 1 - beta
            alpha_cum = self.alphas_cum[t]
            x = (1 / alpha**0.5) * (x - ((1 - alpha) / (1 - alpha_cum)**0.5) * pred_noise)
            if t > 0:
                x += beta**0.5 * torch.randn_like(x)
        signal = x.squeeze().detach().cpu().numpy()
        signal = signal / np.max(np.abs(signal))
        return TimeSeries(data=signal.astype(np.float32), sample_rate=self.sample_rate), []
