import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from contextlib import nullcontext
from generators.base import GenerativeModel
from utils.timeseries import TimeSeries

# Global optimizations for Apple MPS / CUDA / CPU
# Set float32 matrix multiplication precision high for MPS if available
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision('high')
# Use all CPU cores
torch.set_num_threads(os.cpu_count())

def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class GeneratorNet(nn.Module):
    def __init__(self, latent_dim, label_dim, hidden_dim, output_dim):
        super().__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + label_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], dim=1)
        return self.model(x)

class DiscriminatorNet(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim):
        super().__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.feature = nn.Sequential(
            nn.Linear(input_dim + label_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.adv_head = nn.Linear(hidden_dim, 1)
        self.aux_head = nn.Linear(hidden_dim, label_dim)

    def forward(self, x, labels):
        c = self.label_emb(labels)
        d_in = torch.cat([x, c], dim=1)
        features = self.feature(d_in)
        return self.adv_head(features), self.aux_head(features)

class CGANGenerator(GenerativeModel):
    """Conditional GAN generator for synthetic cavitation signals optimized for MPS/CUDA/CPU."""
    def __init__(self, config: dict):
        # Settings
        self.sample_rate = config['sample_rate']
        self.duration = config['duration_seconds']
        self.num_samples = int(self.sample_rate * self.duration)
        self.latent_dim = config.get('latent_dim', 100)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        self.lr = config.get('lr', 1e-4)
        self.label_dim = config.get('max_state', 6) + 1
        self.num_workers = config.get('num_workers', min(4, os.cpu_count()))
        self.pin_memory = config.get('pin_memory', True)

        # Device
        self.device = get_device()
        self.use_amp = (self.device.type == 'cuda')

        # Networks
        self.generator = GeneratorNet(self.latent_dim, self.label_dim,
                                      self.hidden_dim, self.num_samples).to(self.device)
        self.discriminator = DiscriminatorNet(self.num_samples, self.label_dim,
                                              self.hidden_dim).to(self.device)

        # JIT compile only on CUDA/CPU
        if hasattr(torch, 'compile') and self.device.type != 'mps':
            self.generator = torch.compile(self.generator)
            self.discriminator = torch.compile(self.discriminator)

        # Optimizers and losses
        self.opt_g = optim.Adam(self.generator.parameters(), lr=self.lr)
        self.opt_d = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.auxiliary_loss = nn.CrossEntropyLoss()

    def train(self, dataset, **kwargs):
        '''Train CGAN with given dataset (expects (signal, label) pairs).'''
        loader_pin = self.pin_memory and self.device.type == 'cuda'
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=loader_pin
        )
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

        for epoch in range(self.epochs):
            for real, labels in dataloader:
                real = real.to(self.device, non_blocking=True).float()
                labels = labels.to(self.device, non_blocking=True)
                bs = real.size(0)
                valid = torch.ones(bs, 1, device=self.device)
                fake = torch.zeros(bs, 1, device=self.device)

                # Discriminator
                self.opt_d.zero_grad()
                ctx = torch.amp.autocast(device_type='cuda', enabled=self.use_amp)
                with ctx:
                    rv, ra = self.discriminator(real, labels)
                    d_real = self.adversarial_loss(rv, valid) + self.auxiliary_loss(ra, labels)
                    z = torch.randn(bs, self.latent_dim, device=self.device)
                    gl = torch.randint(0, self.label_dim, (bs,), device=self.device)
                    gs = self.generator(z, gl)
                    fv, fa = self.discriminator(gs.detach(), gl)
                    d_fake = self.adversarial_loss(fv, fake) + self.auxiliary_loss(fa, gl)
                    d_loss = 0.5 * (d_real + d_fake)
                if self.use_amp:
                    scaler.scale(d_loss).backward()
                    scaler.step(self.opt_d)
                    scaler.update()
                else:
                    d_loss.backward()
                    self.opt_d.step()

                # Generator
                self.opt_g.zero_grad()
                with ctx:
                    gs = self.generator(z, gl)
                    gv, ga = self.discriminator(gs, gl)
                    g_loss = self.adversarial_loss(gv, valid) + self.auxiliary_loss(ga, gl)
                if self.use_amp:
                    scaler.scale(g_loss).backward()
                    scaler.step(self.opt_g)
                    scaler.update()
                else:
                    g_loss.backward()
                    self.opt_g.step()
        return None

    def generate(self, state: int):
        '''Generate a synthetic signal for given cavitation state.'''
        z = torch.randn(1, self.latent_dim, device=self.device)
        label = torch.tensor([state], device=self.device)
        self.generator.eval()
        with torch.no_grad():
            gen = self.generator(z, label)
        sig = gen.squeeze(0).cpu().numpy()
        mv = np.max(np.abs(sig))
        if mv > 0:
            sig = sig / mv
        return TimeSeries(data=sig.astype(np.float32), sample_rate=self.sample_rate), []
