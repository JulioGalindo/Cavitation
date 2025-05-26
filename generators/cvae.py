import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from generators.base import GenerativeModel
from utils.timeseries import TimeSeries

# Global optimizations for Apple MPS / CUDA / CPU
# Set float32 matrix multiplication precision high for MPS
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

class Encoder(nn.Module):
    def __init__(self, input_dim, label_dim, hidden_dim, latent_dim):
        super().__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim + label_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, labels):
        c = self.label_emb(labels)
        h = torch.cat([x, c], dim=1)
        h = self.net(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, label_dim, hidden_dim, output_dim):
        super().__init__()
        self.label_emb = nn.Embedding(label_dim, label_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + label_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh(),  # normalized output
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        h = torch.cat([z, c], dim=1)
        return self.net(h)

class CVAEGenerator(GenerativeModel):
    """Conditional Variational Autoencoder for synthetic cavitation signals."""
    def __init__(self, config: dict):
        # Hyperparameters
        self.sample_rate = config['sample_rate']
        self.duration = config['duration_seconds']
        self.input_dim = int(self.sample_rate * self.duration)
        self.latent_dim = config.get('latent_dim', 16)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 32)
        self.lr = config.get('lr', 1e-3)
        self.beta = config.get('beta', 1.0)
        self.label_dim = config.get('max_state', 6) + 1
        self.num_workers = config.get('num_workers', min(4, os.cpu_count()))
        self.pin_memory = config.get('pin_memory', True)

        # Device
        self.device = get_device()
        # Networks
        self.encoder = Encoder(self.input_dim, self.label_dim, self.hidden_dim, self.latent_dim).to(self.device)
        self.decoder = Decoder(self.latent_dim, self.label_dim, self.hidden_dim, self.input_dim).to(self.device)

        # JIT compile if available
        if hasattr(torch, 'compile'):
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr
        )
        # Loss
        self.recon_loss_fn = nn.MSELoss(reduction='mean')

    def train(self, dataset, **kwargs):
        '''Train CVAE on given dataset (expects (signal, label) pairs).'''
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        self.encoder.train()
        self.decoder.train()
        for epoch in range(self.epochs):
            for x, labels in dataloader:
                x = x.to(self.device, non_blocking=True).float()
                labels = labels.to(self.device, non_blocking=True)
                # Encode
                with torch.amp.autocast(device_type=self.device.type, dtype=torch.float32):
                    mu, logvar = self.encoder(x, labels)
                    std = (0.5 * logvar).exp()
                    eps = torch.randn_like(std)
                    z = mu + eps * std
                    recon = self.decoder(z, labels)
                    # Losses
                    recon_loss = self.recon_loss_fn(recon, x)
                    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + self.beta * kl
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        return None

    def generate(self, state: int):
        '''Generate a synthetic signal for given cavitation state.'''
        self.encoder.eval()
        self.decoder.eval()
        # Sample latent vector
        z = torch.randn(1, self.latent_dim, device=self.device)
        label = torch.tensor([state], device=self.device)
        with torch.no_grad():
            gen = self.decoder(z, label)
        signal = gen.squeeze(0).cpu().numpy()
        # Normalize to [-1,1]
        maxv = np.max(np.abs(signal))
        if maxv > 0:
            signal = signal / maxv
        return TimeSeries(data=signal.astype(np.float32), sample_rate=self.sample_rate), []
