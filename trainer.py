import os
import signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Utility to report memory usage on MPS/CUDA
def current_memory_usage(device):
    if device.type == 'mps':
        return f"MPS:{torch.mps.current_allocated_memory() / 1e6:.1f}MB"
    elif device.type == 'cuda':
        return f"CUDA:{torch.cuda.memory_allocated() / 1e6:.1f}MB"
    else:
        return "CPU"

from visualizer import Visualizer


class Trainer:
    """
    Trainer class for cavitation detection models.

    - Supports Ctrl+C interruption with checkpointing.
    - Can resume from existing checkpoint.
    - Reports memory usage on MPS/CUDA.
    - Updates live plots of loss/accuracy via Visualizer.
    """

    def __init__(self, model, train_dataset, val_dataset, config, device):
        self.device = device
        self.model = model.to(device)
        self.config = config

        # Initialize DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training'].get('num_workers', 4),
            pin_memory=(device.type == 'cuda')
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training'].get('num_workers', 4),
            pin_memory=(device.type == 'cuda')
        )

        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['lr'])
        self.epochs = config['training']['epochs']
        self.checkpoint_path = config['training']['checkpoint_path']
        self.current_epoch = 0

        # Visualizer instance for live plotting
        self.visualizer = Visualizer()

        # Flag to catch Ctrl+C
        self.interrupted = False
        signal.signal(signal.SIGINT, self._on_interrupt)

        # Resume from checkpoint if present
        if os.path.exists(self.checkpoint_path):
            self._load_checkpoint()

    def _on_interrupt(self, sig, frame):
        print("\n[INFO] Interrupt received. Saving checkpoint...")
        self._save_checkpoint()
        self.interrupted = True

    def _save_checkpoint(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
        }, self.checkpoint_path)
        print(f"[INFO] Checkpoint saved at epoch {self.current_epoch + 1}")

    def _load_checkpoint(self):
        print(f"[INFO] Loading checkpoint from {self.checkpoint_path}")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.current_epoch = ckpt['epoch'] + 1
        print(f"[INFO] Resuming from epoch {self.current_epoch + 1}")

    def train(self):
        """
        Main training loop:
        - Loops over epochs
        - Tracks train & val loss/accuracy
        - Saves checkpoint each epoch
        - Stops cleanly on Ctrl+C
        """
        print(f"[INFO] Starting training on {self.device}")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch
            self.model.train()

            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for inputs, labels in progress:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward + backward + optimize
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Metrics
                batch_size = labels.size(0)
                epoch_loss += loss.item() * batch_size
                _, preds = outputs.max(1)
                epoch_correct += (preds == labels).sum().item()
                epoch_total += batch_size

                # Report progress
                acc = epoch_correct / epoch_total
                mem = current_memory_usage(self.device)
                progress.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{acc:.4f}",
                    'mem': mem
                })

                if self.interrupted:
                    break

            # End of epoch metrics
            train_loss = epoch_loss / epoch_total
            train_acc = epoch_correct / epoch_total
            print(f"[Train] Epoch {epoch+1} | Loss {train_loss:.4f} | Acc {train_acc:.4f}")
            self.visualizer.update(epoch, train_loss, None, train_acc, None)

            # Validation
            val_acc, val_loss = self.validate()
            self.visualizer.update(epoch, train_loss, val_loss, train_acc, val_acc)

            # Save checkpoint
            self._save_checkpoint()
            if self.interrupted:
                print("[INFO] Training interrupted. Exiting loop.")
                break

    def validate(self):
        """
        Runs one full pass over the validation set.
        Returns (accuracy, avg_loss).
        """
        self.model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                batch_size = labels.size(0)
                val_loss += loss.item() * batch_size
                _, preds = outputs.max(1)
                val_correct += (preds == labels).sum().item()
                val_total += batch_size

        avg_loss = val_loss / val_total if val_total else 0.0
        accuracy = val_correct / val_total if val_total else 0.0
        print(f"[Val] Loss {avg_loss:.4f} | Acc {accuracy:.4f}")
        return accuracy, avg_loss
