import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
import wandb

class EmotionConverterTrainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(device)
        
        # Optimizer
        self.optimizer = Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2)
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_epochs
        )
        
        # Loss weights
        self.reconstruction_weight = config.training.reconstruction_weight
        self.adversarial_weight = config.training.adversarial_weight
        self.emotion_weight = config.training.emotion_weight
        self.cycle_weight = config.training.cycle_weight
        
        # Initialize best loss for model saving
        self.best_loss = float('inf')
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def reconstruction_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 loss between output and target mel-spectrograms"""
        return F.l1_loss(output, target)

    def emotion_consistency_loss(self, emotion_logits: torch.Tensor, target_emotion: torch.Tensor) -> torch.Tensor:
        """Cross entropy loss for emotion classification"""
        return F.cross_entropy(emotion_logits, target_emotion)

    def cycle_consistency_loss(self, original: torch.Tensor, cycled: torch.Tensor) -> torch.Tensor:
        """L1 loss between original and cycled mel-spectrograms"""
        return F.l1_loss(original, cycled)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Move batch to device
        mel_spec = batch['mel_spectrogram'].to(self.device)
        source_emotion = batch['source_emotion'].to(self.device)
        target_emotion = batch['target_emotion'].to(self.device)
        
        # Forward pass
        output_mel, features = self.model(mel_spec, source_emotion, target_emotion)
        
        # Calculate losses
        rec_loss = self.reconstruction_loss(output_mel, mel_spec)
        
        # Cycle consistency: convert back to source emotion
        cycled_mel, _ = self.model(output_mel, target_emotion, source_emotion)
        cycle_loss = self.cycle_consistency_loss(mel_spec, cycled_mel)
        
        # Total loss
        total_loss = (
            self.reconstruction_weight * rec_loss +
            self.cycle_weight * cycle_loss
        )
        
        # Optimization step
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'reconstruction_loss': rec_loss.item(),
            'cycle_loss': cycle_loss.item()
        }

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        total_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                mel_spec = batch['mel_spectrogram'].to(self.device)
                source_emotion = batch['source_emotion'].to(self.device)
                target_emotion = batch['target_emotion'].to(self.device)
                
                output_mel, _ = self.model(mel_spec, source_emotion, target_emotion)
                rec_loss = self.reconstruction_loss(output_mel, mel_spec)
                
                cycled_mel, _ = self.model(output_mel, target_emotion, source_emotion)
                cycle_loss = self.cycle_consistency_loss(mel_spec, cycled_mel)
                
                total_loss = (
                    self.reconstruction_weight * rec_loss +
                    self.cycle_weight * cycle_loss
                )
                
                total_losses.append(total_loss.item())
        
        return {'val_loss': np.mean(total_losses)}

    def save_checkpoint(self, epoch: int, loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss
        }
        
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']

    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """Full training loop"""
        # Initialize wandb
        wandb.init(
            project=self.config.project_name,
            config=self.config
        )
        
        for epoch in range(num_epochs):
            # Training
            train_losses = []
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch in pbar:
                losses = self.train_step(batch)
                train_losses.append(losses['total_loss'])
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{np.mean(train_losses):.4f}"
                })
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': np.mean(train_losses),
                **val_metrics,
                'learning_rate': self.scheduler.get_last_lr()[0]
            }
            wandb.log(metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1, val_metrics['val_loss'])
        
        wandb.finish() 