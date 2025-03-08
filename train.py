import argparse
from pathlib import Path
import torch
import yaml

from models.emotion_converter import EmotionConverter
from training.trainer import EmotionConverterTrainer
from data.dataset import create_dataloader
from configs.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Train emotion conversion model')
    parser.add_argument('--config', type=str, required=True,
                      help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='Path to checkpoint to resume training from')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing audio files')
    parser.add_argument('--metadata', type=str, required=True,
                      help='Path to metadata CSV file')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = Config()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create dataloaders
    train_loader = create_dataloader(
        metadata_path=args.metadata,
        audio_dir=args.data_dir,
        config=config,
        split='train',
        batch_size=config.training.batch_size
    )
    
    val_loader = create_dataloader(
        metadata_path=args.metadata,
        audio_dir=args.data_dir,
        config=config,
        split='val',
        batch_size=config.training.batch_size,
        shuffle=False
    )
    
    # Create model
    model = EmotionConverter(config)
    
    # Create trainer
    trainer = EmotionConverterTrainer(
        model=model,
        config=config,
        device=device
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f'Loading checkpoint from {args.checkpoint}')
        start_epoch, _ = trainer.load_checkpoint(args.checkpoint)
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs
    )

if __name__ == '__main__':
    main() 