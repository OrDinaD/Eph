import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ..preprocessing.audio_processor import AudioProcessor

class EmotionAudioDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        audio_dir: str,
        config,
        split: str = 'train',
        transform=None
    ):
        """
        Args:
            metadata_path: Path to CSV file with columns: [file_path, emotion, transcription]
            audio_dir: Directory containing audio files
            config: Configuration object
            split: One of ['train', 'val', 'test']
            transform: Optional transform to be applied on a sample
        """
        self.metadata = pd.read_csv(metadata_path)
        self.audio_dir = Path(audio_dir)
        self.config = config
        self.transform = transform
        self.audio_processor = AudioProcessor(config)
        
        # Create emotion to index mapping
        self.emotion_to_idx = {
            emotion: idx for idx, emotion in enumerate(
                self.metadata['emotion'].unique()
            )
        }
        
        # Split dataset
        self.split_dataset(split)

    def split_dataset(self, split: str):
        """Split dataset into train/val/test"""
        # Shuffle data
        self.metadata = self.metadata.sample(frac=1, random_state=42)
        
        # Calculate split indices
        total_size = len(self.metadata)
        train_size = int(total_size * self.config.data.train_ratio)
        val_size = int(total_size * self.config.data.val_ratio)
        
        if split == 'train':
            self.metadata = self.metadata[:train_size]
        elif split == 'val':
            self.metadata = self.metadata[train_size:train_size + val_size]
        else:  # test
            self.metadata = self.metadata[train_size + val_size:]

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get metadata
        row = self.metadata.iloc[idx]
        audio_path = self.audio_dir / row['file_path']
        emotion = row['emotion']
        
        # Process audio
        audio_features = self.audio_processor.process_audio_file(str(audio_path))
        
        # Convert emotion to index
        emotion_idx = self.emotion_to_idx[emotion]
        
        # Generate random target emotion (different from source)
        available_emotions = list(self.emotion_to_idx.values())
        available_emotions.remove(emotion_idx)
        target_emotion_idx = np.random.choice(available_emotions)
        
        sample = {
            'mel_spectrogram': audio_features['mel_spectrogram'],
            'source_emotion': torch.tensor(emotion_idx, dtype=torch.long),
            'target_emotion': torch.tensor(target_emotion_idx, dtype=torch.long),
            'pitch': audio_features['pitch'],
            'transcription': row['transcription']
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class AudioCollate:
    """Zero-pads audio sequences to the longest sequence in the batch"""
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Get sequence lengths
        lengths = [b['mel_spectrogram'].shape[-1] for b in batch]
        max_length = max(lengths)
        
        # Initialize padded tensors
        batch_size = len(batch)
        mel_dim = batch[0]['mel_spectrogram'].shape[1]
        padded_mels = torch.zeros(batch_size, mel_dim, max_length)
        padded_pitches = torch.zeros(batch_size, max_length)
        
        # Collect other tensors
        emotions = torch.stack([b['source_emotion'] for b in batch])
        target_emotions = torch.stack([b['target_emotion'] for b in batch])
        transcriptions = [b['transcription'] for b in batch]
        
        # Pad sequences
        for i, b in enumerate(batch):
            mel_length = b['mel_spectrogram'].shape[-1]
            padded_mels[i, :, :mel_length] = b['mel_spectrogram']
            padded_pitches[i, :mel_length] = b['pitch']
        
        return {
            'mel_spectrogram': padded_mels,
            'pitch': padded_pitches,
            'source_emotion': emotions,
            'target_emotion': target_emotions,
            'transcription': transcriptions,
            'lengths': torch.tensor(lengths)
        }

def create_dataloader(
    metadata_path: str,
    audio_dir: str,
    config,
    split: str = 'train',
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """Create a DataLoader for the emotion audio dataset"""
    dataset = EmotionAudioDataset(
        metadata_path=metadata_path,
        audio_dir=audio_dir,
        config=config,
        split=split
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=AudioCollate(),
        pin_memory=True
    ) 