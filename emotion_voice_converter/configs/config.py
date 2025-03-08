from dataclasses import dataclass
from typing import List, Optional, Tuple

@dataclass
class AudioConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: Optional[float] = 8000.0

@dataclass
class ModelConfig:
    # Encoder parameters
    encoder_channels: List[int] = (32, 64, 128, 256, 512)
    encoder_kernel_sizes: List[int] = (3, 3, 3, 3, 3)
    encoder_strides: List[int] = (2, 2, 2, 2, 2)
    
    # Emotion embedding
    emotion_embedding_dim: int = 256
    num_emotions: int = 8  # happy, sad, angry, neutral, etc.
    
    # Decoder parameters
    decoder_channels: List[int] = (512, 256, 128, 64, 32)
    decoder_kernel_sizes: List[int] = (3, 3, 3, 3, 3)
    decoder_strides: List[int] = (2, 2, 2, 2, 2)
    
    # Skip connections
    use_skip_connections: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Loss weights
    reconstruction_weight: float = 10.0
    adversarial_weight: float = 1.0
    emotion_weight: float = 5.0
    cycle_weight: float = 5.0
    
    # Optimizer
    optimizer: str = "adam"
    scheduler: str = "cosine"
    
    # Checkpointing
    checkpoint_interval: int = 1000
    eval_interval: int = 100

@dataclass
class DataConfig:
    # Data paths
    raw_audio_dir: str = "data/raw"
    processed_audio_dir: str = "data/processed"
    metadata_path: str = "data/metadata.csv"
    
    # Data processing
    max_wav_length: int = 8192
    min_wav_length: int = 1024
    vad_threshold: float = 0.3
    
    # Train/val/test split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

@dataclass
class Config:
    audio: AudioConfig = AudioConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    
    # Project paths
    project_name: str = "emotion_voice_converter"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # Device
    device: str = "cuda"  # or "cpu"
    
    def __post_init__(self):
        """Validate configuration parameters"""
        assert sum([self.data.train_ratio, 
                   self.data.val_ratio, 
                   self.data.test_ratio]) == 1.0 