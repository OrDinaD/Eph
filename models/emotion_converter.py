import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=(kernel_size - 1) // 2
        )
        self.norm = nn.InstanceNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))

class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=(kernel_size - 1) // 2
        )
        self.norm = nn.InstanceNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.norm(self.conv_transpose(x)))

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        
        channels = [config.model.encoder_channels[0]] + list(config.model.encoder_channels)
        kernel_sizes = config.model.encoder_kernel_sizes
        strides = config.model.encoder_strides
        
        for i in range(len(channels) - 1):
            self.layers.append(
                ConvBlock(
                    channels[i], channels[i + 1],
                    kernel_sizes[i], strides[i]
                )
            )

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return x, features

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_skip_connections = config.model.use_skip_connections
        self.layers = nn.ModuleList()
        
        channels = list(config.model.decoder_channels)
        kernel_sizes = config.model.decoder_kernel_sizes
        strides = config.model.decoder_strides
        
        for i in range(len(channels) - 1):
            in_channels = channels[i]
            if self.use_skip_connections:
                in_channels *= 2  # Double for skip connection
            
            self.layers.append(
                ConvTransposeBlock(
                    in_channels, channels[i + 1],
                    kernel_sizes[i], strides[i]
                )
            )

    def forward(self, x, encoder_features=None):
        for i, layer in enumerate(self.layers):
            if self.use_skip_connections and encoder_features is not None:
                skip_feature = encoder_features[-(i + 1)]
                x = torch.cat([x, skip_feature], dim=1)
            x = layer(x)
        return x

class EmotionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(
            config.model.num_emotions,
            config.model.emotion_embedding_dim
        )
        
    def forward(self, emotion_id):
        return self.embedding(emotion_id)

class EmotionConverter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Main components
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.emotion_encoder = EmotionEncoder(config)
        
        # Projection layers
        self.content_projection = nn.Linear(
            config.model.encoder_channels[-1],
            config.model.emotion_embedding_dim
        )
        
        self.emotion_projection = nn.Linear(
            config.model.emotion_embedding_dim,
            config.model.encoder_channels[-1]
        )

    def forward(self, mel_spec: torch.Tensor, source_emotion: torch.Tensor, target_emotion: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # Encode input mel-spectrogram
        content_features, skip_features = self.encoder(mel_spec)
        
        # Project content features
        content_embedding = self.content_projection(content_features.transpose(1, 2))
        
        # Get emotion embeddings
        target_emotion_embedding = self.emotion_encoder(target_emotion)
        
        # Combine content and emotion
        combined_features = content_embedding + target_emotion_embedding
        
        # Project back to decoder dimension
        decoder_input = self.emotion_projection(combined_features).transpose(1, 2)
        
        # Decode
        output_mel = self.decoder(decoder_input, skip_features)
        
        return output_mel, {
            'content_embedding': content_embedding,
            'target_emotion_embedding': target_emotion_embedding,
            'combined_features': combined_features
        }

    def inference(self, mel_spec: torch.Tensor, target_emotion: torch.Tensor) -> torch.Tensor:
        """Simplified forward pass for inference"""
        with torch.no_grad():
            content_features, skip_features = self.encoder(mel_spec)
            content_embedding = self.content_projection(content_features.transpose(1, 2))
            target_emotion_embedding = self.emotion_encoder(target_emotion)
            combined_features = content_embedding + target_emotion_embedding
            decoder_input = self.emotion_projection(combined_features).transpose(1, 2)
            output_mel = self.decoder(decoder_input, skip_features)
        return output_mel 