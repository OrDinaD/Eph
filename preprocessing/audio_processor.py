import os
import numpy as np
import torch
import torchaudio
import librosa
from typing import Tuple, Optional
from dataclasses import dataclass
import ffmpeg

class AudioProcessor:
    def __init__(self, config):
        self.config = config
        self.sample_rate = config.audio.sample_rate
        self.n_fft = config.audio.n_fft
        self.hop_length = config.audio.hop_length
        self.win_length = config.audio.win_length
        self.n_mels = config.audio.n_mels
        self.mel_fmin = config.audio.mel_fmin
        self.mel_fmax = config.audio.mel_fmax

    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file and convert to target sample rate"""
        if file_path.endswith('.ogg'):
            # Convert OGG to WAV using ffmpeg
            wav_path = file_path.replace('.ogg', '.wav')
            stream = ffmpeg.input(file_path)
            stream = ffmpeg.output(stream, wav_path, acodec='pcm_s16le', ar=self.sample_rate)
            ffmpeg.run(stream, overwrite_output=True)
            file_path = wav_path

        waveform, sr = torchaudio.load(file_path)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        
        return waveform, self.sample_rate

    def get_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel spectrogram"""
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.mel_fmin,
            f_max=self.mel_fmax
        )
        
        mel_spec = mel_transform(waveform)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        return mel_spec

    def extract_pitch(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract pitch (F0) from waveform"""
        waveform_np = waveform.numpy().squeeze()
        f0, voiced_flag, voiced_probs = librosa.pyin(
            waveform_np,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate
        )
        return torch.from_numpy(f0).float()

    def vad_split(self, waveform: torch.Tensor, threshold: float = 0.3) -> list:
        """Split audio based on voice activity detection"""
        # Calculate energy
        energy = torch.sqrt(torch.mean(waveform ** 2, dim=0))
        
        # Find segments above threshold
        is_speech = energy > threshold * torch.max(energy)
        
        # Get continuous segments
        segments = []
        start_idx = None
        
        for i in range(len(is_speech)):
            if is_speech[i] and start_idx is None:
                start_idx = i
            elif not is_speech[i] and start_idx is not None:
                segments.append((start_idx, i))
                start_idx = None
                
        if start_idx is not None:
            segments.append((start_idx, len(is_speech)))
            
        return segments

    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range"""
        return waveform / (torch.max(torch.abs(waveform)) + 1e-8)

    def process_audio_file(self, file_path: str) -> dict:
        """Process single audio file and extract all necessary features"""
        # Load and normalize audio
        waveform, sr = self.load_audio(file_path)
        waveform = self.normalize_audio(waveform)
        
        # Extract features
        mel_spec = self.get_mel_spectrogram(waveform)
        pitch = self.extract_pitch(waveform)
        segments = self.vad_split(waveform)
        
        return {
            'waveform': waveform,
            'mel_spectrogram': mel_spec,
            'pitch': pitch,
            'segments': segments,
            'sample_rate': sr
        } 