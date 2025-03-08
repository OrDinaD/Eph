import argparse
import torch
import torchaudio
from pathlib import Path
import numpy as np

from models.emotion_converter import EmotionConverter
from preprocessing.audio_processor import AudioProcessor
from configs.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description='Convert emotion in audio file')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input audio file')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to save converted audio')
    parser.add_argument('--emotion', type=str, required=True,
                      help='Target emotion (e.g., happy, sad, angry)')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to run inference on (cuda/cpu)')
    return parser.parse_args()

def load_model(checkpoint_path: str, config, device: str) -> EmotionConverter:
    """Load trained model from checkpoint"""
    model = EmotionConverter(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def convert_emotion(
    model: EmotionConverter,
    audio_processor: AudioProcessor,
    input_path: str,
    target_emotion: int,
    device: str
) -> torch.Tensor:
    """Convert emotion in audio file"""
    # Process input audio
    features = audio_processor.process_audio_file(input_path)
    mel_spec = features['mel_spectrogram'].unsqueeze(0).to(device)
    
    # Prepare target emotion
    target_emotion = torch.tensor([target_emotion], device=device)
    
    # Generate output mel-spectrogram
    with torch.no_grad():
        output_mel = model.inference(mel_spec, target_emotion)
    
    return output_mel.squeeze(0)

def main():
    # Parse arguments
    args = parse_args()
    
    # Load config
    config = Config()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize audio processor
    audio_processor = AudioProcessor(config)
    
    # Load model
    model = load_model(args.model, config, device)
    
    # Define emotion mapping
    emotion_to_idx = {
        'neutral': 0,
        'happy': 1,
        'sad': 2,
        'angry': 3,
        'fear': 4,
        'disgust': 5,
        'surprise': 6,
        'calm': 7
    }
    
    if args.emotion not in emotion_to_idx:
        raise ValueError(f'Unknown emotion: {args.emotion}. '
                       f'Available emotions: {list(emotion_to_idx.keys())}')
    
    # Convert emotion
    output_mel = convert_emotion(
        model,
        audio_processor,
        args.input,
        emotion_to_idx[args.emotion],
        device
    )
    
    # TODO: Implement vocoder for mel-spectrogram to waveform conversion
    # For now, we'll save the mel-spectrogram
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy array for now (in practice, you'd use a vocoder)
    np.save(output_path.with_suffix('.npy'), output_mel.cpu().numpy())
    print(f'Saved converted mel-spectrogram to {output_path.with_suffix(".npy")}')
    
    print('Note: This is a basic implementation. For production use:')
    print('1. Implement a vocoder to convert mel-spectrograms back to waveform')
    print('2. Add post-processing for better audio quality')
    print('3. Implement real-time processing for longer files')

if __name__ == '__main__':
    main() 