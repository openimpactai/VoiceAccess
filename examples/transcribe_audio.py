#!/usr/bin/env python3
"""
Example: Basic audio transcription with VoiceAccess
"""

import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.core.asr_engine import ASREngine
from src.core.config import Config


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using VoiceAccess")
    parser.add_argument("audio_path", type=str, help="Path to audio file")
    parser.add_argument("--model", type=str, default="models/wav2vec2-base.pt", 
                       help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, default="wav2vec2",
                       choices=["wav2vec2", "whisper", "conformer"],
                       help="Type of model architecture")
    parser.add_argument("--language", type=str, default=None,
                       help="Language code for multilingual models")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    if Path(args.config).exists():
        config = Config.from_file(args.config)
    else:
        config = Config()
    
    # Initialize ASR engine
    print(f"Initializing ASR engine...")
    engine = ASREngine(config)
    
    # Load model
    print(f"Loading {args.model_type} model from {args.model}")
    try:
        engine.load_model(args.model, model_type=args.model_type)
    except FileNotFoundError:
        print(f"Model file not found: {args.model}")
        print("Please download a pre-trained model or train your own.")
        return
    
    # Transcribe audio
    print(f"Transcribing audio: {args.audio_path}")
    try:
        transcription = engine.transcribe(
            args.audio_path,
            language=args.language,
            return_confidence=True
        )
        
        if isinstance(transcription, tuple):
            text, confidence = transcription
            print(f"\nTranscription: {text}")
            print(f"Confidence: {confidence:.2%}")
        else:
            print(f"\nTranscription: {transcription}")
            
    except FileNotFoundError:
        print(f"Audio file not found: {args.audio_path}")
    except Exception as e:
        print(f"Error during transcription: {e}")


if __name__ == "__main__":
    main()