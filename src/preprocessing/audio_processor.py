"""
Audio preprocessing utilities for VoiceAccess
"""

import numpy as np
import librosa
import torch
import torchaudio
from typing import Union, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handle audio loading and preprocessing"""
    
    def __init__(self, config):
        """
        Initialize audio processor
        
        Args:
            config: Configuration object with audio parameters
        """
        self.config = config
        self.sample_rate = config.sample_rate
        self.n_mels = config.n_mels
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.win_length = config.win_length
        self.max_length = config.max_audio_length * config.sample_rate
        
    def load_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Load audio file and resample to target sample rate
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Audio waveform as numpy array
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        try:
            # Load audio with librosa
            waveform, sr = librosa.load(
                str(audio_path), 
                sr=self.sample_rate,
                mono=True
            )
            
            # Normalize audio
            if np.abs(waveform).max() > 0:
                waveform = waveform / np.abs(waveform).max()
                
            return waveform
            
        except Exception as e:
            logger.error(f"Error loading audio file {audio_path}: {e}")
            raise
            
    def process(self, audio_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Process audio input to features
        
        Args:
            audio_input: Audio file path or waveform array
            
        Returns:
            Processed audio features
        """
        # Load audio if path is provided
        if isinstance(audio_input, (str, Path)):
            waveform = self.load_audio(audio_input)
        else:
            waveform = audio_input
            
        # Ensure correct length
        waveform = self._pad_or_trim(waveform)
        
        # Extract features based on model type
        if self.config.model_type in ["wav2vec2", "whisper"]:
            # Return raw waveform for these models
            features = waveform
        else:
            # Extract mel spectrogram features
            features = self._extract_mel_features(waveform)
            
        return features
        
    def _pad_or_trim(self, waveform: np.ndarray) -> np.ndarray:
        """
        Pad or trim waveform to maximum length
        
        Args:
            waveform: Input audio waveform
            
        Returns:
            Padded or trimmed waveform
        """
        if len(waveform) > self.max_length:
            # Trim to max length
            waveform = waveform[:self.max_length]
        elif len(waveform) < self.max_length:
            # Pad with zeros
            padding = self.max_length - len(waveform)
            waveform = np.pad(waveform, (0, padding), mode='constant')
            
        return waveform
        
    def _extract_mel_features(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram features
        
        Args:
            waveform: Input audio waveform
            
        Returns:
            Mel spectrogram features
        """
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Transpose to [time, freq]
        log_mel_spec = log_mel_spec.T
        
        # Normalize
        mean = np.mean(log_mel_spec, axis=0)
        std = np.std(log_mel_spec, axis=0)
        log_mel_spec = (log_mel_spec - mean) / (std + 1e-8)
        
        return log_mel_spec
        
    def compute_mfcc(self, waveform: np.ndarray, n_mfcc: int = 13) -> np.ndarray:
        """
        Compute MFCC features
        
        Args:
            waveform: Input audio waveform
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Add delta and delta-delta features
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Stack features
        features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
        
        # Transpose to [time, features]
        features = features.T
        
        return features
        
    def apply_cmvn(self, features: np.ndarray) -> np.ndarray:
        """
        Apply Cepstral Mean and Variance Normalization
        
        Args:
            features: Input features
            
        Returns:
            Normalized features
        """
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        normalized = (features - mean) / (std + 1e-8)
        
        return normalized