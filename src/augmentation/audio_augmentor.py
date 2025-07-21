"""
Audio augmentation techniques for low-resource language ASR
"""

import numpy as np
import torch
import torchaudio
from typing import Union, List, Tuple, Optional
import random
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class AudioAugmentor:
    """Audio augmentation for improving ASR robustness"""
    
    def __init__(self, config):
        """
        Initialize audio augmentor
        
        Args:
            config: Configuration object with augmentation parameters
        """
        self.config = config
        self.sample_rate = config.sample_rate
        self.augmentation_enabled = config.augmentation_enabled
        
        # Augmentation probabilities
        self.noise_prob = config.noise_prob
        self.speed_perturb_range = config.speed_perturb_range
        self.pitch_shift_range = config.pitch_shift_range
        
        # Initialize augmentation transforms
        self._init_transforms()
        
    def _init_transforms(self):
        """Initialize augmentation transforms"""
        # Speed perturbation
        self.speed_perturb = torchaudio.transforms.SpeedPerturbation(
            self.sample_rate,
            factors=list(np.arange(
                self.speed_perturb_range[0],
                self.speed_perturb_range[1] + 0.1,
                0.1
            ))
        )
        
        # Resampling for pitch shift
        self.resample_transform = {}
        
    def augment(
        self, 
        waveform: Union[np.ndarray, torch.Tensor],
        augmentations: Optional[List[str]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply augmentations to audio waveform
        
        Args:
            waveform: Audio waveform
            augmentations: List of augmentations to apply (if None, apply all with probability)
            
        Returns:
            Augmented waveform
        """
        if not self.augmentation_enabled:
            return waveform
            
        # Convert to tensor if needed
        is_numpy = isinstance(waveform, np.ndarray)
        if is_numpy:
            waveform = torch.from_numpy(waveform).float()
            
        # Ensure 2D tensor [channels, samples]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # Apply augmentations
        if augmentations is None:
            # Random augmentations based on probability
            if random.random() < self.noise_prob:
                waveform = self.add_noise(waveform)
            if random.random() < 0.3:
                waveform = self.speed_perturbation(waveform)
            if random.random() < 0.3:
                waveform = self.pitch_shift(waveform)
            if random.random() < 0.2:
                waveform = self.time_masking(waveform)
            if random.random() < 0.2:
                waveform = self.frequency_masking(waveform)
        else:
            # Apply specified augmentations
            for aug in augmentations:
                if aug == "noise":
                    waveform = self.add_noise(waveform)
                elif aug == "speed":
                    waveform = self.speed_perturbation(waveform)
                elif aug == "pitch":
                    waveform = self.pitch_shift(waveform)
                elif aug == "time_mask":
                    waveform = self.time_masking(waveform)
                elif aug == "freq_mask":
                    waveform = self.frequency_masking(waveform)
                elif aug == "reverb":
                    waveform = self.add_reverb(waveform)
                    
        # Convert back to numpy if needed
        if is_numpy:
            waveform = waveform.numpy()
            
        return waveform.squeeze()
        
    def add_noise(
        self, 
        waveform: torch.Tensor,
        snr_db: Optional[float] = None
    ) -> torch.Tensor:
        """
        Add white noise to waveform
        
        Args:
            waveform: Input waveform [channels, samples]
            snr_db: Signal-to-noise ratio in dB (if None, random between 10-40)
            
        Returns:
            Noisy waveform
        """
        if snr_db is None:
            snr_db = random.uniform(10, 40)
            
        # Generate white noise
        noise = torch.randn_like(waveform)
        
        # Calculate signal and noise power
        signal_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()
        
        # Calculate scaling factor
        snr_linear = 10 ** (snr_db / 10)
        noise_scaling = torch.sqrt(signal_power / (noise_power * snr_linear))
        
        # Add scaled noise
        noisy_waveform = waveform + noise_scaling * noise
        
        # Normalize to prevent clipping
        max_val = noisy_waveform.abs().max()
        if max_val > 1.0:
            noisy_waveform = noisy_waveform / max_val
            
        return noisy_waveform
        
    def speed_perturbation(
        self, 
        waveform: torch.Tensor,
        speed_factor: Optional[float] = None
    ) -> torch.Tensor:
        """
        Apply speed perturbation (tempo change)
        
        Args:
            waveform: Input waveform [channels, samples]
            speed_factor: Speed change factor (if None, random from range)
            
        Returns:
            Speed-perturbed waveform
        """
        if speed_factor is None:
            speed_factor = random.uniform(
                self.speed_perturb_range[0],
                self.speed_perturb_range[1]
            )
            
        # Apply speed perturbation
        if hasattr(self, 'speed_perturb'):
            perturbed, _ = self.speed_perturb(waveform.unsqueeze(0), speed_factor)
            return perturbed.squeeze(0)
        else:
            # Fallback: simple resampling
            return self._resample_based_speed_change(waveform, speed_factor)
            
    def _resample_based_speed_change(
        self,
        waveform: torch.Tensor,
        speed_factor: float
    ) -> torch.Tensor:
        """
        Speed change using resampling (fallback method)
        
        Args:
            waveform: Input waveform
            speed_factor: Speed change factor
            
        Returns:
            Speed-changed waveform
        """
        # Resample to change speed
        temp_sr = int(self.sample_rate * speed_factor)
        resampler = torchaudio.transforms.Resample(
            self.sample_rate,
            temp_sr
        )
        
        # Resample and then back
        temp_waveform = resampler(waveform)
        
        # Resample back to original sample rate
        resampler_back = torchaudio.transforms.Resample(
            temp_sr,
            self.sample_rate
        )
        
        return resampler_back(temp_waveform)
        
    def pitch_shift(
        self,
        waveform: torch.Tensor,
        n_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Apply pitch shifting
        
        Args:
            waveform: Input waveform [channels, samples]
            n_steps: Number of semitones to shift (if None, random from range)
            
        Returns:
            Pitch-shifted waveform
        """
        if n_steps is None:
            n_steps = random.randint(
                int(self.pitch_shift_range[0]),
                int(self.pitch_shift_range[1])
            )
            
        if n_steps == 0:
            return waveform
            
        # Use phase vocoder for pitch shifting
        # Convert to frequency domain
        n_fft = 2048
        hop_length = n_fft // 4
        
        # Compute STFT
        stft = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft),
            return_complex=True
        )
        
        # Shift pitch by stretching in frequency domain
        shift_factor = 2 ** (n_steps / 12)
        
        # Time stretch to compensate
        time_stretch_factor = 1 / shift_factor
        
        # Apply pitch shift (simplified version)
        # In practice, use librosa or specialized pitch shifting library
        shifted = self._phase_vocoder(stft, time_stretch_factor)
        
        # Convert back to time domain
        shifted_waveform = torch.istft(
            shifted,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft)
        )
        
        # Resample to original length
        if shifted_waveform.shape[-1] != waveform.shape[-1]:
            shifted_waveform = torch.nn.functional.interpolate(
                shifted_waveform.unsqueeze(0),
                size=waveform.shape[-1],
                mode='linear',
                align_corners=False
            ).squeeze(0)
            
        return shifted_waveform
        
    def _phase_vocoder(
        self,
        stft: torch.Tensor,
        stretch_factor: float
    ) -> torch.Tensor:
        """
        Simple phase vocoder for time stretching
        
        Args:
            stft: Complex STFT
            stretch_factor: Time stretch factor
            
        Returns:
            Time-stretched STFT
        """
        # Get magnitude and phase
        magnitude = stft.abs()
        phase = stft.angle()
        
        # Time stretch by interpolating magnitude
        stretched_len = int(magnitude.shape[-1] * stretch_factor)
        
        stretched_magnitude = torch.nn.functional.interpolate(
            magnitude.unsqueeze(0),
            size=(magnitude.shape[-2], stretched_len),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Interpolate phase with unwrapping
        stretched_phase = torch.nn.functional.interpolate(
            phase.unsqueeze(0),
            size=(phase.shape[-2], stretched_len),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        # Reconstruct complex STFT
        stretched_stft = stretched_magnitude * torch.exp(1j * stretched_phase)
        
        return stretched_stft
        
    def time_masking(
        self,
        waveform: torch.Tensor,
        mask_param: int = 100,
        num_masks: int = 1
    ) -> torch.Tensor:
        """
        Apply time masking augmentation
        
        Args:
            waveform: Input waveform [channels, samples]
            mask_param: Maximum mask length in samples
            num_masks: Number of masks to apply
            
        Returns:
            Masked waveform
        """
        masked_waveform = waveform.clone()
        length = waveform.shape[-1]
        
        for _ in range(num_masks):
            mask_length = random.randint(1, min(mask_param, length // 10))
            mask_start = random.randint(0, length - mask_length)
            
            # Apply mask (zero out)
            masked_waveform[..., mask_start:mask_start + mask_length] = 0
            
        return masked_waveform
        
    def frequency_masking(
        self,
        waveform: torch.Tensor,
        mask_param: int = 20,
        num_masks: int = 1
    ) -> torch.Tensor:
        """
        Apply frequency masking augmentation
        
        Args:
            waveform: Input waveform [channels, samples]
            mask_param: Maximum mask width in frequency bins
            num_masks: Number of masks to apply
            
        Returns:
            Frequency-masked waveform
        """
        # Convert to spectrogram
        n_fft = 1024
        hop_length = n_fft // 4
        
        stft = torch.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft),
            return_complex=True
        )
        
        # Apply frequency masks
        freq_bins = stft.shape[-2]
        
        for _ in range(num_masks):
            mask_width = random.randint(1, min(mask_param, freq_bins // 10))
            mask_start = random.randint(0, freq_bins - mask_width)
            
            # Zero out frequency bins
            stft[..., mask_start:mask_start + mask_width, :] = 0
            
        # Convert back to waveform
        masked_waveform = torch.istft(
            stft,
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.hann_window(n_fft)
        )
        
        # Match original length
        if masked_waveform.shape[-1] != waveform.shape[-1]:
            masked_waveform = torch.nn.functional.pad(
                masked_waveform,
                (0, waveform.shape[-1] - masked_waveform.shape[-1])
            )
            
        return masked_waveform
        
    def add_reverb(
        self,
        waveform: torch.Tensor,
        room_size: float = 0.5,
        damping: float = 0.5
    ) -> torch.Tensor:
        """
        Add reverb effect
        
        Args:
            waveform: Input waveform [channels, samples]
            room_size: Room size parameter (0-1)
            damping: Damping parameter (0-1)
            
        Returns:
            Reverberant waveform
        """
        # Simple reverb using comb filters
        # In practice, use torchaudio.sox_effects or specialized reverb
        
        delays = [1557, 1617, 1491, 1422]  # Sample delays
        gains = [0.7, 0.7, 0.7, 0.7]  # Feedback gains
        
        reverb_waveform = waveform.clone()
        
        for delay, gain in zip(delays, gains):
            # Adjust delay based on room size
            actual_delay = int(delay * room_size)
            
            # Apply comb filter
            delayed = torch.nn.functional.pad(waveform, (actual_delay, 0))[..., :-actual_delay]
            reverb_waveform = reverb_waveform + gain * (1 - damping) * delayed
            
        # Mix with original
        mix_ratio = 0.3
        output = (1 - mix_ratio) * waveform + mix_ratio * reverb_waveform
        
        # Normalize
        max_val = output.abs().max()
        if max_val > 1.0:
            output = output / max_val
            
        return output