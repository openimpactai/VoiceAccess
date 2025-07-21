"""
Configuration management for VoiceAccess
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration class for VoiceAccess"""
    
    # Model configuration
    model_type: str = "wav2vec2"
    model_name: str = "facebook/wav2vec2-base"
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    vocab_size: int = 32
    
    # Audio processing
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    max_audio_length: int = 30  # seconds
    
    # Training configuration
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 30
    warmup_steps: int = 500
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = True
    
    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Data augmentation
    augmentation_enabled: bool = True
    noise_prob: float = 0.3
    speed_perturb_range: tuple = (0.9, 1.1)
    pitch_shift_range: tuple = (-2, 2)
    
    # Language adaptation
    adapter_hidden_size: int = 256
    adapter_dropout: float = 0.1
    freeze_base_model: bool = True
    
    # Paths
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    log_dir: Path = Path("logs")
    cache_dir: Path = Path(".cache")
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    
    # API configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    max_concurrent_requests: int = 10
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "Config":
        """
        Load configuration from JSON or YAML file
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Config object
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            if config_path.suffix == '.json':
                config_dict = json.load(f)
            elif config_path.suffix in ['.yaml', '.yml']:
                config_dict = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
                
        return cls(**config_dict)
        
    def to_file(self, config_path: Union[str, Path]) -> None:
        """
        Save configuration to JSON or YAML file
        
        Args:
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_dict = self.__dict__.copy()
        
        # Convert Path objects to strings
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
                
        with open(config_path, 'w') as f:
            if config_path.suffix == '.json':
                json.dump(config_dict, f, indent=2)
            elif config_path.suffix in ['.yaml', '.yml']:
                yaml.dump(config_dict, f, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
                
    def update(self, **kwargs) -> None:
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Config has no attribute '{key}'")
                
    def validate(self) -> None:
        """Validate configuration parameters"""
        # Check sample rate
        if self.sample_rate not in [8000, 16000, 22050, 44100, 48000]:
            raise ValueError(f"Invalid sample rate: {self.sample_rate}")
            
        # Check batch size
        if self.batch_size < 1:
            raise ValueError(f"Batch size must be positive: {self.batch_size}")
            
        # Check learning rate
        if self.learning_rate <= 0:
            raise ValueError(f"Learning rate must be positive: {self.learning_rate}")
            
        # Create directories if they don't exist
        for dir_attr in ['data_dir', 'model_dir', 'log_dir', 'cache_dir']:
            dir_path = getattr(self, dir_attr)
            if isinstance(dir_path, str):
                dir_path = Path(dir_path)
                setattr(self, dir_attr, dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)