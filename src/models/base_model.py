"""
Base model class for ASR models
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseASRModel(nn.Module, ABC):
    """Abstract base class for ASR models"""
    
    def __init__(self, config):
        """
        Initialize base ASR model
        
        Args:
            config: Configuration object
        """
        super().__init__()
        self.config = config
        
    @abstractmethod
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            audio_features: Input audio features [batch_size, seq_len, feature_dim]
            
        Returns:
            Model outputs
        """
        pass
        
    @abstractmethod
    def transcribe(
        self, 
        audio_features: torch.Tensor,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio features to text
        
        Args:
            audio_features: Input audio features
            language: Language code for multilingual models
            
        Returns:
            Dictionary containing transcription and metadata
        """
        pass
        
    def transcribe_batch(
        self,
        audio_features: torch.Tensor,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Transcribe batch of audio features
        
        Args:
            audio_features: Batch of audio features [batch_size, seq_len, feature_dim]
            language: Language code for multilingual models
            
        Returns:
            List of transcription dictionaries
        """
        results = []
        for i in range(audio_features.size(0)):
            result = self.transcribe(
                audio_features[i].unsqueeze(0),
                language=language
            )
            results.append(result)
        return results
        
    def save_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Save model checkpoint
        
        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__,
            'model_type': self.__class__.__name__
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Model checkpoint saved to {checkpoint_path}")
        
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to load checkpoint from
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        self.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Model checkpoint loaded from {checkpoint_path}")
        
    def get_num_params(self) -> int:
        """Get total number of model parameters"""
        return sum(p.numel() for p in self.parameters())
        
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
    def freeze_base_model(self) -> None:
        """Freeze base model parameters for transfer learning"""
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze_base_model(self) -> None:
        """Unfreeze all model parameters"""
        for param in self.parameters():
            param.requires_grad = True
            
    def get_model_size_mb(self) -> float:
        """Get model size in megabytes"""
        param_size = 0
        for param in self.parameters():
            param_size += param.numel() * param.element_size()
            
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.numel() * buffer.element_size()
            
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb