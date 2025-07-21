"""
Conformer model implementation for VoiceAccess
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math
import logging

from .base_model import BaseASRModel

logger = logging.getLogger(__name__)


class ConformerModel(BaseASRModel):
    """Conformer model for speech recognition"""
    
    def __init__(self, config):
        """
        Initialize Conformer model
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        
        # Model dimensions
        self.input_dim = config.n_mels
        self.encoder_dim = config.hidden_size
        self.num_layers = config.num_hidden_layers
        self.num_heads = config.num_attention_heads
        self.vocab_size = config.vocab_size
        
        # Build model components
        self._build_model()
        
    def _build_model(self):
        """Build Conformer model architecture"""
        # Input projection
        self.input_projection = nn.Linear(self.input_dim, self.encoder_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.encoder_dim)
        
        # Conformer encoder blocks
        self.encoder_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=self.encoder_dim,
                num_heads=self.num_heads,
                conv_kernel_size=31,
                dropout=0.1
            ) for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(self.encoder_dim, self.vocab_size)
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Conformer model
        
        Args:
            audio_features: Input features [batch_size, time, feature_dim]
            
        Returns:
            Logits [batch_size, time, vocab_size]
        """
        # Input projection
        x = self.input_projection(audio_features)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through Conformer blocks
        for block in self.encoder_blocks:
            x = block(x)
            
        # Output projection
        logits = self.output_projection(x)
        
        return logits
        
    def transcribe(
        self,
        audio_features: torch.Tensor,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio features to text
        
        Args:
            audio_features: Input audio features
            language: Language code (optional)
            
        Returns:
            Dictionary containing transcription and metadata
        """
        # Get logits
        with torch.no_grad():
            logits = self.forward(audio_features)
            
        # Greedy decoding
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode to text
        text = self._decode_predictions(predicted_ids[0])
        
        # Calculate confidence
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0].mean().item()
        
        return {
            "text": text,
            "confidence": confidence,
            "language": language or "unknown"
        }
        
    def _decode_predictions(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs to text
        
        Args:
            token_ids: Predicted token IDs
            
        Returns:
            Decoded text
        """
        # Simple character-based decoding
        chars = []
        prev_id = None
        
        for token_id in token_ids:
            token_id = token_id.item()
            
            # Skip blank token (0) and repeated tokens
            if token_id == 0 or token_id == prev_id:
                continue
                
            # Map to character
            if 1 <= token_id <= 26:  # a-z
                chars.append(chr(ord('a') + token_id - 1))
            elif token_id == 27:  # space
                chars.append(' ')
            elif token_id == 28:  # apostrophe
                chars.append("'")
                
            prev_id = token_id
            
        return ''.join(chars).strip()


class ConformerBlock(nn.Module):
    """Single Conformer encoder block"""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        conv_kernel_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Feed-forward module 1
        self.ff1 = FeedForwardModule(d_model, dropout=dropout)
        
        # Multi-head self-attention module
        self.mhsa = MultiHeadSelfAttentionModule(
            d_model,
            num_heads,
            dropout=dropout
        )
        
        # Convolution module
        self.conv = ConvolutionModule(
            d_model,
            kernel_size=conv_kernel_size,
            dropout=dropout
        )
        
        # Feed-forward module 2
        self.ff2 = FeedForwardModule(d_model, dropout=dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Conformer block
        
        Args:
            x: Input tensor [batch_size, time, d_model]
            
        Returns:
            Output tensor [batch_size, time, d_model]
        """
        # First feed-forward
        x = x + 0.5 * self.ff1(x)
        
        # Multi-head self-attention
        x = x + self.mhsa(x)
        
        # Convolution
        x = x + self.conv(x)
        
        # Second feed-forward
        x = x + 0.5 * self.ff2(x)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        return x


class FeedForwardModule(nn.Module):
    """Feed-forward module with Swish activation"""
    
    def __init__(self, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.swish = nn.SiLU()  # Swish activation
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class MultiHeadSelfAttentionModule(nn.Module):
    """Multi-head self-attention module with relative positional encoding"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.mhsa = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.layer_norm(x)
        attn_output, _ = self.mhsa(x_norm, x_norm, x_norm)
        return self.dropout(attn_output)


class ConvolutionModule(nn.Module):
    """Convolution module with gated linear units"""
    
    def __init__(self, d_model: int, kernel_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Pointwise convolution
        self.pointwise_conv1 = nn.Linear(d_model, 2 * d_model)
        
        # GLU activation
        self.glu = nn.GLU(dim=-1)
        
        # 1D depthwise convolution
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        
        # Pointwise convolution
        self.pointwise_conv2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through convolution module
        
        Args:
            x: Input tensor [batch_size, time, d_model]
            
        Returns:
            Output tensor [batch_size, time, d_model]
        """
        # Layer norm
        x = self.layer_norm(x)
        
        # First pointwise conv
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        
        # Depthwise convolution
        x = x.transpose(1, 2)  # [batch, d_model, time]
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = x.transpose(1, 2)  # [batch, time, d_model]
        
        # Second pointwise conv
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for Conformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input
        
        Args:
            x: Input tensor [batch_size, time, d_model]
            
        Returns:
            Output with positional encoding
        """
        return x + self.pe[:, :x.size(1), :]