"""
Wav2Vec2 model implementation for VoiceAccess
"""

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import Dict, Any, Optional, List
import logging

from .base_model import BaseASRModel

logger = logging.getLogger(__name__)


class Wav2Vec2Model(BaseASRModel):
    """Wav2Vec2 model for speech recognition"""
    
    def __init__(self, config):
        """
        Initialize Wav2Vec2 model
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        
        self.model_name = config.model_name
        self.vocab_size = config.vocab_size
        
        # Load processor and model
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForCTC.from_pretrained(
                self.model_name,
                vocab_size=self.vocab_size,
                attention_dropout=0.1,
                hidden_dropout=0.1,
                feat_proj_dropout=0.0,
                layerdrop=0.1,
                ctc_loss_reduction="mean",
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )
        except Exception as e:
            logger.warning(f"Could not load pretrained model {self.model_name}: {e}")
            logger.info("Initializing model from scratch")
            self._init_from_scratch()
            
    def _init_from_scratch(self):
        """Initialize model from scratch when pretrained is not available"""
        from transformers import Wav2Vec2Config
        
        config = Wav2Vec2Config(
            vocab_size=self.vocab_size,
            hidden_size=self.config.hidden_size,
            num_hidden_layers=self.config.num_hidden_layers,
            num_attention_heads=self.config.num_attention_heads,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            feat_extract_norm="group",
            feat_proj_dropout=0.0,
            feat_extract_dropout=0.0,
            feat_extract_activation="gelu",
            conv_dim=[512, 512, 512, 512, 512, 512, 512],
            conv_stride=[5, 2, 2, 2, 2, 2, 2],
            conv_kernel=[10, 3, 3, 3, 3, 2, 2],
            conv_bias=False,
            num_conv_pos_embeddings=128,
            num_conv_pos_embedding_groups=16,
            do_stable_layer_norm=False,
            apply_spec_augment=True,
            mask_time_prob=0.05,
            mask_time_length=10,
            mask_feature_prob=0.0,
            mask_feature_length=10,
            ctc_loss_reduction="mean",
            ctc_zero_infinity=False,
        )
        
        self.model = Wav2Vec2ForCTC(config)
        
        # Initialize processor with basic tokenizer
        from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
        
        # Create vocabulary
        vocab_dict = {f"<pad>": 0, f"<unk>": 1, f"<s>": 2, f"</s>": 3}
        vocab_dict.update({chr(i): idx + 4 for idx, i in enumerate(range(ord('a'), ord('z') + 1))})
        vocab_dict["|"] = len(vocab_dict)  # Space token
        vocab_dict["'"] = len(vocab_dict)  # Apostrophe
        
        self.tokenizer = Wav2Vec2CTCTokenizer(
            vocab_dict,
            unk_token="<unk>",
            pad_token="<pad>",
            word_delimiter_token="|",
        )
        
        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.config.sample_rate,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True,
        )
        
        self.processor = Wav2Vec2Processor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer
        )
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Wav2Vec2 model
        
        Args:
            audio_features: Input audio waveform [batch_size, seq_len]
            
        Returns:
            Logits from the model
        """
        # Ensure input is properly shaped
        if audio_features.dim() == 3:
            audio_features = audio_features.squeeze(-1)
            
        outputs = self.model(audio_features)
        return outputs.logits
        
    def transcribe(
        self, 
        audio_features: torch.Tensor,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio features to text using Wav2Vec2
        
        Args:
            audio_features: Input audio features [batch_size, seq_len]
            language: Language code (not used for monolingual Wav2Vec2)
            
        Returns:
            Dictionary containing transcription and metadata
        """
        # Get logits
        with torch.no_grad():
            logits = self.forward(audio_features)
            
        # Get predicted token ids
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode to text
        if hasattr(self, 'processor') and hasattr(self.processor, 'batch_decode'):
            transcription = self.processor.batch_decode(predicted_ids)[0]
        else:
            # Manual decoding if processor not available
            transcription = self._decode_tokens(predicted_ids[0])
            
        # Calculate confidence (average probability of selected tokens)
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1)[0].mean().item()
        
        return {
            "text": transcription,
            "confidence": confidence,
            "language": language or "unknown"
        }
        
    def _decode_tokens(self, token_ids: torch.Tensor) -> str:
        """
        Manually decode token IDs to text
        
        Args:
            token_ids: Tensor of token IDs
            
        Returns:
            Decoded text
        """
        # Remove duplicates and special tokens
        decoded_tokens = []
        previous = None
        
        for token_id in token_ids:
            token_id = token_id.item()
            
            # Skip padding and duplicates
            if token_id == 0 or token_id == previous:
                continue
                
            # Map token ID to character
            if hasattr(self, 'tokenizer'):
                token = self.tokenizer.decode([token_id])
            else:
                # Basic mapping for ASCII characters
                if 4 <= token_id <= 29:  # a-z
                    token = chr(ord('a') + token_id - 4)
                elif token_id == 30:  # space
                    token = " "
                else:
                    token = ""
                    
            decoded_tokens.append(token)
            previous = token_id
            
        return "".join(decoded_tokens).strip()
        
    def adapt_vocabulary(self, vocabulary: List[str]) -> None:
        """
        Adapt model to new vocabulary
        
        Args:
            vocabulary: List of vocabulary tokens
        """
        # Create new vocabulary
        vocab_dict = {token: idx for idx, token in enumerate(vocabulary)}
        
        # Update tokenizer
        if hasattr(self, 'processor'):
            self.processor.tokenizer.vocab = vocab_dict
            
        # Resize model embeddings if needed
        if len(vocabulary) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(vocabulary))
            self.vocab_size = len(vocabulary)
            
        logger.info(f"Adapted vocabulary to {len(vocabulary)} tokens")