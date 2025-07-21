"""
Whisper model implementation for VoiceAccess
"""

import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperForConditionalGeneration, WhisperProcessor
from typing import Dict, Any, Optional, List
import logging
import numpy as np

from .base_model import BaseASRModel

logger = logging.getLogger(__name__)


class WhisperModel(BaseASRModel):
    """OpenAI Whisper model for multilingual speech recognition"""
    
    def __init__(self, config):
        """
        Initialize Whisper model
        
        Args:
            config: Configuration object
        """
        super().__init__(config)
        
        self.model_name = config.model_name or "openai/whisper-base"
        
        # Load processor and model
        try:
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_name,
                dropout=0.1,
                attention_dropout=0.1,
                activation_dropout=0.0,
                use_cache=True,
            )
            
            # Set language tokens
            self.supported_languages = [
                "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",
                "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
                "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
                "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
                "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
                "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
                "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
                "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
                "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
                "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
            ]
            
        except Exception as e:
            logger.warning(f"Could not load pretrained model {self.model_name}: {e}")
            logger.info("Initializing Whisper model from scratch")
            self._init_from_scratch()
            
    def _init_from_scratch(self):
        """Initialize model from scratch when pretrained is not available"""
        from transformers import WhisperConfig
        
        config = WhisperConfig(
            vocab_size=51865,
            num_mel_bins=80,
            encoder_layers=6,
            encoder_attention_heads=8,
            decoder_layers=6,
            decoder_attention_heads=8,
            decoder_ffn_dim=1536,
            encoder_ffn_dim=1536,
            encoder_layerdrop=0.0,
            decoder_layerdrop=0.0,
            decoder_start_token_id=50258,
            use_cache=True,
            is_encoder_decoder=True,
            activation_function="gelu",
            d_model=384,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            init_std=0.02,
            scale_embedding=False,
            max_source_positions=1500,
            max_target_positions=448,
            pad_token_id=50257,
            bos_token_id=50257,
            eos_token_id=50257,
            suppress_tokens=[1, 2, 7, 8, 9, 10, 14, 25, 26, 27, 28, 29, 31, 58, 59, 60, 61, 62, 63, 90, 91, 92, 93, 357, 366, 438, 532, 685, 705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377, 1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211, 4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786, 11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791, 17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409, 34949, 40283, 40493, 40549, 47282, 49146, 50257, 50357, 50358, 50359, 50360, 50361],
            begin_suppress_tokens=[220, 50257],
        )
        
        self.model = WhisperForConditionalGeneration(config)
        
        # Initialize processor
        from transformers import WhisperTokenizer, WhisperFeatureExtractor
        
        self.feature_extractor = WhisperFeatureExtractor(
            feature_size=80,
            sampling_rate=16000,
            hop_length=160,
            chunk_length=30,
            n_fft=400,
            padding_value=0.0,
            return_attention_mask=True,
        )
        
        # Basic tokenizer initialization
        self.tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
        
        self.processor = WhisperProcessor(
            feature_extractor=self.feature_extractor,
            tokenizer=self.tokenizer
        )
    
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Whisper encoder
        
        Args:
            audio_features: Mel spectrogram features [batch_size, n_mels, time]
            
        Returns:
            Encoder hidden states
        """
        encoder_outputs = self.model.model.encoder(
            audio_features,
            return_dict=True
        )
        return encoder_outputs.last_hidden_state
        
    def transcribe(
        self, 
        audio_features: torch.Tensor,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio features to text using Whisper
        
        Args:
            audio_features: Input audio features (mel spectrogram or waveform)
            language: Target language code
            
        Returns:
            Dictionary containing transcription and metadata
        """
        # Process audio to mel spectrogram if needed
        if audio_features.dim() == 2 and audio_features.shape[-1] > 1000:
            # Assume it's raw waveform
            audio_features = self._extract_mel_features(audio_features)
            
        # Ensure correct shape [batch_size, n_mels, time]
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
            
        # Set language token if specified
        if language and language in self.supported_languages:
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language, 
                task="transcribe"
            )
        else:
            forced_decoder_ids = None
            
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(
                audio_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448,
                num_beams=5,
                temperature=0.0,
                return_dict_in_generate=True,
                output_scores=True,
            )
            
        # Decode tokens to text
        transcription = self.processor.batch_decode(
            generated_ids.sequences, 
            skip_special_tokens=True
        )[0]
        
        # Calculate confidence from generation scores
        if hasattr(generated_ids, 'scores') and generated_ids.scores:
            # Average log probability of generated tokens
            scores = torch.stack(generated_ids.scores, dim=1)
            probs = torch.softmax(scores, dim=-1)
            selected_probs = probs.max(dim=-1)[0]
            confidence = selected_probs.mean().item()
        else:
            confidence = 1.0
            
        # Detect language if not specified
        if not language:
            language = self._detect_language(audio_features)
            
        return {
            "text": transcription.strip(),
            "confidence": confidence,
            "language": language
        }
        
    def _extract_mel_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract mel spectrogram features from waveform
        
        Args:
            waveform: Audio waveform [batch_size, seq_len]
            
        Returns:
            Mel spectrogram features
        """
        # Use processor's feature extractor
        if hasattr(self, 'processor'):
            # Convert to numpy for processing
            waveform_np = waveform.cpu().numpy()
            features = self.processor.feature_extractor(
                waveform_np,
                sampling_rate=self.config.sample_rate,
                return_tensors="pt"
            )
            return features.input_features.to(waveform.device)
        else:
            # Manual mel spectrogram extraction
            return self._manual_mel_extraction(waveform)
            
    def _manual_mel_extraction(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Manually extract mel spectrogram (fallback method)
        
        Args:
            waveform: Audio waveform
            
        Returns:
            Mel spectrogram
        """
        # This is a simplified version - in practice, use torchaudio or librosa
        # Placeholder implementation
        batch_size = waveform.shape[0]
        n_frames = waveform.shape[1] // self.config.hop_length
        mel_features = torch.randn(
            batch_size, 
            self.config.n_mels, 
            n_frames,
            device=waveform.device
        )
        return mel_features
        
    def _detect_language(self, audio_features: torch.Tensor) -> str:
        """
        Detect language from audio
        
        Args:
            audio_features: Mel spectrogram features
            
        Returns:
            Detected language code
        """
        # Use Whisper's language detection
        with torch.no_grad():
            # Get encoder outputs
            encoder_outputs = self.model.model.encoder(audio_features)
            
            # Get language token probabilities
            decoder_input_ids = torch.tensor([[50258]], device=audio_features.device)  # <|startoftranscript|>
            
            outputs = self.model.model.decoder(
                decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
            )
            
            logits = self.model.lm_head(outputs.last_hidden_state)
            
            # Language tokens are in specific range
            lang_token_ids = list(range(50259, 50359))  # Language token IDs
            lang_logits = logits[0, 0, lang_token_ids]
            lang_probs = torch.softmax(lang_logits, dim=-1)
            
            # Get most probable language
            max_prob_idx = lang_probs.argmax().item()
            
            # Map to language code (simplified - in practice use proper mapping)
            if max_prob_idx < len(self.supported_languages):
                return self.supported_languages[max_prob_idx]
            else:
                return "en"  # Default to English
                
    def translate(
        self,
        audio_features: torch.Tensor,
        target_language: str = "en"
    ) -> Dict[str, Any]:
        """
        Translate speech to target language
        
        Args:
            audio_features: Input audio features
            target_language: Target language for translation
            
        Returns:
            Dictionary containing translation and metadata
        """
        # Whisper primarily translates to English
        if target_language != "en":
            logger.warning(f"Whisper only supports translation to English. Ignoring target_language={target_language}")
            
        # Process audio
        if audio_features.dim() == 2 and audio_features.shape[-1] > 1000:
            audio_features = self._extract_mel_features(audio_features)
            
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
            
        # Generate translation
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            task="translate"
        )
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                audio_features,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448,
                num_beams=5,
                temperature=0.0,
            )
            
        translation = self.processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True
        )[0]
        
        return {
            "text": translation.strip(),
            "target_language": "en",
            "task": "translate"
        }