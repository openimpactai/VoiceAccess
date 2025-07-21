"""
Core ASR Engine for VoiceAccess
Handles speech recognition, model loading, and inference
"""

import torch
import numpy as np
from typing import Union, Optional, Dict, List, Tuple
from pathlib import Path
import logging

from ..models.base_model import BaseASRModel
from ..preprocessing.audio_processor import AudioProcessor
from .config import Config

logger = logging.getLogger(__name__)


class ASREngine:
    """Main ASR Engine for speech recognition"""
    
    def __init__(self, config: Config):
        """
        Initialize ASR Engine with configuration
        
        Args:
            config: Configuration object containing model and processing parameters
        """
        self.config = config
        self.model: Optional[BaseASRModel] = None
        self.audio_processor = AudioProcessor(config)
        self.device = torch.device(config.device)
        
    def load_model(self, model_path: Union[str, Path], model_type: str = "wav2vec2") -> None:
        """
        Load ASR model from checkpoint
        
        Args:
            model_path: Path to model checkpoint
            model_type: Type of model architecture
        """
        logger.info(f"Loading {model_type} model from {model_path}")
        
        if model_type == "wav2vec2":
            from ..models.wav2vec2_model import Wav2Vec2Model
            self.model = Wav2Vec2Model(self.config)
        elif model_type == "whisper":
            from ..models.whisper_model import WhisperModel
            self.model = WhisperModel(self.config)
        elif model_type == "conformer":
            from ..models.conformer_model import ConformerModel
            self.model = ConformerModel(self.config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.model.load_checkpoint(model_path)
        self.model.to(self.device)
        self.model.eval()
        
    def transcribe(
        self, 
        audio_input: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        return_confidence: bool = False
    ) -> Union[str, Tuple[str, float]]:
        """
        Transcribe audio to text
        
        Args:
            audio_input: Path to audio file or numpy array
            language: Language code for multilingual models
            return_confidence: Whether to return confidence scores
            
        Returns:
            Transcribed text or tuple of (text, confidence)
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
            
        # Process audio
        audio_features = self.audio_processor.process(audio_input)
        audio_features = torch.from_numpy(audio_features).to(self.device)
        
        # Add batch dimension if necessary
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(0)
            
        # Run inference
        with torch.no_grad():
            output = self.model.transcribe(
                audio_features, 
                language=language
            )
            
        text = output["text"]
        
        if return_confidence:
            confidence = output.get("confidence", 1.0)
            return text, confidence
            
        return text
        
    def transcribe_batch(
        self,
        audio_inputs: List[Union[str, Path, np.ndarray]],
        language: Optional[str] = None
    ) -> List[str]:
        """
        Transcribe multiple audio files in batch
        
        Args:
            audio_inputs: List of audio file paths or numpy arrays
            language: Language code for multilingual models
            
        Returns:
            List of transcribed texts
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
            
        # Process all audio files
        audio_features_list = []
        for audio in audio_inputs:
            features = self.audio_processor.process(audio)
            audio_features_list.append(torch.from_numpy(features))
            
        # Pad sequences for batch processing
        audio_batch = torch.nn.utils.rnn.pad_sequence(
            audio_features_list, 
            batch_first=True
        ).to(self.device)
        
        # Run batch inference
        with torch.no_grad():
            outputs = self.model.transcribe_batch(
                audio_batch,
                language=language
            )
            
        return [output["text"] for output in outputs]
        
    def adapt_to_language(
        self, 
        language_code: str,
        adaptation_data_path: Union[str, Path]
    ) -> None:
        """
        Adapt model to new language using transfer learning
        
        Args:
            language_code: Target language code
            adaptation_data_path: Path to adaptation dataset
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
            
        logger.info(f"Adapting model to language: {language_code}")
        
        # Load language-specific adapter
        from ..languages.language_adapter import LanguageAdapter
        adapter = LanguageAdapter(
            self.config,
            source_model=self.model,
            target_language=language_code
        )
        
        # Perform adaptation
        adapter.adapt(adaptation_data_path)
        
        # Update model with adapted parameters
        self.model = adapter.get_adapted_model()
        
    def evaluate(
        self,
        test_data_path: Union[str, Path],
        metrics: List[str] = ["wer", "cer"]
    ) -> Dict[str, float]:
        """
        Evaluate model performance on test dataset
        
        Args:
            test_data_path: Path to test dataset
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric names and values
        """
        from ..evaluation.evaluator import ASREvaluator
        
        evaluator = ASREvaluator(self, metrics=metrics)
        results = evaluator.evaluate(test_data_path)
        
        return results