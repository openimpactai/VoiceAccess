"""
ASR evaluation module with various metrics
"""

import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from pathlib import Path
import jiwer
import Levenshtein
from tqdm import tqdm
import logging
import json

logger = logging.getLogger(__name__)


class ASREvaluator:
    """Evaluate ASR model performance with various metrics"""
    
    def __init__(self, asr_engine, metrics: List[str] = None):
        """
        Initialize evaluator
        
        Args:
            asr_engine: ASR engine instance
            metrics: List of metrics to compute
        """
        self.asr_engine = asr_engine
        self.metrics = metrics or ["wer", "cer", "mer", "wil"]
        
        # Metric functions
        self.metric_functions = {
            "wer": self.compute_wer,
            "cer": self.compute_cer,
            "mer": self.compute_mer,
            "wil": self.compute_wil,
            "rtf": self.compute_rtf,
            "confidence": self.compute_confidence_metrics
        }
        
    def evaluate(
        self,
        test_data_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, float]:
        """
        Evaluate ASR model on test dataset
        
        Args:
            test_data_path: Path to test dataset
            output_path: Optional path to save detailed results
            
        Returns:
            Dictionary of metric values
        """
        test_data_path = Path(test_data_path)
        
        if not test_data_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_data_path}")
            
        # Load test data
        test_samples = self._load_test_data(test_data_path)
        
        if not test_samples:
            logger.warning("No test samples found")
            return {}
            
        logger.info(f"Evaluating on {len(test_samples)} samples")
        
        # Perform transcriptions
        results = []
        total_audio_duration = 0
        total_inference_time = 0
        
        for sample in tqdm(test_samples, desc="Transcribing"):
            audio_path = sample['audio_path']
            reference = sample['transcript']
            
            # Measure inference time
            import time
            start_time = time.time()
            
            # Transcribe
            try:
                hypothesis, confidence = self.asr_engine.transcribe(
                    audio_path,
                    return_confidence=True
                )
                inference_time = time.time() - start_time
                
                # Get audio duration
                from ..preprocessing.audio_processor import AudioProcessor
                processor = AudioProcessor(self.asr_engine.config)
                waveform = processor.load_audio(audio_path)
                audio_duration = len(waveform) / self.asr_engine.config.sample_rate
                
                results.append({
                    'audio_path': str(audio_path),
                    'reference': reference,
                    'hypothesis': hypothesis,
                    'confidence': confidence,
                    'audio_duration': audio_duration,
                    'inference_time': inference_time
                })
                
                total_audio_duration += audio_duration
                total_inference_time += inference_time
                
            except Exception as e:
                logger.error(f"Error transcribing {audio_path}: {e}")
                results.append({
                    'audio_path': str(audio_path),
                    'reference': reference,
                    'hypothesis': "",
                    'confidence': 0.0,
                    'audio_duration': 0,
                    'inference_time': 0,
                    'error': str(e)
                })
                
        # Compute metrics
        metrics_results = {}
        
        for metric in self.metrics:
            if metric in self.metric_functions:
                try:
                    value = self.metric_functions[metric](results)
                    metrics_results[metric] = value
                except Exception as e:
                    logger.error(f"Error computing {metric}: {e}")
                    
        # Add RTF if timing data available
        if total_audio_duration > 0:
            rtf = total_inference_time / total_audio_duration
            metrics_results['rtf'] = rtf
            
        # Save detailed results if requested
        if output_path:
            self._save_results(results, metrics_results, output_path)
            
        # Log summary
        logger.info("Evaluation Results:")
        for metric, value in metrics_results.items():
            if isinstance(value, dict):
                for sub_metric, sub_value in value.items():
                    logger.info(f"  {metric}.{sub_metric}: {sub_value:.4f}")
            else:
                logger.info(f"  {metric}: {value:.4f}")
                
        return metrics_results
        
    def _load_test_data(self, test_data_path: Path) -> List[Dict[str, Any]]:
        """Load test dataset"""
        samples = []
        
        # Check for transcripts file
        transcript_file = test_data_path / "transcripts.txt"
        if transcript_file.exists():
            with open(transcript_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        audio_file = parts[0]
                        transcript = parts[1]
                        
                        audio_path = test_data_path / "audio" / audio_file
                        if audio_path.exists():
                            samples.append({
                                'audio_path': audio_path,
                                'transcript': transcript
                            })
                            
        # Alternative: JSON format
        json_file = test_data_path / "test_data.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    audio_path = test_data_path / item['audio_path']
                    if audio_path.exists():
                        samples.append({
                            'audio_path': audio_path,
                            'transcript': item['transcript']
                        })
                        
        return samples
        
    def compute_wer(self, results: List[Dict[str, Any]]) -> float:
        """
        Compute Word Error Rate (WER)
        
        Args:
            results: List of transcription results
            
        Returns:
            WER value (0-1)
        """
        references = [r['reference'] for r in results if 'error' not in r]
        hypotheses = [r['hypothesis'] for r in results if 'error' not in r]
        
        if not references:
            return 1.0
            
        # Normalize texts
        references = [self._normalize_text(ref) for ref in references]
        hypotheses = [self._normalize_text(hyp) for hyp in hypotheses]
        
        # Compute WER
        wer = jiwer.wer(references, hypotheses)
        
        return wer
        
    def compute_cer(self, results: List[Dict[str, Any]]) -> float:
        """
        Compute Character Error Rate (CER)
        
        Args:
            results: List of transcription results
            
        Returns:
            CER value (0-1)
        """
        total_chars = 0
        total_errors = 0
        
        for result in results:
            if 'error' in result:
                continue
                
            ref = self._normalize_text(result['reference'])
            hyp = self._normalize_text(result['hypothesis'])
            
            # Compute character-level distance
            distance = Levenshtein.distance(ref, hyp)
            
            total_errors += distance
            total_chars += len(ref)
            
        if total_chars == 0:
            return 1.0
            
        cer = total_errors / total_chars
        
        return cer
        
    def compute_mer(self, results: List[Dict[str, Any]]) -> float:
        """
        Compute Match Error Rate (MER)
        
        Args:
            results: List of transcription results
            
        Returns:
            MER value (0-1)
        """
        references = [r['reference'] for r in results if 'error' not in r]
        hypotheses = [r['hypothesis'] for r in results if 'error' not in r]
        
        if not references:
            return 1.0
            
        # Normalize texts
        references = [self._normalize_text(ref) for ref in references]
        hypotheses = [self._normalize_text(hyp) for hyp in hypotheses]
        
        # Compute MER
        mer = jiwer.mer(references, hypotheses)
        
        return mer
        
    def compute_wil(self, results: List[Dict[str, Any]]) -> float:
        """
        Compute Word Information Lost (WIL)
        
        Args:
            results: List of transcription results
            
        Returns:
            WIL value (0-1)
        """
        references = [r['reference'] for r in results if 'error' not in r]
        hypotheses = [r['hypothesis'] for r in results if 'error' not in r]
        
        if not references:
            return 1.0
            
        # Normalize texts
        references = [self._normalize_text(ref) for ref in references]
        hypotheses = [self._normalize_text(hyp) for hyp in hypotheses]
        
        # Compute WIL
        wil = jiwer.wil(references, hypotheses)
        
        return wil
        
    def compute_rtf(self, results: List[Dict[str, Any]]) -> float:
        """
        Compute Real-Time Factor (RTF)
        
        Args:
            results: List of transcription results
            
        Returns:
            RTF value (lower is better)
        """
        total_audio_duration = sum(r.get('audio_duration', 0) for r in results)
        total_inference_time = sum(r.get('inference_time', 0) for r in results)
        
        if total_audio_duration == 0:
            return float('inf')
            
        rtf = total_inference_time / total_audio_duration
        
        return rtf
        
    def compute_confidence_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Compute confidence-related metrics
        
        Args:
            results: List of transcription results
            
        Returns:
            Dictionary of confidence metrics
        """
        confidences = [r['confidence'] for r in results if 'error' not in r]
        
        if not confidences:
            return {
                'mean_confidence': 0.0,
                'std_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0
            }
            
        return {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences)
        }
        
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for evaluation
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        import string
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Normalize whitespace
        text = ' '.join(text.split())
        
        return text
        
    def _save_results(
        self,
        results: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        output_path: Union[str, Path]
    ):
        """Save detailed evaluation results"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare output data
        output_data = {
            'metrics': metrics,
            'detailed_results': results,
            'config': {
                'model_type': self.asr_engine.config.model_type,
                'metrics_computed': self.metrics
            }
        }
        
        # Save as JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Detailed results saved to {output_path}")
        
    def evaluate_language_specific(
        self,
        test_data_path: Union[str, Path],
        language: str
    ) -> Dict[str, float]:
        """
        Evaluate with language-specific considerations
        
        Args:
            test_data_path: Path to test dataset
            language: Language code
            
        Returns:
            Dictionary of metric values
        """
        # Set language for transcription
        original_transcribe = self.asr_engine.transcribe
        
        def language_specific_transcribe(audio_input, **kwargs):
            kwargs['language'] = language
            return original_transcribe(audio_input, **kwargs)
            
        self.asr_engine.transcribe = language_specific_transcribe
        
        try:
            # Run evaluation
            results = self.evaluate(test_data_path)
            
            # Add language-specific metrics if needed
            if language in ['zh', 'ja', 'ko']:  # Character-based languages
                # CER is more relevant than WER
                results['primary_metric'] = results.get('cer', 1.0)
            else:
                results['primary_metric'] = results.get('wer', 1.0)
                
            return results
            
        finally:
            # Restore original method
            self.asr_engine.transcribe = original_transcribe