"""
Language adaptation module for transfer learning to new languages
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import logging
from tqdm import tqdm

from ..models.base_model import BaseASRModel
from ..augmentation.audio_augmentor import AudioAugmentor
from ..preprocessing.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class LanguageAdapter:
    """Adapt ASR models to new languages using transfer learning"""
    
    def __init__(
        self,
        config,
        source_model: BaseASRModel,
        target_language: str
    ):
        """
        Initialize language adapter
        
        Args:
            config: Configuration object
            source_model: Pre-trained source model
            target_language: Target language code
        """
        self.config = config
        self.source_model = source_model
        self.target_language = target_language
        self.device = torch.device(config.device)
        
        # Initialize components
        self.audio_processor = AudioProcessor(config)
        self.augmentor = AudioAugmentor(config)
        
        # Create adapter layers
        self._create_adapter_layers()
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.best_loss = float('inf')
        
    def _create_adapter_layers(self):
        """Create adapter layers for the model"""
        # Get model hidden size
        if hasattr(self.source_model, 'model'):
            if hasattr(self.source_model.model, 'config'):
                hidden_size = self.source_model.model.config.hidden_size
            else:
                hidden_size = self.config.hidden_size
        else:
            hidden_size = self.config.hidden_size
            
        # Create adapter modules
        self.adapters = nn.ModuleDict({
            'encoder_adapter': AdapterLayer(
                hidden_size,
                self.config.adapter_hidden_size,
                self.config.adapter_dropout
            ),
            'decoder_adapter': AdapterLayer(
                hidden_size,
                self.config.adapter_hidden_size,
                self.config.adapter_dropout
            ) if hasattr(self.source_model, 'decoder') else None
        })
        
        # Language-specific output layer
        self.language_head = nn.Linear(
            hidden_size,
            self.config.vocab_size
        )
        
        # Move to device
        self.adapters.to(self.device)
        self.language_head.to(self.device)
        
    def adapt(
        self,
        adaptation_data_path: Union[str, Path],
        num_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Adapt model to target language
        
        Args:
            adaptation_data_path: Path to adaptation dataset
            num_epochs: Number of training epochs
            learning_rate: Learning rate for adaptation
            
        Returns:
            Dictionary of adaptation metrics
        """
        adaptation_data_path = Path(adaptation_data_path)
        
        if not adaptation_data_path.exists():
            raise FileNotFoundError(f"Adaptation data not found: {adaptation_data_path}")
            
        # Set training parameters
        num_epochs = num_epochs or self.config.num_epochs
        learning_rate = learning_rate or self.config.learning_rate
        
        # Freeze source model if configured
        if self.config.freeze_base_model:
            self.source_model.freeze_base_model()
            
        # Setup optimizer
        self._setup_optimizer(learning_rate)
        
        # Load adaptation data
        train_loader = self._load_adaptation_data(adaptation_data_path / "train")
        val_loader = self._load_adaptation_data(adaptation_data_path / "dev")
        
        # Training loop
        logger.info(f"Starting adaptation for {self.target_language}")
        metrics = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self._train_epoch(train_loader, epoch)
            metrics['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self._validate(val_loader)
            metrics['val_loss'].append(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self._save_checkpoint(f"adapter_{self.target_language}_best.pt")
                
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
                
        # Final evaluation
        test_loader = self._load_adaptation_data(adaptation_data_path / "test")
        if test_loader:
            test_metrics = self._evaluate(test_loader)
            metrics.update(test_metrics)
            
        return metrics
        
    def _setup_optimizer(self, learning_rate: float):
        """Setup optimizer and scheduler"""
        # Get adapter parameters
        adapter_params = list(self.adapters.parameters())
        adapter_params += list(self.language_head.parameters())
        
        # Add unfrozen model parameters if any
        if not self.config.freeze_base_model:
            adapter_params += list(self.source_model.parameters())
            
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            adapter_params,
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Create scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
    def _load_adaptation_data(self, data_path: Path):
        """Load adaptation dataset"""
        if not data_path.exists():
            return None
            
        # Simple data loading - in practice, use proper dataset class
        from torch.utils.data import DataLoader, Dataset
        
        class AdaptationDataset(Dataset):
            def __init__(self, data_path, processor, augmentor):
                self.data_path = data_path
                self.processor = processor
                self.augmentor = augmentor
                
                # Load audio-transcript pairs
                self.samples = []
                transcript_file = data_path / "transcripts.txt"
                
                if transcript_file.exists():
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) == 2:
                                audio_path = data_path / "audio" / parts[0]
                                transcript = parts[1]
                                if audio_path.exists():
                                    self.samples.append((audio_path, transcript))
                                    
            def __len__(self):
                return len(self.samples)
                
            def __getitem__(self, idx):
                audio_path, transcript = self.samples[idx]
                
                # Load and process audio
                waveform = self.processor.load_audio(audio_path)
                
                # Apply augmentation during training
                if self.augmentor and self.augmentor.augmentation_enabled:
                    waveform = self.augmentor.augment(waveform)
                    
                # Process features
                features = self.processor.process(waveform)
                
                return {
                    'features': torch.from_numpy(features).float(),
                    'transcript': transcript,
                    'audio_path': str(audio_path)
                }
                
        # Create dataset
        dataset = AdaptationDataset(data_path, self.audio_processor, self.augmentor)
        
        if len(dataset) == 0:
            logger.warning(f"No samples found in {data_path}")
            return None
            
        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self._collate_fn
        )
        
        return loader
        
    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        # Pad sequences
        features = [item['features'] for item in batch]
        transcripts = [item['transcript'] for item in batch]
        
        # Pad features
        features_padded = torch.nn.utils.rnn.pad_sequence(
            features,
            batch_first=True,
            padding_value=0
        )
        
        return {
            'features': features_padded,
            'transcripts': transcripts
        }
        
    def _train_epoch(self, train_loader, epoch: int) -> float:
        """Train for one epoch"""
        self.source_model.train()
        self.adapters.train()
        self.language_head.train()
        
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress_bar:
            # Move to device
            features = batch['features'].to(self.device)
            
            # Forward pass through source model
            with torch.set_grad_enabled(not self.config.freeze_base_model):
                if hasattr(self.source_model, 'model'):
                    # Get encoder outputs
                    encoder_outputs = self.source_model.model.model.encoder(features)
                    hidden_states = encoder_outputs.last_hidden_state
                else:
                    hidden_states = self.source_model(features)
                    
            # Apply adapter
            adapted_features = self.adapters['encoder_adapter'](hidden_states)
            
            # Language-specific head
            logits = self.language_head(adapted_features)
            
            # Compute loss (simplified - in practice, use proper loss computation)
            loss = self._compute_loss(logits, batch['transcripts'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.adapters.parameters(), 
                max_norm=1.0
            )
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
        return total_loss / num_batches
        
    def _validate(self, val_loader) -> float:
        """Validate on validation set"""
        if not val_loader:
            return 0.0
            
        self.source_model.eval()
        self.adapters.eval()
        self.language_head.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                
                # Forward pass
                if hasattr(self.source_model, 'model'):
                    encoder_outputs = self.source_model.model.model.encoder(features)
                    hidden_states = encoder_outputs.last_hidden_state
                else:
                    hidden_states = self.source_model(features)
                    
                # Apply adapter
                adapted_features = self.adapters['encoder_adapter'](hidden_states)
                logits = self.language_head(adapted_features)
                
                # Compute loss
                loss = self._compute_loss(logits, batch['transcripts'])
                
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
        
    def _evaluate(self, test_loader) -> Dict[str, float]:
        """Evaluate on test set"""
        # Placeholder for evaluation metrics
        # In practice, compute WER, CER, etc.
        return {
            'test_wer': 0.0,
            'test_cer': 0.0
        }
        
    def _compute_loss(self, logits: torch.Tensor, transcripts: List[str]) -> torch.Tensor:
        """
        Compute CTC loss
        
        Args:
            logits: Model outputs [batch, time, vocab]
            transcripts: Target transcripts
            
        Returns:
            Loss value
        """
        # Simplified loss computation
        # In practice, properly encode transcripts and compute CTC loss
        batch_size = logits.size(0)
        
        # Dummy loss for now
        loss = logits.mean() * 0.1
        
        return loss
        
    def _save_checkpoint(self, filename: str):
        """Save adapter checkpoint"""
        checkpoint_path = self.config.model_dir / filename
        
        checkpoint = {
            'adapters_state_dict': self.adapters.state_dict(),
            'language_head_state_dict': self.language_head.state_dict(),
            'target_language': self.target_language,
            'config': self.config.__dict__,
            'best_loss': self.best_loss
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
    def get_adapted_model(self) -> BaseASRModel:
        """Get adapted model with integrated adapters"""
        # Create a wrapper model that includes adapters
        adapted_model = AdaptedASRModel(
            self.source_model,
            self.adapters,
            self.language_head,
            self.config
        )
        
        return adapted_model


class AdapterLayer(nn.Module):
    """Adapter layer for efficient fine-tuning"""
    
    def __init__(self, input_size: int, adapter_size: int, dropout: float = 0.1):
        super().__init__()
        
        self.down_projection = nn.Linear(input_size, adapter_size)
        self.activation = nn.ReLU()
        self.up_projection = nn.Linear(adapter_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        residual = x
        
        # Down projection
        x = self.down_projection(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Up projection
        x = self.up_projection(x)
        x = self.dropout(x)
        
        # Residual connection
        x = x + residual
        x = self.layer_norm(x)
        
        return x


class AdaptedASRModel(BaseASRModel):
    """Wrapper model with integrated adapters"""
    
    def __init__(self, base_model, adapters, language_head, config):
        super().__init__(config)
        self.base_model = base_model
        self.adapters = adapters
        self.language_head = language_head
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        # Forward through base model
        if hasattr(self.base_model, 'model'):
            encoder_outputs = self.base_model.model.model.encoder(audio_features)
            hidden_states = encoder_outputs.last_hidden_state
        else:
            hidden_states = self.base_model(audio_features)
            
        # Apply adapter
        adapted_features = self.adapters['encoder_adapter'](hidden_states)
        
        # Language-specific head
        logits = self.language_head(adapted_features)
        
        return logits
        
    def transcribe(self, audio_features: torch.Tensor, language: Optional[str] = None) -> Dict[str, Any]:
        # Use adapted forward pass
        logits = self.forward(audio_features)
        
        # Decode using base model's decoding logic
        # Simplified version - adapt based on actual model
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Convert to text (placeholder)
        text = "Adapted transcription"
        
        return {
            "text": text,
            "language": language or self.config.target_language,
            "adapted": True
        }