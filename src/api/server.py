"""
FastAPI server for VoiceAccess ASR service
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import torch
import numpy as np
import io
import soundfile as sf
import asyncio
from pathlib import Path
import logging
import uuid
from datetime import datetime
import uvicorn

from ..core.asr_engine import ASREngine
from ..core.config import Config

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VoiceAccess API",
    description="ASR API for low-resource and endangered languages",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
asr_engine: Optional[ASREngine] = None
config: Optional[Config] = None
active_requests: Dict[str, Dict[str, Any]] = {}


# Pydantic models
class TranscriptionRequest(BaseModel):
    language: Optional[str] = Field(None, description="Target language code")
    return_confidence: bool = Field(False, description="Return confidence scores")
    enable_augmentation: bool = Field(False, description="Apply augmentation during inference")


class TranscriptionResponse(BaseModel):
    request_id: str
    text: str
    language: str
    confidence: Optional[float] = None
    processing_time: float
    timestamp: str


class BatchTranscriptionRequest(BaseModel):
    files: List[str] = Field(..., description="List of file IDs to transcribe")
    language: Optional[str] = None
    return_confidence: bool = False


class ModelInfoResponse(BaseModel):
    model_type: str
    model_name: str
    supported_languages: List[str]
    vocab_size: int
    device: str
    model_size_mb: float


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    active_requests: int


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize ASR engine on startup"""
    global asr_engine, config
    
    logger.info("Starting VoiceAccess API server...")
    
    # Load configuration
    config_path = Path("configs/default.yaml")
    if config_path.exists():
        config = Config.from_file(config_path)
    else:
        config = Config()
        
    # Initialize ASR engine
    asr_engine = ASREngine(config)
    
    # Load default model if available
    default_model = Path("models/pretrained/wav2vec2-base.pt")
    if default_model.exists():
        try:
            asr_engine.load_model(default_model, model_type="wav2vec2")
            logger.info("Default model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load default model: {e}")
            
    logger.info("VoiceAccess API server started")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down VoiceAccess API server...")
    # Cleanup resources if needed


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=asr_engine is not None and asr_engine.model is not None,
        active_requests=len(active_requests)
    )


# Model information endpoint
@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about loaded model"""
    if asr_engine is None or asr_engine.model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
        
    model_info = ModelInfoResponse(
        model_type=config.model_type,
        model_name=config.model_name,
        supported_languages=["en", "es", "fr", "de", "zh"],  # Example languages
        vocab_size=config.vocab_size,
        device=config.device,
        model_size_mb=asr_engine.model.get_model_size_mb()
    )
    
    return model_info


# Load model endpoint
@app.post("/model/load")
async def load_model(
    model_path: str,
    model_type: str = "wav2vec2"
):
    """Load a specific model"""
    if asr_engine is None:
        raise HTTPException(status_code=503, detail="ASR engine not initialized")
        
    try:
        asr_engine.load_model(model_path, model_type=model_type)
        return {"message": f"Model loaded successfully: {model_path}"}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=400, detail=str(e))


# Single file transcription endpoint
@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    request: TranscriptionRequest = TranscriptionRequest()
):
    """Transcribe a single audio file"""
    if asr_engine is None or asr_engine.model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
        
    # Generate request ID
    request_id = str(uuid.uuid4())
    start_time = datetime.utcnow()
    
    # Track active request
    active_requests[request_id] = {
        "status": "processing",
        "start_time": start_time
    }
    
    try:
        # Read audio file
        audio_data = await file.read()
        
        # Load audio using soundfile
        audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
        
        # Resample if needed
        if sample_rate != config.sample_rate:
            import librosa
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sample_rate,
                target_sr=config.sample_rate
            )
            
        # Transcribe
        if request.return_confidence:
            text, confidence = asr_engine.transcribe(
                audio_array,
                language=request.language,
                return_confidence=True
            )
        else:
            text = asr_engine.transcribe(
                audio_array,
                language=request.language
            )
            confidence = None
            
        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Update request status
        active_requests[request_id]["status"] = "completed"
        
        # Clean up request after delay
        background_tasks.add_task(cleanup_request, request_id, delay=300)
        
        return TranscriptionResponse(
            request_id=request_id,
            text=text,
            language=request.language or "auto",
            confidence=confidence,
            processing_time=processing_time,
            timestamp=start_time.isoformat()
        )
        
    except Exception as e:
        # Update request status
        active_requests[request_id]["status"] = "failed"
        active_requests[request_id]["error"] = str(e)
        
        logger.error(f"Transcription error for request {request_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Batch transcription endpoint
@app.post("/transcribe/batch")
async def transcribe_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    language: Optional[str] = None
):
    """Transcribe multiple audio files"""
    if asr_engine is None or asr_engine.model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
        
    batch_id = str(uuid.uuid4())
    results = []
    
    for file in files:
        try:
            # Read audio
            audio_data = await file.read()
            audio_array, sample_rate = sf.read(io.BytesIO(audio_data))
            
            # Resample if needed
            if sample_rate != config.sample_rate:
                import librosa
                audio_array = librosa.resample(
                    audio_array,
                    orig_sr=sample_rate,
                    target_sr=config.sample_rate
                )
                
            # Transcribe
            text = asr_engine.transcribe(audio_array, language=language)
            
            results.append({
                "filename": file.filename,
                "text": text,
                "status": "success"
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "text": "",
                "status": "error",
                "error": str(e)
            })
            
    return {
        "batch_id": batch_id,
        "results": results,
        "total": len(files),
        "successful": sum(1 for r in results if r["status"] == "success")
    }


# Language adaptation endpoint
@app.post("/adapt/language")
async def adapt_to_language(
    language_code: str,
    adaptation_data_path: str,
    num_epochs: int = 10
):
    """Adapt model to a new language"""
    if asr_engine is None or asr_engine.model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
        
    try:
        # Start adaptation in background
        asyncio.create_task(
            run_adaptation(language_code, adaptation_data_path, num_epochs)
        )
        
        return {
            "message": f"Adaptation started for language: {language_code}",
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Adaptation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Request status endpoint
@app.get("/request/{request_id}/status")
async def get_request_status(request_id: str):
    """Get status of a transcription request"""
    if request_id not in active_requests:
        raise HTTPException(status_code=404, detail="Request not found")
        
    return active_requests[request_id]


# Supported languages endpoint
@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages"""
    # This should be dynamically determined based on available models
    languages = [
        {"code": "en", "name": "English", "status": "supported"},
        {"code": "es", "name": "Spanish", "status": "supported"},
        {"code": "fr", "name": "French", "status": "supported"},
        {"code": "de", "name": "German", "status": "supported"},
        {"code": "zh", "name": "Chinese", "status": "supported"},
        {"code": "qu", "name": "Quechua", "status": "experimental"},
        {"code": "mi", "name": "Maori", "status": "experimental"},
        {"code": "cy", "name": "Welsh", "status": "experimental"},
    ]
    
    return {"languages": languages}


# Helper functions
async def cleanup_request(request_id: str, delay: int = 300):
    """Clean up completed request after delay"""
    await asyncio.sleep(delay)
    if request_id in active_requests:
        del active_requests[request_id]


async def run_adaptation(language_code: str, data_path: str, num_epochs: int):
    """Run language adaptation asynchronously"""
    try:
        asr_engine.adapt_to_language(language_code, data_path)
        logger.info(f"Adaptation completed for language: {language_code}")
    except Exception as e:
        logger.error(f"Adaptation failed for language {language_code}: {e}")


# Main function for running the server
def main():
    """Run the API server"""
    uvicorn.run(
        "src.api.server:app",
        host=config.api_host if config else "0.0.0.0",
        port=config.api_port if config else 8000,
        workers=config.api_workers if config else 4,
        log_level="info"
    )


if __name__ == "__main__":
    main()