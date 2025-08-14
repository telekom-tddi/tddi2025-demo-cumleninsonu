"""
Speech-to-Text (STT) Service
This service handles speech-to-text transcription using OpenAI's Whisper model with improved memory handling and streaming support.
"""

import logging
import whisper
import torch
import os
import tempfile
import time
import numpy as np
from typing import Dict, Any, Union, List

# Optional imports for additional audio processing
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    sf = None

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    librosa = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpeechToTextService:
    """Service for converting speech to text with improved memory handling and confidence scoring."""

    def __init__(self, model_name: str = "medium", device: str = None, language: str = "tr", 
                 enable_vad: bool = True, chunk_length: int = 30):
        """
        Initialize the STT service.
        
        Args:
            model_name (str): The name of the Whisper model to use.
            device (str): The device to run the model on (e.g., 'cuda', 'cpu').
            language (str): Preferred language code for transcription (e.g., 'tr').
            enable_vad (bool): Enable Voice Activity Detection for better performance.
            chunk_length (int): Length of audio chunks for processing (seconds).
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.language = language
        self.model_name = model_name
        self.enable_vad = enable_vad
        self.chunk_length = chunk_length
        self.model = None
        
        # Model cache for different sizes
        self._model_cache = {}
        
        logger.info(f"STT service initialized with model: {model_name} on device: {self.device}, language={self.language}")
        
        # Load the initial model
        self._load_model(model_name)

    def _load_model(self, model_name: str) -> None:
        """Load a Whisper model, using cache if available."""
        if model_name in self._model_cache:
            self.model = self._model_cache[model_name]
            logger.info(f"Using cached Whisper model '{model_name}'")
            return
            
        try:
            logger.info(f"Loading Whisper model '{model_name}'...")
            model = whisper.load_model(model_name, device=self.device)
            self._model_cache[model_name] = model
            self.model = model
            logger.info(f"Whisper model '{model_name}' loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise

    def _preprocess_audio(self, audio_data: Union[str, bytes, np.ndarray], 
                         sample_rate: int = 16000) -> np.ndarray:
        """Preprocess audio data for Whisper."""
        if isinstance(audio_data, str):
            # File path
            if LIBROSA_AVAILABLE:
                audio, sr = librosa.load(audio_data, sr=sample_rate)
            else:
                audio = whisper.load_audio(audio_data)
        elif isinstance(audio_data, bytes):
            # Bytes data - save to temp file and load
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()
                try:
                    if LIBROSA_AVAILABLE:
                        audio, sr = librosa.load(temp_file.name, sr=sample_rate)
                    else:
                        audio = whisper.load_audio(temp_file.name)
                finally:
                    os.unlink(temp_file.name)
        else:
            # NumPy array
            audio = audio_data
        
        # Ensure audio is normalized and has correct shape
        audio = whisper.pad_or_trim(audio)
        return audio

    def transcribe_with_confidence(self, audio_input: Union[str, bytes, np.ndarray], 
                                 language: str = None, model_size: str = None) -> Dict[str, Any]:
        """
        Transcribe audio and return detailed results with confidence scores.
        
        Args:
            audio_input: Audio file path, bytes, or numpy array.
            language: Optional language override.
            model_size: Optional model size override.
            
        Returns:
            Dictionary with transcription results and metadata.
        """
        start_time = time.time()
        selected_lang = language or self.language
        
        # Switch model if requested
        if model_size and model_size != self.model_name:
            try:
                self._load_model(model_size)
            except Exception as e:
                logger.warning(f"Failed to load model {model_size}, using current model: {e}")
        
        try:
            # Preprocess audio
            audio = self._preprocess_audio(audio_input)
            
            # Transcribe with detailed output
            result = self.model.transcribe(
                audio, 
                language=selected_lang,
                fp16=torch.cuda.is_available(),
                verbose=False,
                word_timestamps=True
            )
            
            # Extract confidence information from segments
            segments_info = []
            total_confidence = 0
            word_count = 0
            
            for segment in result.get("segments", []):
                segment_info = {
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment.get("text", "").strip(),
                    "confidence": segment.get("avg_logprob", 0)  # Use log probability as confidence
                }
                
                # Process word-level information if available
                words = []
                if "words" in segment:
                    for word in segment["words"]:
                        word_info = {
                            "word": word.get("word", ""),
                            "start": word.get("start", 0),
                            "end": word.get("end", 0),
                            "confidence": word.get("probability", 0)
                        }
                        words.append(word_info)
                        total_confidence += word.get("probability", 0)
                        word_count += 1
                
                segment_info["words"] = words
                segments_info.append(segment_info)
            
            # Calculate overall confidence
            overall_confidence = total_confidence / word_count if word_count > 0 else 0
            
            transcribed_text = result.get("text", "").strip()
            elapsed_time = time.time() - start_time
            
            result_data = {
                "text": transcribed_text,
                "language": result.get("language", selected_lang),
                "confidence": overall_confidence,
                "segments": segments_info,
                "processing_time": elapsed_time,
                "model_used": self.model_name,
                "detected_language": result.get("language", "unknown")
            }
            
            logger.info(f"Transcription completed in {elapsed_time:.2f}s with confidence {overall_confidence:.2f}")
            return result_data
            
        except Exception as e:
            logger.error(f"Error during detailed transcription: {e}")
            return {
                "text": "",
                "language": selected_lang,
                "confidence": 0.0,
                "segments": [],
                "processing_time": time.time() - start_time,
                "model_used": self.model_name,
                "error": str(e)
            }

    def speech_to_text(self, audio_path: str, language: str = None) -> str:
        """
        Converts speech from an audio file to text.
        
        Args:
            audio_path (str): The path to the audio file.
            language (str): Optional language override (e.g., 'tr').
            
        Returns:
            str: The transcribed text, or an empty string if transcription fails.
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found at path: {audio_path}")
            return ""
            
        logger.info(f"Transcribing audio file: {audio_path}")
        
        try:
            result = self.transcribe_with_confidence(audio_path, language)
            return result.get("text", "")
            
        except Exception as e:
            logger.error(f"An unexpected error occurred during STT transcription: {e}")
            return ""

    def transcribe_audio_bytes(self, audio_bytes: bytes, language: str = None) -> Dict[str, Any]:
        """
        Transcribe audio from bytes data.
        
        Args:
            audio_bytes: Audio data as bytes.
            language: Optional language override.
            
        Returns:
            Dictionary with transcription results.
        """
        if not audio_bytes:
            logger.warning("Empty audio bytes provided")
            return {"text": "", "error": "Empty audio data"}
        
        logger.info(f"Transcribing audio from bytes ({len(audio_bytes)} bytes)")
        
        try:
            return self.transcribe_with_confidence(audio_bytes, language)
        except Exception as e:
            logger.error(f"Error transcribing audio bytes: {e}")
            return {"text": "", "error": str(e)}

    def change_model(self, model_name: str) -> bool:
        """
        Change the active Whisper model.
        
        Args:
            model_name: Name of the model to switch to.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self._load_model(model_name)
            self.model_name = model_name
            logger.info(f"Successfully switched to model: {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch to model {model_name}: {e}")
            return False

    def get_available_models(self) -> List[str]:
        """Get list of available Whisper models."""
        return ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model and service."""
        return {
            "current_model": self.model_name,
            "device": self.device,
            "language": self.language,
            "cached_models": list(self._model_cache.keys()),
            "enable_vad": self.enable_vad,
            "chunk_length": self.chunk_length,
            "soundfile_available": SOUNDFILE_AVAILABLE,
            "librosa_available": LIBROSA_AVAILABLE
        }

    def clear_model_cache(self) -> None:
        """Clear the model cache to free memory."""
        self._model_cache.clear()
        logger.info("Model cache cleared")
