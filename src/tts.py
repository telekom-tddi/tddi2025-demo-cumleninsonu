"""
Text-to-Speech (TTS) Service
This service handles text-to-speech conversion using gTTS with caching and improved error handling.
"""

import logging
import io
import hashlib
import os
from typing import Optional, Dict, Any
from gtts import gTTS, gTTSError

try:
    import pyttsx3
    OFFLINE_TTS_AVAILABLE = True
except ImportError:
    OFFLINE_TTS_AVAILABLE = False
    pyttsx3 = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextToSpeechService:
    """Service for converting text to speech with caching and offline fallback."""
    
    def __init__(self, language: str = 'tr', tld: str = 'com', cache_dir: str = "tts_cache", use_offline_fallback: bool = True):
        """
        Initialize the TTS service.
        
        Args:
            language (str): The language of the text.
            tld (str): The top-level domain for the Google Translate host.
            cache_dir (str): Directory to store cached audio files.
            use_offline_fallback (bool): Whether to use offline TTS as fallback.
        """
        self.language = language
        self.tld = tld
        self.cache_dir = cache_dir
        self.use_offline_fallback = use_offline_fallback
        self._cache: Dict[str, bytes] = {}
        
        # Create cache directory
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        # Initialize offline TTS if available and enabled
        self.offline_engine = None
        if self.use_offline_fallback and OFFLINE_TTS_AVAILABLE:
            try:
                self.offline_engine = pyttsx3.init()
                # Set voice properties
                voices = self.offline_engine.getProperty('voices')
                if voices:
                    # Try to find a voice that matches the language
                    for voice in voices:
                        if language in voice.id.lower() or language in voice.name.lower():
                            self.offline_engine.setProperty('voice', voice.id)
                            break
                self.offline_engine.setProperty('rate', 150)  # Speaking rate
                self.offline_engine.setProperty('volume', 0.8)  # Volume level
                logger.info("Offline TTS engine initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize offline TTS engine: {e}")
                self.offline_engine = None
        
        logger.info(f"TTS service initialized for language: {language}, offline fallback: {self.offline_engine is not None}")

    def _generate_cache_key(self, text: str, language: str) -> str:
        """Generate a cache key for the given text and language."""
        content = f"{text}_{language}_{self.tld}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cached_audio(self, cache_key: str) -> Optional[bytes]:
        """Get cached audio data."""
        # Check memory cache first
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        # Check file cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.mp3")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    audio_data = f.read()
                self._cache[cache_key] = audio_data  # Add to memory cache
                return audio_data
            except Exception as e:
                logger.warning(f"Failed to read cached audio file {cache_file}: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, audio_data: bytes) -> None:
        """Save audio data to cache."""
        # Save to memory cache
        self._cache[cache_key] = audio_data
        
        # Save to file cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.mp3")
        try:
            with open(cache_file, 'wb') as f:
                f.write(audio_data)
        except Exception as e:
            logger.warning(f"Failed to save audio to cache file {cache_file}: {e}")
    
    def _offline_tts(self, text: str, language: str) -> Optional[bytes]:
        """Generate TTS using offline engine as fallback."""
        if not self.offline_engine:
            return None
            
        try:
            # Generate a temporary file for offline TTS
            temp_file = os.path.join(self.cache_dir, f"temp_offline_{os.getpid()}.wav")
            
            # Save to file using pyttsx3
            self.offline_engine.save_to_file(text, temp_file)
            self.offline_engine.runAndWait()
            
            # Read the generated file
            if os.path.exists(temp_file):
                with open(temp_file, 'rb') as f:
                    audio_data = f.read()
                
                # Clean up temp file
                os.remove(temp_file)
                
                logger.info(f"Successfully generated offline TTS audio: {len(audio_data)} bytes")
                return audio_data
            else:
                logger.error("Offline TTS failed to generate audio file")
                return None
                
        except Exception as e:
            logger.error(f"Offline TTS error: {e}")
            return None

    def text_to_speech(self, text: str, language: str | None = None) -> Optional[bytes]:
        """
        Converts text to speech and returns the audio data as bytes.
        
        Args:
            text (str): The text to convert.
            language (str | None): Optional override for language code (e.g., 'en', 'tr').
            
        Returns:
            bytes: The audio data in MP3 format, or None if conversion fails.
        """
        if not text or not text.strip():
            logger.warning("TTS conversion requested for empty text.")
            return None
            
        selected_lang = language or self.language
        text = text.strip()
        
        # Generate cache key
        cache_key = self._generate_cache_key(text, selected_lang)
        
        # Check cache first
        cached_audio = self._get_cached_audio(cache_key)
        if cached_audio:
            logger.info(f"Retrieved cached audio for text: '{text[:50]}...'")
            return cached_audio
        
        logger.info(f"Converting text to speech (lang={selected_lang}): '{text[:50]}...'")
        
        # Try gTTS first
        try:
            # Create an in-memory binary stream
            audio_fp = io.BytesIO()
            
            # Generate the speech using gTTS
            tts = gTTS(text=text, lang=selected_lang, tld=self.tld, slow=False)
            tts.write_to_fp(audio_fp)
            
            # Go to the beginning of the stream
            audio_fp.seek(0)
            
            # Read the audio data
            audio_bytes = audio_fp.read()
            
            if audio_bytes:
                # Save to cache
                self._save_to_cache(cache_key, audio_bytes)
                logger.info(f"Successfully converted text to {len(audio_bytes)} bytes of audio data.")
                return audio_bytes
            else:
                logger.error("gTTS returned empty audio data")
                
        except gTTSError as e:
            logger.error(f"gTTS error during conversion: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during gTTS conversion: {e}")
        
        # Try offline TTS as fallback
        if self.use_offline_fallback and self.offline_engine:
            logger.info("Attempting offline TTS fallback...")
            offline_audio = self._offline_tts(text, selected_lang)
            if offline_audio:
                # Save to cache
                self._save_to_cache(cache_key, offline_audio)
                return offline_audio
        
        logger.error("All TTS methods failed")
        return None
    
    def clear_cache(self) -> None:
        """Clear the TTS cache."""
        self._cache.clear()
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.mp3'):
                    os.remove(os.path.join(self.cache_dir, file))
            logger.info("TTS cache cleared successfully")
        except Exception as e:
            logger.warning(f"Failed to clear cache directory: {e}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        cache_files = []
        cache_size = 0
        try:
            for file in os.listdir(self.cache_dir):
                if file.endswith('.mp3'):
                    file_path = os.path.join(self.cache_dir, file)
                    file_size = os.path.getsize(file_path)
                    cache_files.append(file)
                    cache_size += file_size
        except Exception as e:
            logger.warning(f"Failed to get cache info: {e}")
        
        return {
            "memory_cache_size": len(self._cache),
            "file_cache_size": len(cache_files),
            "total_cache_size_bytes": cache_size,
            "cache_directory": self.cache_dir,
            "offline_tts_available": self.offline_engine is not None
        }
