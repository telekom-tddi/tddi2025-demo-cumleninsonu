#
# Text-to-Speech (TTS) Service
# Bu servis, metni sese dönüştürmek için farklı motorları kullanır.
# Ana motor olarak hızlı ve çevrimdışı çalışan Piper TTS (v1.2.0) hedeflenmiştir.
# Yedek olarak gTTS (online) ve pyttsx3 (düşük kaliteli offline) motorları bulunur.
#

import logging
import io
import hashlib
import os
import threading
import wave  # Piper'ın 1.2.0 sürümü için wave kütüphanesi gerekiyor
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod
from src.utils.config import BASE_DIR  # Proje dizinini almak için BASEDIR kullanıldı

import tempfile  # Geçici dosyalar oluşturmak için eklendi

# Gerekli kütüphaneleri import etme
try:
    from gtts import gTTS, gTTSError
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    OFFLINE_TTS_AVAILABLE = True
except ImportError:
    OFFLINE_TTS_AVAILABLE = False
    pyttsx3 = None

try:
    from piper import PiperVoice
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False

# Standart loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- TTSEngine Arayüzü (Tüm motorlar için genel şablon) ---
class TTSEngine(ABC):
    """Tüm TTS motorları için temel arayüz (soyut sınıf)."""
    @abstractmethod
    def synthesize(self, text: str, language: str) -> Optional[bytes]:
        """Metni sese dönüştürür ve byte olarak döndürür."""
        pass

# --- ANA MOTORUMUZ: Piper TTS (DOKÜMANTASYONA UYGUN HALİ) ---
class PiperEngine(TTSEngine):
    """Piper TTS v1.2.0 motoru (Dokümantasyona uygun, stabil çalışan versiyon)."""
    def __init__(self, model_path: str = f'{BASE_DIR}/piper_models/tr_TR-fahrettin-medium.onnx'):
        if not PIPER_AVAILABLE:
            raise ImportError("piper-tts kütüphanesi kurulu değil.")
        
        self.voice = None
        if not os.path.exists(model_path):
            logger.error(f"Piper modeli bulunamadı: {model_path}")
            return

        try:
            logger.info(f"Piper TTS v1.2.0 modeli yükleniyor: {model_path}")
            # Dokümantasyona uygun olarak .load() metodu kullanılıyor
            self.voice = PiperVoice.load(model_path) #
            logger.info("Piper TTS v1.2.0 modeli başarıyla yüklendi.")
        except Exception as e:
            logger.error(f"Piper TTS motoru başlatılamadı: {e}")

    def synthesize(self, text: str, language: str) -> Optional[bytes]:
        if not self.voice:
            logger.error("Piper TTS motoru hazır değil veya yüklenemedi.")
            return None
        
        try:
            # Sesi önce diske geçici bir dosyaya yazıp sonra okuyacağız.
            # Bu, dokümantasyondaki yöntemdir ve sessiz dosya sorununu çözer.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_wav_file:
                file_path = tmp_wav_file.name
                
                # Dokümanda gösterildiği gibi, wave.open ile dosyayı açıp sentezletiyoruz
                with wave.open(file_path, "wb") as wav_file:
                    self.voice.synthesize_wav(text, wav_file) #
                
                # Sentezleme bittikten sonra, geçici dosyanın içeriğini okuyoruz
                tmp_wav_file.seek(0)
                audio_bytes = tmp_wav_file.read()

            return audio_bytes
        except Exception as e:
            logger.error(f"Piper ile ses üretimi sırasında hata oluştu: {e}")
            return None
                

# --- YEDEK MOTORLAR ---
class GTTSEngine(TTSEngine):
    """Google Text-to-Speech (gTTS) motoru. İnternet bağlantısı gerektirir."""
    def synthesize(self, text: str, language: str) -> Optional[bytes]:
        if not GTTS_AVAILABLE:
            raise ImportError("gTTS kütüphanesi yüklü değil.")
        try:
            audio_fp = io.BytesIO()
            tts = gTTS(text=text, lang=language, tld='com', slow=False)
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            return audio_fp.read() or None
        except Exception as e:
            logger.error(f"gTTS hatası: {e}")
            return None

class Pyttsx3Engine(TTSEngine):
    """pyttsx3 motoru. En son çare, düşük kaliteli çevrimdışı yedek."""
    def __init__(self):
        if not OFFLINE_TTS_AVAILABLE:
            raise ImportError("pyttsx3 kütüphanesi yüklü değil.")
        self.engine = pyttsx3.init()
        self.cache_dir = "." # Geçici dosya için bir dizin

    def synthesize(self, text: str, language: str) -> Optional[bytes]:
        try:
            temp_wav_path = os.path.join(self.cache_dir, f'temp_offline_{os.getpid()}.wav')
            self.engine.save_to_file(text, temp_wav_path)
            self.engine.runAndWait()
            if os.path.exists(temp_wav_path):
                with open(temp_wav_path, 'rb') as f:
                    audio_data = f.read()
                os.remove(temp_wav_path)
                return audio_data
            return None
        except Exception as e:
            logger.error(f"pyttsx3 hatası: {e}")
            return None


# --- Ana TTS Servisi ---
class TextToSpeechService:
    """Farklı TTS motorlarını öncelik sırasına göre kullanan servis."""
    def __init__(self, language: str = 'tr', cache_dir: str = "tts_cache"):
        self.language = language
        self.cache_dir = cache_dir
        self._cache: Dict[str, bytes] = {}
        self._lock = threading.Lock()
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        self.engines: List[TTSEngine] = self._initialize_engines()
        logger.info(f"Kullanılabilir motorlar: {[e.__class__.__name__ for e in self.engines]}")

    def _initialize_engines(self) -> List[TTSEngine]:
        """Motorları öncelik sırasına göre başlatır."""
        engine_list = []
        
        # Öncelik sırası: Piper (hızlı, yerel) -> gTTS (online yedek) -> pyttsx3 (düşük kaliteli yedek)
        if PIPER_AVAILABLE:
            try:
                engine_list.append(PiperEngine())
            except Exception as e:
                logger.warning(f"PiperEngine başlatılamadı, atlanıyor: {e}")
        
        if GTTS_AVAILABLE:
            engine_list.append(GTTSEngine())
        
        if OFFLINE_TTS_AVAILABLE:
            try:
                pyttsx3_engine = Pyttsx3Engine()
                pyttsx3_engine.cache_dir = self.cache_dir
                engine_list.append(pyttsx3_engine)
            except Exception as e:
                logger.warning(f"Pyttsx3Engine başlatılamadı, atlanıyor: {e}")

        if not engine_list:
            logger.critical("Hiçbir TTS motoru yüklenemedi! Servis çalışmayacak.")
            
        return engine_list
    
    def _generate_cache_key(self, text: str, language: str) -> str:
        """Verilen metin için bir MD5 hash cache anahtarı oluşturur."""
        content = f"{text}_{language}"
        return hashlib.md5(content.encode()).hexdigest()

    def text_to_speech(self, text: str, language: str | None = None) -> Optional[bytes]:
        """Metni sese çevirir, cache kontrolü yapar ve motorları sırayla dener."""
        selected_lang = language or self.language
        cache_key = self._generate_cache_key(text, selected_lang)
        
        cached_audio = self._get_cached_audio(cache_key)
        if cached_audio:
            logger.info(f"'{text[:30]}...' için cache'den bulundu.")
            return cached_audio

        with self._lock:
            cached_audio = self._get_cached_audio(cache_key)
            if cached_audio:
                return cached_audio

            for engine in self.engines:
                print(f"'{engine.__class__.__name__}' ile ses üretimi deneniyor...")
                logger.info(f"'{engine.__class__.__name__}' ile ses üretimi deneniyor...")
                try:
                    audio_data = engine.synthesize(text, selected_lang)
                    if audio_data:
                        logger.info(f"'{engine.__class__.__name__}' başarılı oldu.")
                        self._save_to_cache(cache_key, audio_data)
                        return audio_data
                except Exception as e:
                    logger.error(f"Motor '{engine.__class__.__name__}' çalışırken hata verdi: {e}")
            
            logger.error("Tüm TTS motorları başarısız oldu.")
            return None
    
    def _get_cached_audio(self, cache_key: str) -> Optional[bytes]:
        """Bellek ve disk cache'inden ses verisini okur."""
        if cache_key in self._cache:
            return self._cache[cache_key]
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.wav")
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                audio_data = f.read()
            self._cache[cache_key] = audio_data
            return audio_data
        return None
    
    def _save_to_cache(self, cache_key: str, audio_data: bytes) -> None:
        """Ses verisini bellek ve disk cache'ine yazar."""
        self._cache[cache_key] = audio_data
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.wav")
        with open(cache_file, 'wb') as f:
            f.write(audio_data)

# --- Test Bloğu ---
# Bu dosya doğrudan çalıştırıldığında aşağıdaki kod çalışır.
if __name__ == "__main__":
    print("TTS Servisi test ediliyor...")
    
    # Servisi başlat
    tts_service = TextToSpeechService()
    
    # Servisin motorları doğru yükleyip yüklemediğini kontrol et
    if not tts_service.engines:
        print("Hiçbir motor yüklenemedi, test başarısız.")
    else:
        metin = "Merhaba dünya, ben Piper'ın stabil sürümüyüm. Artık sorunsuz çalışıyorum."
        print(f"Seslendirilecek metin: '{metin}'")
        
        # Metni sese çevir
        ses_verisi = tts_service.text_to_speech(metin)
        
        if ses_verisi:
            with open("test_ciktisi.wav", "wb") as f:
                f.write(ses_verisi)
            print("\nBaşarılı! Ses dosyası 'test_ciktisi.wav' olarak kaydedildi.")
            print("Lütfen dosyayı dinleyerek kaliteyi kontrol et.")
        else:
            print("\nBaşarısız! Ses üretilemedi. Logları kontrol et.")