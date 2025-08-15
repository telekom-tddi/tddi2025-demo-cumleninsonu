import wave
import os
from piper import PiperVoice

print("Minimalist Piper Testi Başlatılıyor...")

# --- AYARLAR ---
# Yeni indirdiğin 'fettah' modelinin yolunu buraya yaz
model_path = 'en_US-lessac-medium.onnx' 

# Çıktı dosyasının adı
output_path = 'minimal_test_output.wav'

# Seslendirilecek basit bir metin
text_to_speak = "Bu, fettah modeli ile yapılan bir testtir."
# ---------------

# 1. Model dosyasının var olup olmadığını kontrol et
if not os.path.exists(model_path):
    print(f"HATA: Model dosyası bulunamadı: {model_path}")
    print("Lütfen modelin doğru klasörde ('piper_models/') olduğundan emin ol.")
else:
    try:
        # 2. Modeli yükle
        print(f"Model yükleniyor: {model_path}")
        voice = PiperVoice.load(model_path)
        print("Model başarıyla yüklendi.")

        # 3. Sesi üret ve dosyaya kaydet
        print(f"Ses üretiliyor ve '{output_path}' dosyasına kaydediliyor...")
        with wave.open(output_path, "wb") as wav_file:
            voice.synthesize_wav(text_to_speak, wav_file)
        
        print(f"\nBaşarılı! Lütfen '{output_path}' dosyasını dinleyin ve ses olup olmadığını kontrol edin.")

    except Exception as e:
        print(f"\nBİR HATA OLUŞTU: {e}")