# Çağrı Merkezi Chatbotu

Yapay zekâ destekli bu çağrı merkezi asistanı, konuşma bağlamına göre fonksiyon çağırma özelliğiyle müşteri hizmeti operasyonlarını otomatikleştirir. Hesap yönetimi, faturalandırma, paket değişiklikleri ve teknik destek gibi işlemleri akıllı şekilde yerine getirir. Bu proje Türkçe Doğal Dil İşleme Yarışması 2025 Senaryo Kategorisi için tasarlanmıştır.

## Özellikler

- **Akıllı Fonksiyon Çağırma**: Sohbet bağlamına göre müşteri hizmeti fonksiyonlarını otomatik çalıştırır
- **Müşteri Hesap Yönetimi**: Müşteri bilgileri, fatura durumu ve kullanım verilerine erişim
- **Paket Yönetimi**: Mevcut paketleri listeleme, detayları gösterme ve paket değişikliği başlatma
- **Ödeme İşlemleri**: Ödeme işlemleri ve fatura sorguları
- **Destek Kaydı Oluşturma**: Karmaşık sorunlar için destek talebi açma
- **LLM Entegrasyonu**: Doğal diyalog için açık kaynak dil modelleri
- **Web Arayüzü**: Streamlit tabanlı etkileşimli sohbet arayüzü
- **FastAPI Backend**: İstekleri işleyen ve fonksiyonları yöneten sağlam API
- **Kapsamlı Test Sistemi**: Fonksiyon çağırma doğruluğu ve performans ölçümü için KPI test sistemi

## Proje Yapısı

```
.
├── src/
│   ├── functions/           # Çağrı merkezi fonksiyon implementasyonları
│   │   ├── call_center_functions.py  # Örnek müşteri hizmetleri fonksiyonları
│   │   └── function_caller.py        # Fonksiyon ayrıştırma ve yürütme
│   ├── models/              # LLM entegrasyonu
│   │   ├── call_center_llm.py        # Çağrı merkezine özel LLM yöneticisi
│   ├── api/                 # FastAPI backend
│   │   ├── call_center_api.py        # API uç noktaları
│   ├── translator/                 # Çeviri modelleri
│   │   ├── translator.py        # Çeviri modelleri ve fonksiyonları
│   └── stt.py/               # Konuşmayı metne çevirme
│   └── tts.py/               # Metini konuşmaya çevirme
│   └── utils/    
│       └── config.py
├── webapp/                  # Streamlit web uygulamaları
│   ├── call_center_app.py   # Çağrı merkezi sohbet arayüzü
├── run.py                   # Projeyi kolay başlatmak için
├── requirements.txt         # Bağımlılıklar
└── README.md                # Proje dokümantasyonu
```

## Kullanılabilir Mock Fonksiyonlar

1. **get_customer_info(customer_id)** - Müşteri hesap bilgilerini getirir
2. **get_available_packages()** - Tüm mevcut servis paketlerini gösterir
3. **get_package_details(package_name)** - Belirli bir pakete dair detayları getirir
4. **initiate_package_change(customer_id, new_package, effective_date)** - Müşterinin paketini değiştirir
5. **check_billing_status(customer_id)** - Faturalandırma ve ödeme bilgilerini kontrol eder
6. **process_payment(customer_id, amount, payment_method)** - Ödeme işlemi yapar
7. **get_usage_summary(customer_id, period)** - Müşteri kullanım istatistiklerini getirir
8. **create_support_ticket(customer_id, issue_type, description, priority)** - Destek kaydı oluşturur

## Ortam Değişkenleri

```bash
# API Ayarları
API_HOST=0.0.0.0 # Ngrok entegrasyonunda public domain
API_PORT=8000

# Web Uygulaması Ayarları
WEBAPP_HOST=0.0.0.0
WEBAPP_PORT=8501

# LLM Model Ayarları
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.3

# Hugging Face Ayarları
HUGGINGFACE_TOKEN=your_huggingface_token_here

# NGROK Ayarları (opsiyonel, genel erişim için)
NGROK_AUTHTOKEN=your_ngrok_token_here
API_USAGE=True # İzin varsa tam bir bilgi alamadık ondan ikisi de çalışıyor ama API usage daha iyi bir test ortamı sağladığından onu kullanıyoruz
GOOGLE_GENAI_API_KEY=your_genai_api_key # Kullanması ücretsiz open source modeli çalıştırmak için kullandık
```

## Kurulum (Yerel)

1. Depoyu klonlayın:
```bash
git clone [repository-url]
cd call-center-chatbot
```

2. Sanal ortam oluşturun ve aktifleştirin:
```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
venv\Scripts\activate
```

3. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

4. Ortam değişkenlerini ayarlayın (yukarıdaki değişkenlerle bir `.env` dosyası oluşturun)

5. API’yi başlatın:
```bash
python -m src.api.call_center_api
```

6. Streamlit arayüzünü açın:
```bash
streamlit run webapp/call_center_app.py
```

## Kullanım

### Web Arayüzü

1. http://localhost:8501 adresini açın
2. Kişiselleştirme için bir müşteri ID’si girin (örn. `customer_001`)
3. Hızlı işlem butonlarını kullanın veya doğal dilde talebinizi yazın
4. Asistan talebinize göre uygun fonksiyonları otomatik çağıracaktır

### Örnek Etkileşimler

- **"customer_001 için hesap bilgilerimi göster"** → `get_customer_info()` çağrılır
- **"Hangi paketler mevcut?"** → `get_available_packages()` çağrılır
- **"Premium pakete geçmek istiyorum"** → `initiate_package_change()` çağrılır
- **"Güncel fatura durumum nedir?"** → `check_billing_status()` çağrılır

### Test Müşteri Verisi

- **customer_001** (John Doe): Premium, aktif hesap, borç yok
- **customer_002** (Jane Smith): Basic, aktif hesap, $25.99 borç
- **customer_003** (Bob Johnson): Standard, askıya alınmış hesap, $75.99 borç

## API Uç Noktaları

- **Health Check**: `GET /health` - Sistem durumunu kontrol eder
- **Chat**: `POST /chat` - Sohbete mesaj gönderir
- **Functions**: `GET /functions` - Uygun fonksiyonları listeler
- **Doğrudan Fonksiyon Çağrısı**: `POST /function/call` - Fonksiyonu direkt çalıştırır
- **API Dokümantasyonu**: `/docs` - Etkileşimli API dokümanı

## Colab Kurulumu

1. Projeyi Colab’e yükleyin
2. Bağımlılıkları yükleyin: `!pip install -r requirements.txt`
3. Ortam değişkenlerini ayarlayın:
```python
import os
os.environ["HUGGINGFACE_TOKEN"] = "your_token"
os.environ["NGROK_AUTHTOKEN"] = "your_ngrok_token"  # opsiyonel
```
4. API’yi çalıştırın: `!python -m src.api.call_center_api`

## Özelleştirme

- **Yeni Fonksiyon Ekleme**: `src/functions/call_center_functions.py`
- **Müşteri Verisi**: `MOCK_CUSTOMERS` sözlüğünü güncelleyin
- **LLM Modeli**: `LLM_MODEL` ortam değişkenini değiştirin
- **UI Özelleştirme**: `webapp/call_center_app.py`
- **Fonksiyon Ayrıştırma**: `src/functions/function_caller.py`

## Test ve Performans Değerlendirmesi

### KPI Test Sistemi

Proje, LLM'nin fonksiyon çağırma performansını ölçmek için kapsamlı bir test sistemi içerir. Test sistemi 111 adet Türkçe soru kullanarak 9 farklı kategoriyi değerlendirir:

```bash
# Tüm testleri çalıştırmak için:
python test_call_center_kpis.py

# Veya parametreli çalıştırma:
python run_kpi_tests.py --use-api --sample-size 5

# Google API anahtarını ayarlamayı unutmayın:
export GOOGLE_GENAI_API_KEY="your_api_key_here"

# Türkçe locale ayarları (opsiyonel ama önerilen):
export LANG=tr_TR.UTF-8
export LC_ALL=tr_TR.UTF-8
```

### Test Kategorileri ve Soru Sayıları

- **get_customer_info**: 13 soru (müşteri bilgileri)
- **get_available_packages**: 12 soru (paket listesi)
- **initiate_package_change**: 13 soru (paket değişikliği)
- **check_billing_status**: 13 soru (fatura durumu)
- **process_payment**: 11 soru (ödeme işlemleri)
- **create_support_ticket**: 18 soru (destek talepleri)
- **get_usage_summary**: 12 soru (kullanım özeti)
- **get_package_details**: 12 soru (paket detayları)
- **don't_know**: 7 soru (fonksiyon çağrılmaması gereken durumlar)

### Mevcut Performans Metrikleri (Gemma-3-27B-IT)

Son test sonuçlarına göre:
- **Toplam Test**: 111 soru
- **Fonksiyon Tespit Doğruluğu**: %76.58
- **Parametre Çıkarma Doğruluğu**: %87.06
- **Genel Başarı Oranı**: %76.58
- **Ortalama Yanıt Süresi**: 3.60 saniye
- **Medyan Yanıt Süresi**: 3.89 saniye

### Fonksiyon Bazında Performans:
- **process_payment**: %100 başarı (ödeme işlemleri)
- **get_usage_summary**: %100 başarı (kullanım özeti)
- **don't_know**: %100 başarı (gereksiz fonksiyon çağrısı engelleme)
- **get_customer_info**: %92.31 başarı (müşteri bilgileri)
- **initiate_package_change**: %84.62 başarı (paket değişikliği)
- **check_billing_status**: %84.62 başarı (fatura durumu)
- **create_support_ticket**: %72.22 başarı (destek talepleri)
- **get_available_packages**: %50.00 başarı (paket listesi)
- **get_package_details**: %16.67 başarı (paket detayları)*

*Not: Bazı fonksiyonlarda düşük performans, prompt optimizasyonu ve fine-tuning ile iyileştirilebilir.

### Dil İşleme Uyarısı

⚠️ **Kritik**: Sistem tamamen Türkçe sorularla eğitilmiş ve test edilmiştir. **Eğer LLM yanıtları İngilizce geliyorsa, dil işleme modülünde sapma meydana gelmiş demektir.** Bu durum sistem performansını ciddi şekilde etkiler.

**Dil sapması tespit edildiğinde yapılacaklar:**
1. Şu anki sürümde kısıtlı zaman nedeniyle dil tespiti yapamadık, open source dil çeviri modelini kullanarak sonucu sağlıyoruz
2. Sonraki geliştirmelerimiz ile dil tespiti sağlayıp buna göre aksiyon sağlama (regenerate, dil çevirisini sağlama )

### Test Raporları

Test sonuçları otomatik olarak şu dosyalarda saklanır:
- `test_results/kpi_report_api_[timestamp].json` - Detaylı JSON raporu
- `test_results/kpi_results_api_[timestamp].xlsx` - Excel analiz dosyası
- `test_results/kpi_summary_api_[timestamp].txt` - Özet metin raporu

## Ekran Görüntüleri

- `ScreenShots/call-center-1.png`
- `ScreenShots/call-center-2.png`
- `ScreenShots/call-center-3.png`

---

## İyileştirmeler ve Yol Haritası

- Ürün ve Deneyim (UX)
  - Sık işlemler için yönlendirmeli akışlar (fatura itirazı, paket yükseltme, arıza bildirimi)
  - Sohbet içinde doğrulamalı, argüman toplayan mini formlar
  - Çok turlu slot doldurma ve netleştirme soruları
  - Sohbet özetleri ve takip mesajları
- Ses ve Gerçek Zamanlılık
  - Düşük gecikmeli WebRTC tabanlı ses akışı
  - SSML desteği, vurgu kontrolü, çoklu ses kütüphanesi
  - VAD (konuşma etkinliği algılama) ve konuşmacı ayrımı
- LLM ve Fonksiyon Çağırma
  - Araç (tool) seçimi akıl yürütme ve otomatik yeniden deneme (retry)
  - Fonksiyon sonuçlarının doğrulanması ve kendi kendini onarma
  - Kullanılan modellerin optimizasyonu için senaryo verilerimiz ile modellere ince ayar yapmak
- Bilgi ve RAG implementasyonu
  - Kurumsal bilgi tabanı (SSS, politikalar, ürün dokümanları) entegrasyonu
  - Semantik filtreleme
  - Yanıtlarda atıf ve kaynak bağlama
- Kalite ve Değerlendirme
  - Niyet başına sentetik test setleri; referans sohbetler
  - Araç çağırma doğruluğu için otomatik regresyon testleri
  - Kullanıcı geri bildirimi (like/dislike) ve hata etiketleme
- Gözlemlenebilirlik
  - İzleme (OpenTelemetry), gecikme/hata panelleri
  - Prompt kayıtları, fonksiyon çağrı metrikleri, TTS/STT süreleri
- Performans ve Maliyet
  - Niyet/uzunluğa göre model seçim politikası
  - Yanıt önbelleği ve fonksiyon çağrısı belleklemesi
  - Uygun yerlerde mixed-precision ve batching
- Çok Dilli
  - Türkçe’ye özel ayarlı LLM ve akustik modeller
  - Otomatik dil algılama ve kullanıcı üstüne yazma
  - Yerelleştirme
  - Senaryo verileri ile modellere ince ayar yaparak LLM verimini arttırma
- Güvenlik ve Uyum
  - Oran sınırlama (rate limiting) ve kötüye kullanım önleme
- Dağıtım
  - API ve WebApp için Docker imajları; CI/CD boru hattı
  - Konfigürasyon profilleri (dev/stage/prod)
  - Sağlık kontrolleri ve otomatik ölçekleme politikaları
- Test
  - Araçlar ve uç noktalar için birim/entegrasyon testleri
  - Streamlit akışları için headless tarayıcı ile Uçtan Uca (E2E) testler
  - Yoğun saatler için yük testleri

---
## Lisans

Bu proje, [Apache 2.0](./LICENSE) lisansı altında lisanslanmıştır.

Açıkça başka bir şekilde belirtmedikçe, sizin tarafınızdan bu projeye dahil edilmek üzere kasıtlı olarak gönderilen her katkı, Apache-2.0 lisansında tanımlandığı şekilde yukarıdaki lisans ile, herhangi bir ek hüküm veya koşul olmaksızın lisanslanacaktır.
