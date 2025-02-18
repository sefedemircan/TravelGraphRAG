# Seyahat ve Konaklama Bilgi Grafiği RAG Sistemi

Seyahat ve konaklama verileri üzerinde doğal dil sorguları yapabilen, Neo4j graf veritabanı ve LangChain tabanlı bir soru-cevap sistemi.

## Proje Açıklaması

Bu sistem, kişiler, oteller, şehirler ve seyahat bilgileri arasındaki ilişkileri graf veritabanında modelleyerek, kullanıcıların doğal dil ile sorgular yapmasına olanak sağlar. Proje, şu temel bileşenleri içerir:

- **Neo4j Graf Veritabanı**: Seyahat ve konaklama verilerini ve ilişkilerini saklar
- **LangChain**: Doğal dil işleme ve soru-cevap zincirleri için kullanılır
- **OpenAI GPT-4**: Doğal dil sorgularını Cypher sorgularına çevirmek için kullanılır
- **HuggingFace Embeddings**: Metin temsilleri için kullanılır
- **Gradio**: Kullanıcı arayüzü için kullanılır

### Veri Modeli

Sistem aşağıdaki veri tiplerini ve ilişkilerini içerir:

**Düğümler (Nodes):**
- **Person**: id, name, age, gender, email, phone
- **Hotel**: name
- **City**: name
- **Airline**: name
- **BusCompany**: name

**İlişkiler (Relationships):**
- **KONAKLADI**: Person -> Hotel (check_in, duration, room_type, daily_rate, breakfast_included, all_inclusive)
- **UCAKLA_SEYAHAT_ETTI**: Person -> City (airline, flight_date, class, fare, flight_type, baggage, duration, from, to)
- **OTOBUSLE_SEYAHAT_ETTI**: Person -> City (company, travel_date, seat_no, fare, bus_type, duration, from, to)
- **LOCATED_IN**: Hotel -> City

## Kurulum

1. Gerekli Python paketlerini yükleyin:
```bash
pip install -r requirements.txt
```

2. `.env` dosyasında Neo4j ve OpenAI API bilgilerinizi ayarlayın:
```env
OPENAI_API_KEY=your-openai-api-key
NEO4J_URI=your-neo4j-uri
NEO4J_USERNAME=your-username
NEO4J_PASSWORD=your-password
```

## Kullanım

1. Sistemi başlatın:
```python
python app.py
```

2. Tarayıcınızda açılan Gradio arayüzünü kullanarak doğal dil ile sorgular yapın. Örnek sorgular:
- "Ahmet Yılmaz ve Ayşe Demir arasındaki ilişkileri göster"
- "İstanbul'da konaklayan kişileri listele"
- "Ankara'ya uçakla seyahat eden yolcuları göster"
- "Ali Kaya'nın tüm seyahat ve konaklama bilgilerini getir"

## Özellikler

- **Doğal Dil İşleme**: Kullanıcı sorguları otomatik olarak Cypher sorgularına çevrilir
- **Graf Tabanlı Sorgulama**: İlişkisel veriler üzerinde karmaşık sorgular yapabilme
- **Web Arayüzü**: Gradio ile kullanıcı dostu web arayüzü
- **Detaylı Yanıtlar**: GPT-4 ile zenginleştirilmiş, anlamlı yanıtlar

## Teknik Detaylar

### Veri Yükleme

Sistem CSV dosyalarından veri yükler:
- `kisiler.csv`: Kişi bilgileri
- `otel_konaklamalari.csv`: Otel konaklamaları
- `ucak_seyahatleri.csv`: Uçak seyahatleri
- `otobus_seyahatleri.csv`: Otobüs seyahatleri
