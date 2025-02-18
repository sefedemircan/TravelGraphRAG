from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts.prompt import PromptTemplate
import pandas as pd
import gradio as gr

# API anahtarları ve bağlantı bilgileri
OPENAI_API_KEY = "your-openai-api-key"
NEO4J_URI = "your-neo4j-uri"
NEO4J_USERNAME = "your-username"
NEO4J_PASSWORD = "your-password"

# Hugging Face model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Neo4j graph nesnesi oluşturma
graph = Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD
)

def load_csv_data():
    """CSV dosyalarını yükle"""
    persons = pd.read_csv('/content/kisiler.csv')
    hotels = pd.read_csv('/content/otel_konaklamalari.csv')
    flights = pd.read_csv('/content/ucak_seyahatleri.csv')
    buses = pd.read_csv('/content/otobus_seyahatleri.csv')
    return persons, hotels, flights, buses

def create_constraints_and_indexes():
    """Neo4j için gerekli kısıtlamaları ve indeksleri oluştur"""
    constraints = [
        "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
        "CREATE CONSTRAINT hotel_name IF NOT EXISTS FOR (h:Hotel) REQUIRE h.name IS UNIQUE",
        "CREATE CONSTRAINT city_name IF NOT EXISTS FOR (c:City) REQUIRE c.name IS UNIQUE",
        "CREATE CONSTRAINT airline_name IF NOT EXISTS FOR (a:Airline) REQUIRE a.name IS UNIQUE",
        "CREATE CONSTRAINT bus_company_name IF NOT EXISTS FOR (b:BusCompany) REQUIRE b.name IS UNIQUE"
    ]

    for constraint in constraints:
        try:
            graph.query(constraint)
        except Exception as e:
            print(f"Kısıtlama oluşturulurken hata: {str(e)}")

def create_graph_data():
    """CSV verilerini Neo4j'ye aktar"""
    # CSV dosyalarını yükle
    persons, hotels, flights, buses = load_csv_data()

    # Kısıtlamaları ve indeksleri oluştur
    create_constraints_and_indexes()

    # Tüm verileri temizle
    graph.query("MATCH (n) DETACH DELETE n")

    # Kişileri oluştur
    for _, person in persons.iterrows():
        person_query = f"""
        CREATE (p:Person {{
            id: {person.id},
            name: '{person.ad} {person.soyad}',
            age: {person.yaş},
            gender: '{person.cinsiyet}',
            email: '{person.email}',
            phone: '{person.telefon}'
        }})
        """
        graph.query(person_query)

    # Otelleri ve konaklama ilişkilerini oluştur
    for _, hotel in hotels.iterrows():
        hotel_query = f"""
        MERGE (h:Hotel {{name: '{hotel.otel_adı}'}})
        MERGE (c:City {{name: '{hotel.şehir}'}})
        WITH h, c
        MATCH (p:Person {{id: {hotel.müşteri_id}}})
        CREATE (p)-[:KONAKLADI {{
            check_in: '{hotel.giriş_tarihi}',
            duration: {hotel.kalış_süresi},
            room_type: '{hotel.oda_tipi}',
            daily_rate: {hotel.günlük_ücret},
            breakfast_included: {str(hotel.kahvaltı_dahil).lower()},
            all_inclusive: {str(hotel.all_inclusive).lower()}
        }}]->(h)
        CREATE (h)-[:LOCATED_IN]->(c)
        """
        graph.query(hotel_query)

    # Uçuşları ve ilişkileri oluştur
    for _, flight in flights.iterrows():
        flight_query = f"""
        MERGE (a:Airline {{name: '{flight.havayolu}'}})
        MERGE (c1:City {{name: '{flight.kalkış_şehri}'}})
        MERGE (c2:City {{name: '{flight.varış_şehri}'}})
        WITH a, c1, c2
        MATCH (p:Person {{id: {flight.yolcu_id}}})
        CREATE (p)-[:UCAKLA_SEYAHAT_ETTI {{
            airline: '{flight.havayolu}',
            flight_date: '{flight.uçuş_tarihi}',
            class: '{flight.uçuş_sınıfı}',
            fare: {flight.bilet_ücreti},
            flight_type: '{flight.uçuş_tipi}',
            baggage: '{flight.bagaj_hakkı}',
            duration: '{flight.uçuş_süresi}',
            from: '{flight.kalkış_şehri}',
            to: '{flight.varış_şehri}'
        }}]->(c2)
        """
        graph.query(flight_query)

    # Otobüs seyahatlerini ve ilişkileri oluştur
    for _, bus in buses.iterrows():
        bus_query = f"""
        MERGE (bc:BusCompany {{name: '{bus.firma}'}})
        MERGE (c1:City {{name: '{bus.kalkış_şehri}'}})
        MERGE (c2:City {{name: '{bus.varış_şehri}'}})
        WITH bc, c1, c2
        MATCH (p:Person {{id: {bus.yolcu_id}}})
        CREATE (p)-[:OTOBUSLE_SEYAHAT_ETTI {{
            company: '{bus.firma}',
            travel_date: '{bus.seyahat_tarihi}',
            seat_no: '{bus.koltuk_no}',
            fare: {bus.bilet_ücreti},
            bus_type: '{bus.sefer_tipi}',
            duration: '{bus.tahmini_süre}',
            from: '{bus.kalkış_şehri}',
            to: '{bus.varış_şehri}'
        }}]->(c2)
        """
        graph.query(bus_query)

    print("Veriler başarıyla Neo4j'ye aktarıldı!")

# Özel sorgulama şablonu
CYPHER_TEMPLATE = """
Verilen soruyu yanıtlamak için bir Cypher sorgusu oluştur.

Soru: {query}

Neo4j veritabanı şeması:
Düğümler:
- Person: id, name, age, gender, email, phone
- Hotel: name
- City: name
- Airline: name
- BusCompany: name

İlişkiler:
- KONAKLADI: Person -> Hotel (check_in, duration, room_type, daily_rate, breakfast_included, all_inclusive)
- UCAKLA_SEYAHAT_ETTI: Person -> City (airline, flight_date, class, fare, flight_type, baggage, duration, from, to)
- OTOBUSLE_SEYAHAT_ETTI: Person -> City (company, travel_date, seat_no, fare, bus_type, duration, from, to)
- LOCATED_IN: Hotel -> City

İlişki örnekleri ve sorgu kalıpları:
1. İki kişi arasındaki ilişkiyi bulmak için:
   MATCH (p1:Person)-[r1*..2]-(p2:Person)
   WHERE p1.name CONTAINS 'Kişi1' AND p2.name CONTAINS 'Kişi2'
   WITH p1, p2, r1
   UNWIND r1 AS rel
   RETURN DISTINCT type(rel) as İlişki_Tipi,
          CASE
            WHEN type(rel) = 'KONAKLADI' THEN endNode(rel).name
            WHEN type(rel) IN ['UCAKLA_SEYAHAT_ETTI', 'OTOBUSLE_SEYAHAT_ETTI'] THEN endNode(rel).name
          END as Ortak_Nokta,
          properties(rel) as Detaylar
   LIMIT 10

2. Ortak seyahat edilen yerleri bulmak için:
   MATCH (p1:Person)-[r1]->(c:City)<-[r2]-(p2:Person)
   WHERE p1.name CONTAINS 'Kişi1' AND p2.name CONTAINS 'Kişi2'
   RETURN DISTINCT c.name as Şehir,
          type(r1) as İlişki1, type(r2) as İlişki2,
          properties(r1) as Detay1, properties(r2) as Detay2

3. Ortak konaklama yerlerini bulmak için:
   MATCH (p1:Person)-[r1:KONAKLADI]->(h:Hotel)<-[r2:KONAKLADI]-(p2:Person)
   WHERE p1.name CONTAINS 'Kişi1' AND p2.name CONTAINS 'Kişi2'
   RETURN DISTINCT h.name as Otel,
          properties(r1) as Konaklama1,
          properties(r2) as Konaklama2

Lütfen soruyu analiz ederek en uygun sorgu kalıbını seç ve gerekli düzenlemeleri yap.

Cypher sorgusu:
"""

# Özel yanıt şablonu
RESPONSE_TEMPLATE = """
Verilen Cypher sorgusu ve sonuçlarına dayanarak soruyu detaylı bir şekilde yanıtla.

Soru: {query}
Cypher Sorgusu: {cypher_query}
Sonuçlar: {results}

Yanıtı oluştururken dikkat edilecek noktalar:
1. Eğer iki kişi arasında doğrudan veya dolaylı bir ilişki varsa, bu ilişkiyi açıkça belirt
2. Ortak seyahat edilen yerler varsa, tarihleri ve seyahat tiplerini de belirt
3. Ortak konaklama varsa, otel bilgilerini ve tarihleri de ekle
4. Sonuç bulunamadıysa, bunun nedenini açıkla
5. Yanıtı doğal bir dil kullanarak, anlaşılır şekilde formatla

Yanıt:
"""

def create_qa_chain():
    """QA zinciri oluştur"""
    llm = ChatOpenAI(
        temperature=0.2,  # Biraz yaratıcılık için sıcaklığı artırıyoruz
        openai_api_key=OPENAI_API_KEY,
        model_name="gpt-4",  # Daha iyi anlama için GPT-4 kullanıyoruz
        max_tokens=2000  # Daha detaylı yanıtlar için token limitini artırıyoruz
    )

    cypher_prompt = PromptTemplate(
        template=CYPHER_TEMPLATE,
        input_variables=["query"]
    )

    qa_prompt = PromptTemplate(
        template=RESPONSE_TEMPLATE,
        input_variables=["query", "cypher_query", "results"]
    )

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=cypher_prompt,
        qa_prompt=qa_prompt,
        verbose=True,
        return_direct=True,
        return_intermediate_steps=True,  # Ara adımları görmek için
        top_k=5,  # Daha fazla sonuç almak için
        allow_dangerous_requests=True
    )

    return chain

# Veritabanını oluştur ve QA zincirini hazırla
print("Sistem başlatılıyor...")

# Veritabanının yeniden oluşturulup oluşturulmayacağını kontrol et
RECREATE_DATABASE = False  # Veritabanını yeniden oluşturmak için True yapın

if RECREATE_DATABASE:
    print("Neo4j grafiği hazırlanıyor ve veriler aktarılıyor...")
    create_graph_data()
else:
    print("Mevcut Neo4j grafiği kullanılıyor...")

qa_chain = create_qa_chain()
print("Sistem hazır! Artık Gradio arayüzü ile sorularınızı yanıtlayabilirsiniz.")

def answer_question(question):
    """Gradio'dan gelen soruyu QA zinciri üzerinden yanıtla"""
    try:
        response = qa_chain({"query": question})
        # Gelen yanıtın yapısına göre düzenleme yapıyoruz
        if isinstance(response, dict):
            # Eğer key'lerden biri "result" veya "output" ise onu döndürüyoruz.
            if "result" in response:
                return response["result"]
            elif "output" in response:
                return response["output"]
            else:
                return str(response)
        else:
            return response
    except Exception as e:
        return f"Hata oluştu: {str(e)}"

# Gradio arayüzünü oluştur
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.components.Textbox(lines=2, placeholder="Sorunuzu girin...", label="Soru"),
    outputs=gr.components.Textbox(label="Yanıt"),
    title="Seyahat ve Konaklama Bilgi Grafiği RAG Sistemi",
    description="Neo4j veritabanındaki veriler üzerinden sorularınızı yanıtlayın."
)

if __name__ == "__main__":
    iface.launch(share=True)