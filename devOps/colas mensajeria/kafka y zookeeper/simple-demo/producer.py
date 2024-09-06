"""
confluent-kafka
kafka-python
aiokafka
"""
import json
import pandas as pd
from confluent_kafka import Producer
# Messages will be serialized as JSON
def serializer(message):
    return json.dumps(message).encode("utf-8")
# Kafka Producer (No stream)
producer = Producer(
    {"bootstrap.servers": "192.168.1.60:9092"},
)
if __name__ == "_main_":
    
    df = pd.read_csv('tu_archivo.csv')
    for index, row in df.iterrows():
        row_json = row.to_json()   # {'fecha': 'queso', 'tf': 10, 'pt': 100}
        
    producer.flush()
    
    
    
    
    dummy_message = {"nombre": "queso"}
    producer.produce(
        "messages",
        key="key",
        value=serializer(dummy_message),
    )
    producer.flush()
