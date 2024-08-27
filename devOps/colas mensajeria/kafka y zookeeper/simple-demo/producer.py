"""
confluent-kafka
kafka-python
aiokafka
"""
import json
from confluent_kafka import Producer
# Messages will be serialized as JSON
def serializer(message):
    return json.dumps(message).encode("utf-8")
# Kafka Producer (No stream)
producer = Producer(
    {"bootstrap.servers": "192.168.1.60:9092"},
)
if _name_ == "_main_":
    dummy_message = {"nombre": "queso"}
    producer.produce(
        "messages",
        key="key",
        value=serializer(dummy_message),
    )
    producer.flush()
