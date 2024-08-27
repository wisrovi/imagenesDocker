"""
kafka-python-ng
confluent-kafka
kafka-python
aiokafka
"""


from kafka import KafkaConsumer
import json


def deserializer(message):
    return json.loads(message.decode("utf-8"))


consumer = KafkaConsumer(
    "messages",
    bootstrap_servers="192.168.1.60:9092",
    auto_offset_reset="latest",
)


for message in consumer:
    value = deserializer(message.value)
    
    print(f"Message received: {value}")
