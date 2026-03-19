Message Queues
==============

This section documents the message queue systems in the ``Queues for services/`` directory.

Celery
------

Celery is a distributed task queue for Python applications.

Location: ``Queues for services/celery/``

Components
~~~~~~~~~~

- ``celery_core.py``: Core Celery configuration
- ``Machine.py``: Machine learning model wrapper
- ``State.py``: State management
- ``tasks.py``: Task definitions
- ``example_machine.py``: Usage example

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Queues\ for\ services/celery/
   docker-compose up -d

Usage
~~~~~

Example task execution:

.. code-block:: python

   from tasks import add
   result = add.delay(4, 4)
   print(result.get())

Kafka and Zookeeper
-------------------

Apache Kafka is a distributed event streaming platform.

Location: ``Queues for services/kafka y zookeeper/``

Components
~~~~~~~~~~

- **Kafka Broker**: Message broker
- **Zookeeper**: Coordination service
- **Python Examples**: Producer and consumer scripts

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd "Queues for services/kafka y zookeeper/"
   docker-compose up -d

Python Examples
~~~~~~~~~~~~~~~

Producer:

.. code-block:: python

   from kafka import KafkaProducer
   producer = KafkaProducer(bootstrap_servers='localhost:9092')
   producer.send('test-topic', b'Hello, Kafka!')

Consumer:

.. code-block:: python

   from kafka import KafkaConsumer
   consumer = KafkaConsumer('test-topic', bootstrap_servers='localhost:9092')
   for message in consumer:
       print(message.value)

Kurento
-------

Kurento is a WebRTC media server.

Location: ``Queues for services/kurento/``

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Queues\ for\ services/kurento/
   docker-compose up -d

Configuration
~~~~~~~~~~~~~

- Custom Kurento configuration in ``etc/kurento/kurento.conf.json``
- SDP pattern file for media processing

MQTT
----

MQTT is a lightweight messaging protocol for IoT and mobile applications.

Location: ``Queues for services/mqtt/``

Components
~~~~~~~~~~

- **Mosquitto Broker**: MQTT broker
- **Python Examples**: Publisher and receiver scripts

Deployment
~~~~~~~~~~

.. code-block:: bash

   cd Queues\ for\ services/mqtt/
   docker-compose up -d

Python Examples
~~~~~~~~~~~~~~~

Publisher:

.. code-block:: python

   import paho.mqtt.client as mqtt
   client = mqtt.Client()
   client.connect("localhost", 1883)
   client.publish("test/topic", "Hello, MQTT!")

Receiver:

.. code-block:: python

   import paho.mqtt.client as mqtt

   def on_message(client, userdata, message):
       print(f"Received: {message.payload.decode()}")

   client = mqtt.Client()
   client.on_message = on_message
   client.connect("localhost", 1883)
   client.subscribe("test/topic")
   client.loop_forever()