

import time

class Reader:
    import pandas as pd
    def __init__(self, path_file, basic=True):     
        if basic:  
            data = self.pd.read_csv(path_file)
        else:
            data = self.pd.read_csv(path_file,sep=',',header='infer',encoding='iso-8859-1')
        columns_tmp = data.columns
        #columns_tmp = [column_name.lower() for column_name in columns_tmp]
        data.columns = columns_tmp

        #data.rename(columns={'open': 'open_price', 'close': 'close_price', 'adj close': 'adj_close'}, inplace=True)
        self.data_columns = list(data.columns)
        self.data = data
    
    def leer_fila_dict(self, x):
        row = self.data.iloc[x, :]
        row_dict = {}
        for key, value in zip(self.data_columns, row):
            row_dict[key] = value
        return row_dict

    def getCantidadFilas(self):
        return self.data.shape[0]

class PublicarKafka:
    import json
    import pickle
    from kafka import KafkaProducer
    def __init__(self, topic, servers=['localhost:9092']) -> None:
        self.producer = self.KafkaProducer(
            bootstrap_servers=servers,
            value_serializer=self.serializer
        )

        self.topic = topic

    def serializer(self, message):
        #return self.json.dumps(message).encode('utf-8')
        #return lambda x: self.pickle.dumps(x)
        return self.pickle.dumps(message)

    def enviar(self, msg):
        self.producer.send(self.topic, msg)

if __name__ == '__main__':
    #reader = Reader('NFLX_2019.csv')
    reader = Reader('data.csv')
    kafka = PublicarKafka('messages', ['localhost:9092'])
    for fila_n in range(reader.getCantidadFilas()):
        message = reader.leer_fila_dict(fila_n)
        message['id_fila'] = fila_n
        kafka.enviar(message)
        print(f'Message {fila_n + 1}: {message}')
        time.sleep(5)

    
