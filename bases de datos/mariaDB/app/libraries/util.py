import base64
import datetime
import json

import requests

from config.config import key_aes
from libraries.AES_Union import AES_Union


class Util:
    @staticmethod
    def dict_to_aes128(diccionario):
        data = json.dumps(diccionario, sort_keys=True)
        data = data.encode('ascii')
        data = base64.b64encode(data)
        data = data.decode('ascii')
        aes_last_internet = AES_Union(key=key_aes, msg=data)
        info_save = aes_last_internet.encript()
        return info_save

    @staticmethod
    def aes_to_json(data):
        aes_last_internet = AES_Union(key=key_aes, msg=data)
        data_str = aes_last_internet.decrypt(data)
        data_str = data_str.encode("ascii")
        data_str = base64.b64decode(data_str)
        data_str = data_str.decode("ascii")
        return json.loads(data_str)

    @staticmethod
    def get_time_internet(hay_internet=False):
        date_time_obj = datetime.datetime.now()
        if hay_internet:
            rta = requests.get("http://worldtimeapi.org/api/timezone/America/Bogota", timeout=2.50)
            if rta.status_code == 200:
                time = rta.json().get("datetime")
                time = str(time).split(".")[0]
                format = '%Y-%m-%dT%H:%M:%S'
                date_time_obj = datetime.datetime.strptime(time, format)
                print("Datos online")
        return date_time_obj, hay_internet

    @staticmethod
    def valid_internet_conection():
        conexion_internet = False
        try:
            request = requests.get("http://www.google.com", timeout=5)
        except (requests.ConnectionError, requests.Timeout):
            # print("Sin conexión a internet.")
            conexion_internet = False
        else:
            # print("Con conexión a internet.")
            conexion_internet = True
        return conexion_internet