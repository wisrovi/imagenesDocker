from django.test import TestCase

# Create your tests here.

import requests

server="http://localhost:8000"

parametros = {
    "username": "wisrovi",
    "password": 12345678
}
rta = requests.post(server+"/login/", data=parametros)
token = None
if rta.status_code == 200:
    token = rta.json().get("token")

    parametros = {
        "Authorization": "Token " + token
    }

    data = {
        "mac": "10:E7:C6:F8:7F:E7",
        "project": "Smartcorner",
        "path": "/home/prueba",
        "computername": "computername",
        "md5": "md5",
        "hash": "hash",
        "checksum": "checksum"
    }

    rta = requests.post(server+"/licence/", headers=parametros, data=data)

    print(rta.status_code)
    print(rta.text)

if __name__ == '__main__':
    pass


