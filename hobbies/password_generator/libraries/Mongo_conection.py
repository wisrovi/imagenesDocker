from pymongo import MongoClient, ASCENDING
from Dto.Dto import *

# ********************************   CONFIG   ************************************
import json

data = dict()
with open('config/config_mongo.json') as f:
    data = json.load(f)
    #print(data)

mongo = data['mongo']
user = data['user']
password = data['password']
database = data['database']
coleccion_mongo = data['coleccion_mongo']

# ********************************   CONECTION   ************************************
client = MongoClient(f'mongodb://{user}:{password}@{mongo}:27017/')
db = client[database]
collection = db[coleccion_mongo]

# ********************************   CREO REGLAS EN MONGO PARA NO REPETIR DATOS   ************************************
collection.create_index([(Columns.author, ASCENDING)], unique=True)
collection.create_index([(Columns.password, ASCENDING)], unique=True)

# ********************************   FUNCIONES   ************************************
def buscar(find:dict):
    return collection.find( find ).sort(Columns.date_create)

def cuantos_datos_hay_guardados(find:str):
    return collection.count_documents(find)

def validar_existe_author(author_find:str):
    user_find = None
    curs = buscar( { Columns.author: author_find } )
    # print("conincidencias", cuantos_datos_hay_guardados({ Columns.author: author_find }))
    for item in curs:
        #print("author find", item)
        user_find = dict(author=item['author'], password=item['password'])

    return user_find

def validar_existe_password(password_find:str):
    pass_find = None
    curs = buscar( { f'{Columns.password}.{Columns.basic}': password_find } )
    for item in curs:
        #print("password find", item)
        pass_find = dict(author=item['author'], password=item['password'])

    return pass_find

if __name__ == "__main__":
    print(sorted(list(collection.index_information())))
    print("nombres colecciones existentes", db.list_collection_names())
