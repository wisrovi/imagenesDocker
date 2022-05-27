import uvicorn

import math
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.responses import JSONResponse

from config.metadata import titulo, description, version, contact
from lib.Motor_recomendacion import Motor_recomendacion
from lib.Uniongr import Uniongr
from lib.GeneradorSoluciones import GeneradorSoluciones
from shapely.geometry import Polygon


class Coordenada(BaseModel):
    lat: float
    lng: float


class Item(BaseModel):
    coordenadas: List[Coordenada]
    solicitud_cantidad_soluciones: Optional[int] = 5


class ItemOut(BaseModel):
    coordenadas: List[Coordenada]


class Item_Recomendado(BaseModel):
    sugerencias: List[ItemOut]


class ItemRta(BaseModel):
    original: Item
    recomendado: Item_Recomendado
    intersecciones: List[str]
    cantidad_zonas_comparadas: int
    file_import: str
    error: str
    son_iguales: bool


def preparar_coordenada_comparar(zona):
    coordenadas = list()
    for coordenada in zona:
        coordenadas.append((coordenada.get("lat"), coordenada.get("lng")))
    return coordenadas


def correcion_angulo(x, y, angulo):
    if x < 0 and y < 0:
        return 180 + angulo
    if x < 0 and y > 0:
        return 180 - angulo
    if x > 0 and y > 0:
        return angulo
    if x > 0 and y < 0:
        return 360 - angulo


def calculo_angulo(punto_a, centro):
    diferencia_x = punto_a[0] - centro[0]
    diferencia_y = punto_a[1] - centro[1]
    catetox = abs(diferencia_x)
    catetoy = abs(diferencia_y)
    angulo = math.atan(catetoy / catetox)
    angulo = math.degrees(angulo)
    angulo = correcion_angulo(diferencia_x, diferencia_y, angulo)
    return angulo


def bubbleSort(alist):
    for _ in range(len(alist) - 1):
        swp = False
        for j in range(len(alist) - 1):
            if alist[j][0] > alist[j + 1][0]:
                temp_point = alist[j]
                alist[j] = alist[j + 1]
                alist[j + 1] = temp_point
                swp = True
        if not swp:
            break


def preparar_coordenada_entregar(zona, ordenar=False):
    coordenadas = list()

    if ordenar:
        if isinstance(zona, list):
            zona = list(set(zona))
        else:
            # print(zona)
            pass

        lat, lng = list(), list()
        for zon in zona:
            lat.append(zon[0])
            lng.append(zon[1])

        centro = (min(lat) + (max(lat) - min(lat)) / 2, min(lng) + (max(lng) - min(lng)) / 2)

        almacen = list()
        for i, zon in enumerate(zona):
            angulo = calculo_angulo(zon, centro)
            almacen.append((angulo, i, zon))
        bubbleSort(almacen)

        entregar = list()
        for i in almacen:
            # print(i[1])
            entregar.append(zona[i[1]])

        zona = entregar

    for coordenada in zona:
        coord = dict()
        coord['lat'] = coordenada[0]
        coord['lng'] = coordenada[1]
        coordenadas.append(coord)

    return coordenadas


tags_metadata = [
    {
        "name": "recomendar",
        "description": "Webservice que busca solapamiento en las coordenadas recibidas y brindar una propuesta de coordenadas donde el solapamiento no se da",
        "externalDocs": {
            "description": "Referencia",
            "url": "https://pro.arcgis.com/es/pro-app/2.7/help/mapping/properties/coordinate-systems-and-projections.htm",
        },
    },
]

app = FastAPI(title=titulo,
              description=description,
              version=version,
              contact=contact,
              openapi_tags=tags_metadata)
motor_recomendacion = Motor_recomendacion()


@app.middleware("http")
async def verify_user_agent(request: Request, call_next):
    if request.headers['User-Agent'].find("Mobile") == -1:
        response = await call_next(request)
        return response
    else:
        return JSONResponse(content={
            "message": "we do not allow mobiles"
        }, status_code=401)


origins = [
    "http://localhost",
    "https://localhost",
    "http://18.215.72.70",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/evaluar_coordenadas_correctas", response_model=ItemRta, tags=["recomendar"])
async def create_item(item: Item):
    global son_iguales, listado_soluciones_entregar
    str_file = str()
    error = str()
    recomendado = list()
    intercecciones = list()
    cantidad_zonas_usadas_comparacion = 0

    item_dict = item.dict()
    if len(item_dict.get("coordenadas")) < 3:
        error = "Error, no hay suficientes coordenadas para generar un poligono"

    if len(error) == 0:
        coordenadas_recibidas = item_dict.get("coordenadas")
        nueva_zona = preparar_coordenada_comparar(coordenadas_recibidas)

        union = Uniongr()
        zonas = union.get_zonas_coordenadas()
        cantidad_zonas_usadas_comparacion = len(zonas)

        hay_error = False
        son_iguales = False
        coleccion_zonas_error = list()

        for zona in zonas:
            vieja_zona = zona.get("poligono")

            coord_zona_db = Polygon(vieja_zona)
            coord_nueva_zona = Polygon(nueva_zona)
            # Yserna : recorrer zonas para validar  si las coord no han tenido modificacion
            if coord_zona_db.equals(coord_nueva_zona):
                son_iguales = True
                break

        if not son_iguales:
            for zona in zonas:
                vieja_zona = zona.get("poligono")

                if motor_recomendacion.deteccion_interceccion(vieja_zona, nueva_zona):
                    intercecciones.append(zona.get("nombre"))

                    temporal = motor_recomendacion.correccion_coordenadas(vieja_zona, nueva_zona)
                    if isinstance(temporal, list):
                        nueva_zona = temporal
                    else:
                        hay_error = True
                        coleccion_zonas_error.append(zona['nombre'])
                        print(zona)
        if not hay_error:
            error = ""
        else:
            error += "No se pudo completar la corrección de coordenadas al 100%, debe verificar las coordenadas cercanas a la(s) zona(s): "
            for err in coleccion_zonas_error:
                error += err + ", "

            error = error[:-2]
            error += " para finalizar el registro de la zona."

        print("Nueva zona", len(nueva_zona), nueva_zona)

        cantidad_soluciones_generar = item.dict().get("solicitud_cantidad_soluciones")
        generator = GeneradorSoluciones(nueva_zona)
        generator.solve(cantidad_soluciones_generar)
        soluciones = generator.get_response()

        listado_soluciones_entregar = list()
        unica = False
        for solucion in soluciones:
            recomendado = [
                Coordenada(lat=coor.get("lat"), lng=coor.get("lng"))
                           for coor in preparar_coordenada_entregar(solucion, False)
            ] if len(intercecciones) > 0 else item.dict().get("coordenadas")
            if not unica:
                listado_soluciones_entregar.append(Item(coordenadas=recomendado))
                if len(intercecciones) == 0:
                    unica = True

        str_file = str()
        for rec in recomendado:
            try:
                str_file += str(rec.lat)
                str_file += ","
                str_file += str(rec.lng)
                str_file += " | "
            except AttributeError as atributoError:
                str_file += str(rec.get("lat"))
                str_file += ","
                str_file += str(rec.get("lng"))
                str_file += " | "
        str_file = str_file[:-3]
        with open("zonas_importar.txt", "w") as f:
            f.write(str_file)

    return ItemRta(original=item,
                   recomendado=Item_Recomendado(sugerencias=listado_soluciones_entregar),
                   intersecciones=intercecciones,
                   cantidad_zonas_comparadas=cantidad_zonas_usadas_comparacion,
                   file_import=str_file,
                   error=error,
                   son_iguales=son_iguales)


@app.get("/")
async def version():
    return "Version 1.1. LOCAL del motor de recomendacion de coordenadas"

# para ejecutar localmente en Debug
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5050)
