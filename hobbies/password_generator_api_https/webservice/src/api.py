from local_aplication.libraries.Mongo_Password_Generator import Buscar_generar_clave, Users

from api_template.base import app


from pydantic import BaseModel
class Item(BaseModel):
    user: str


class ItemRta(BaseModel):
    basic: str
    encript: str


@app.get("/info/")
async def version():
    return "Esta es una api que genera claves unicas para un usuario dado."


@app.post("/password")
async def version(item: Item, response_model=ItemRta, tags=["password"]):
    ORIGIN = Users.api
    item_dict = item.dict()
    USER = item_dict.get("user")
    rta = Buscar_generar_clave(USER, ORIGIN)
    return ItemRta(basic=rta.get('basic'),
                   encript=rta.get('sha256'))


# para ejecutar localmente en Debug
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
