from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

app = FastAPI()

FOLDER_SAVE_ZIPs = os.environ.get("PATH_FILES", "/data")

# Ruta para servir archivos estáticos (como CSS, JavaScript, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Plantillas HTML usando Jinja2
templates = Jinja2Templates(directory="templates")


def crear_folder_segun_timestamp():
    import time
    import datetime

    # Obtener el timestamp actual
    timestamp = time.time()
    # Convertir el timestamp a datetime
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    # Crear la carpeta con el timestamp
    #os.makedirs(f"{FOLDER_SAVE_ZIPs}/{dt_object.strftime('%Y-%m-%d_%H-%M-%S')}")
    # Retornar el nombre de la carpeta creada
    return dt_object.strftime("%Y-%m-%d_%H-%M-%S")


# Ruta para manejar el formulario del primer card
@app.post("/uploadfile1/")
async def create_upload_file1(file: UploadFile = File(...)):
    if len(file.filename) < 1:
        return {"filename": "No se ha seleccionado ningún archivo"}

    sub_folder = crear_folder_segun_timestamp()

    contents = await file.read()

    final_file = f"{FOLDER_SAVE_ZIPs}/{sub_folder}-{file.filename}"
    with open(f"{final_file}", "wb") as f:
        f.write(contents)
    return {"filename": final_file}


# Ruta para manejar el formulario del segundo card
@app.post("/uploadfile2/")
async def create_upload_file2(file: UploadFile = File(...)):
    if len(file.filename) < 1:
        return {"filename": "No se ha seleccionado ningún archivo"}

    sub_folder = crear_folder_segun_timestamp()

    contents = await file.read()

    final_file = f"{FOLDER_SAVE_ZIPs}/{sub_folder}-{file.filename}"
    with open(f"{final_file}", "wb") as f:
        f.write(contents)
    return {"filename": final_file}


# Ruta para manejar el formulario del tercer card
@app.post("/uploadfile3/")
async def create_upload_file3(file: UploadFile = File(...)):
    if len(file.filename) < 1:
        return {"filename": "No se ha seleccionado ningún archivo"}

    sub_folder = crear_folder_segun_timestamp()

    contents = await file.read()

    final_file = f"{FOLDER_SAVE_ZIPs}/{sub_folder}-{file.filename}"
    with open(f"{final_file}", "wb") as f:
        f.write(contents)
    return {"filename": final_file}


# Ruta principal
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    # Datos dinámicos para la Card 1
    datos_card1 = "Datos de la Card 1"
    datos_card2 = "Datos de la Card 2"
    datos_card3 = "Datos de la Card 3"
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "datos_card1": datos_card1,
            "datos_card2": datos_card2,
            "datos_card3": datos_card3,
        },
    )


# Crear la carpeta 'data' si no existe
if not os.path.exists("data"):
    os.makedirs("data")
