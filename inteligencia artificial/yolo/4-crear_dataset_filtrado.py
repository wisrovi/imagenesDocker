import pandas as pd
import shutil
import os
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

os.makedirs("./dataset_filtrado", exist_ok=True)
# borrar todo el contenido de dataset_filtrado, si existe contenido, tanto archivos como carpetas
shutil.rmtree("./dataset_filtrado", ignore_errors=True)
os.makedirs("./dataset_filtrado", exist_ok=True)

df = pd.read_csv("analisis_filtrado.csv")

all_image_file = df["image_file"].tolist()

random.shuffle(all_image_file)  # revuelve la lista de nombres de archivos

# dividir la lista en train (85%) y val (5%) y test (10%) usando train_test_split
train, val = train_test_split(all_image_file, test_size=0.05)
train, test = train_test_split(train, test_size=0.10)
datos = {
    "train": train,
    "val": val,
    "test": test,
}

for grupo_datos in tqdm(["train", "val", "test"], desc="Creando dataset filtrado"):
    lista_imagenes = datos[grupo_datos]
    folder_images = f"./dataset_filtrado/{grupo_datos}/images/"
    folder_labels = f"./dataset_filtrado/{grupo_datos}/labels/"

    os.makedirs(folder_images, exist_ok=True)
    os.makedirs(folder_labels, exist_ok=True)

    for i in tqdm(range(len(lista_imagenes)), desc=f"Procesando {grupo_datos}"):
        image_file = lista_imagenes[i]
        todas_anotaciones = df[df["image_file"] == image_file][
            "annotation_original"
        ].tolist()

        file_name = image_file.split("/")[-1]
        label_file = f"{folder_labels}/{file_name.replace('.jpg', '.txt')}"

        # copiar la imagen a ./dataset_filtrado/images
        shutil.copy(image_file, f"{folder_images}/{file_name}")

        # escribir las anotaciones en el archivo de texto
        with open(label_file, "w") as file:
            for anotacion in todas_anotaciones:
                file.write(anotacion)  # no se agrega \n porque ya lo tiene

# finalmente copiar data.yaml a ./dataset_filtrado
shutil.copy("data.yaml", "./dataset_filtrado/data.yaml")