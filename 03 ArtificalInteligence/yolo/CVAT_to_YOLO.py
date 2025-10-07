import os
import zipfile
import tempfile
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from sklearn.model_selection import train_test_split


def descomprimir_zip(archivo_zip, directorio_destino):
    try:
        with zipfile.ZipFile(archivo_zip, "r") as zip_ref:
            zip_ref.extractall(directorio_destino)
        print("¡Archivo ZIP descomprimido correctamente!")
    except zipfile.BadZipFile:
        print("Error: El archivo proporcionado no es un archivo ZIP válido.")
    except Exception as e:
        print(f"Error al descomprimir el archivo ZIP: {e}")


def yml_to_img_and_label_dict(archivo_xml, imagenes_cvat_buscar):
    # Cargar y analizar el archivo XML
    tree = ET.parse(archivo_xml)
    root = tree.getroot()

    dataset = {}
    for image in root.findall(".//image"):
        nombre_imagen = image.get("name")

        # Si la imagen no está en la lista de imágenes a buscar, saltar a la siguiente
        if not nombre_imagen in imagenes_cvat_buscar:
            print(
                f"La imagen {nombre_imagen} no está en la lista de imágenes a buscar."
            )
            continue

        ancho_img = int(image.get("width"))
        alto_img = int(image.get("height"))

        # Crear un diccionario para cada imagen
        etiquetas = {"ancho": ancho_img, "alto": alto_img}

        # Iterar sobre cada polígono dentro de la imagen
        for polygon in image.findall(".//polygon"):
            etiqueta = polygon.get("label")
            coordenadas_str = polygon.get("points")

            # Convertir el string de coordenadas en una lista de listas [x, y]
            coordenadas = [
                [float(coord) for coord in punto.split(",")]
                for punto in coordenadas_str.split(";")
            ]

            # Agregar las coordenadas a la lista de la etiqueta correspondiente
            if etiqueta in etiquetas:
                etiquetas[etiqueta].append(coordenadas)
            else:
                etiquetas[etiqueta] = [coordenadas]

        # Agregar el diccionario de etiquetas a la imagen correspondiente cuando se hayan encontrado coordenadas
        if len(etiquetas) > 2:
            dataset[nombre_imagen] = etiquetas

            # print(dataset[nombre_imagen])

    # Conjunto para almacenar etiquetas únicas
    etiquetas_unicas = set()
    # Recorrer cada imagen y extraer las etiquetas
    for nombre_imagen, etiquetas in dataset.items():
        for etiqueta in etiquetas:
            if etiqueta != "ancho" and etiqueta != "alto":
                etiquetas_unicas.add(etiqueta)

    # Convertir el conjunto en una lista
    lista_etiquetas = sorted(list(etiquetas_unicas))

    # Asignar un ID de clase a cada etiqueta
    etiquetas_a_id = {}
    for i, etiqueta in enumerate(lista_etiquetas):
        if etiqueta != "alto" and etiqueta != "ancho":
            etiquetas_a_id[etiqueta] = i

    return dataset, etiquetas_a_id


def normalizar_poligono(poligono, ancho_img, alto_img):
    # Normalizar y aplanar las coordenadas
    coords = [coord for punto in poligono for coord in punto]
    coords_normalizadas = [
        str(c / ancho_img if i % 2 == 0 else c / alto_img) for i, c in enumerate(coords)
    ]
    return " ".join(coords_normalizadas)


# Función para mover imágenes y etiquetas
def move_files(images, subset, yolo_images, yolo_labels, BASE_FINAL):
    for img_path in images:
        label_path = img_path.replace(".jpg", ".txt")

        img_path = os.path.join(yolo_images, img_path)
        label_path = os.path.join(yolo_labels, label_path)

        try:
            shutil.copy(label_path, os.path.join(BASE_FINAL, subset, "labels"))
            shutil.copy(img_path, os.path.join(BASE_FINAL, subset, "images"))
        except Exception as e:
            print(e)


def leer_etiquetas_y_contar_clases(directorio):
    contador_clases = Counter()
    for archivo in os.listdir(directorio):
        if archivo.endswith(".txt"):
            ruta_archivo = os.path.join(directorio, archivo)
            with open(ruta_archivo, "r") as file:
                for linea in file:
                    clase = int(linea.split()[0])
                    contador_clases[clase] += 1
    return contador_clases


def leer_parametros():
    parser = argparse.ArgumentParser(
        description="Convierte datos de CVAT a formato YOLO"
    )
    parser.add_argument(
        "project", type=str, help="Ruta al archivo zip del proyecto CVAT"
    )
    parser.add_argument(
        "dataset_yolo",
        type=str,
        help="Ruta al directorio de salida para el dataset YOLO",
    )

    args = parser.parse_args()

    # Accede a los argumentos pasados por línea de comandos
    project_path = args.project
    yolo_dataset_path = args.dataset_yolo

    return project_path, yolo_dataset_path


def main():
    NOMBRE_DATASET_USAR, NOMBRE_DATASET_YOLO = leer_parametros()

    if not NOMBRE_DATASET_USAR.endswith(".zip"):
        print("El archivo proporcionado no es un archivo ZIP.")
        exit()

    if not os.path.exists(NOMBRE_DATASET_USAR):
        print(f"El archivo {NOMBRE_DATASET_USAR} no existe.")
        print("Los archivos disponibles son:")
        archivos_disponibles = [f for f in os.listdir() if f.endswith(".zip")]
        print(archivos_disponibles)
        print(NOMBRE_DATASET_USAR)
        exit()

    if NOMBRE_DATASET_USAR.find("cvat") == -1:
        print(f"El nombre del dataset YOLO no es válido.")
        exit()

    with tempfile.TemporaryDirectory() as temp_dir:
        """
        Descomprimir el archivo ZIP y crear la estructura de carpetas YOLO
        para el dataset seleccionado.
        """
        # descomprimir el archivo ZIP
        descomprimir_zip(f"{NOMBRE_DATASET_USAR}", temp_dir)

        # Crear estructura de carpetas YOLO
        base_dir = f"{temp_dir}/{NOMBRE_DATASET_USAR}/YOLO"
        yolo_images = os.path.join(base_dir, "images")
        yolo_labels = os.path.join(base_dir, "labels")
        os.makedirs(yolo_images, exist_ok=True)
        os.makedirs(yolo_labels, exist_ok=True)

        """
            Crear un inventario de las imágenes existentes en las carpetas Train, Test y Validation
        """
        # Listar las imágenes en las carpetas Train, Test y Validation que estan dentro de la carpeta images
        imagenes_cvat_train = glob.glob(
            os.path.join(temp_dir, "images", "Train", "*.jpg")
        )
        imagenes_cvat_test = glob.glob(
            os.path.join(temp_dir, "images", "Test", "*.jpg")
        )
        imagenes_cvat_valid = glob.glob(
            os.path.join(temp_dir, "images", "Validation", "*.jpg")
        )
        todas_las_imagenes = (
            imagenes_cvat_train + imagenes_cvat_test + imagenes_cvat_valid
        )
        imagenes_cvat_buscar = [img.split(os.sep)[-1] for img in todas_las_imagenes]

        """
            Crear un diccionario con las etiquetas y sus respectivos IDs
            también crear el dataset de imágenes que tengan etiquetas
        """

        # Iterar sobre cada elemento 'image'

        dataset, etiquetas_a_id = yml_to_img_and_label_dict(
            f"{temp_dir}/annotations.xml", imagenes_cvat_buscar
        )

        """
            Las coordenadas de los polígonos deben ser normalizadas y aplanadas
            para ser escritas en el archivo de etiquetas YOLO
            esto es un requisito para el entrenamiento de YOLO
        """

        # Procesar el JSON
        for nombre_imagen, info in dataset.items():
            # Preparar el archivo de etiquetas YOLO
            contenido_etiqueta = []

            for etiqueta, poligonos in info.items():
                if etiqueta in ["ancho", "alto"]:
                    continue  # Saltar las claves de ancho y alto
                class_id = etiquetas_a_id.get(etiqueta, -1)
                if class_id == -1:
                    continue  # Saltar etiquetas no mapeadas

                for poligono in poligonos:
                    linea = f"{class_id} " + normalizar_poligono(
                        poligono, info["ancho"], info["alto"]
                    )
                    contenido_etiqueta.append(linea)

            if nombre_imagen in [img.split(os.sep)[-1] for img in imagenes_cvat_train]:
                ruta_origen = os.path.join(temp_dir, "images", "Train", nombre_imagen)
            elif nombre_imagen in [img.split(os.sep)[-1] for img in imagenes_cvat_test]:
                ruta_origen = os.path.join(temp_dir, "images", "Test", nombre_imagen)
            elif nombre_imagen in [
                img.split(os.sep)[-1] for img in imagenes_cvat_valid
            ]:
                ruta_origen = os.path.join(
                    temp_dir, "images", "Validation", nombre_imagen
                )

            ruta_destino = os.path.join(yolo_images, nombre_imagen)
            shutil.copy(ruta_origen, ruta_destino)

            # Escribir el archivo .txt de etiquetas YOLO
            with open(
                os.path.join(yolo_labels, nombre_imagen.replace(".jpg", ".txt")), "w"
            ) as f:
                f.write("\n".join(contenido_etiqueta))

        # listar cantidad de imagenes y etiquetas en yolo_images y yolo_labels
        print(f"Imágenes en yolo_images: {len(os.listdir(yolo_images))}")
        print(f"Etiquetas en yolo_labels: {len(os.listdir(yolo_labels))}")

        nombres_clases = [etiqueta for etiqueta, id in etiquetas_a_id.items()]

        print(etiquetas_a_id, nombres_clases)

        """
            Crear la carpeta final, llamada dataset dentro de temp_dir
            Dentro de dataset, crear las carpetas train, test y valid
            y dentro de cada una de estas, crear las carpetas images y labels
        """
        BASE_FINAL = os.path.join(temp_dir, "dataset")
        os.makedirs(BASE_FINAL, exist_ok=True)
        os.makedirs(os.path.join(BASE_FINAL, "train", "images"), exist_ok=True)
        os.makedirs(os.path.join(BASE_FINAL, "train", "labels"), exist_ok=True)
        os.makedirs(os.path.join(BASE_FINAL, "test", "images"), exist_ok=True)
        os.makedirs(os.path.join(BASE_FINAL, "test", "labels"), exist_ok=True)
        os.makedirs(os.path.join(BASE_FINAL, "val", "images"), exist_ok=True)
        os.makedirs(os.path.join(BASE_FINAL, "val", "labels"), exist_ok=True)

        """
            Mover las imágenes y etiquetas de YOLO a la carpeta final
            para esto se toma la lista de imágenes y etiquetas de YOLO
            y se divide en train, test y valid usando train_test_split
            
            Se usan los porcentajes de 85% para train, 10% para test y 5% para valid
        """
        train_images, test_images = train_test_split(
            os.listdir(yolo_images), test_size=0.15, random_state=42
        )
        test_images, valid_images = train_test_split(
            test_images, test_size=0.33, random_state=42
        )

        # Mover archivos
        move_files(train_images, "train", yolo_images, yolo_labels, BASE_FINAL)
        move_files(test_images, "test", yolo_images, yolo_labels, BASE_FINAL)
        move_files(valid_images, "val", yolo_images, yolo_labels, BASE_FINAL)

        """
            crear el archivo data.yaml que contiene la información de las clases
        """
        # Crear archivo data.yml
        data_yml_content = f"""
train: '{NOMBRE_DATASET_YOLO}/train/images'
val: '{NOMBRE_DATASET_YOLO}/val/images'
test: '{NOMBRE_DATASET_YOLO}/test/images'

# Classes
nc: {len(nombres_clases)}  # número de clases
names: {nombres_clases}  # lista de nombres de clases
        """

        with open(os.path.join(BASE_FINAL, "data.yaml"), "w") as file:
            file.write(data_yml_content)

        # crear un grafico de barras con la cantidad de imagenes en train, test y valid
        fig, ax = plt.subplots()
        ax.bar(
            ["train", "test", "val"],
            [len(train_images), len(test_images), len(valid_images)],
        )
        # poner encima de las barras el valor de cada una
        for i, v in enumerate([len(train_images), len(test_images), len(valid_images)]):
            ax.text(i, v + 3, str(v), ha="center", va="bottom")
        # poner titulo al grafico
        ax.set_title("Cantidad de imágenes en train, test y valid")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        # plt.show()
        # guardar el grafico como imagen
        fig.savefig(os.path.join(BASE_FINAL, "cantidad_imagenes_train_test_valid.png"))

        """
            Crear un histograma de clases
        """
        # contar la cantidad de datos por clase
        contador_clases_train = leer_etiquetas_y_contar_clases(
            os.path.join(BASE_FINAL, "train", "labels")
        )
        contador_clases_test = leer_etiquetas_y_contar_clases(
            os.path.join(BASE_FINAL, "test", "labels")
        )
        contador_clases_valid = leer_etiquetas_y_contar_clases(
            os.path.join(BASE_FINAL, "val", "labels")
        )
        contador_total = (
            contador_clases_train + contador_clases_test + contador_clases_valid
        )

        # crear un grafico de barras con la cantidad de datos por clase
        fig, ax = plt.subplots()
        ax.bar(
            [nombres_clases[i] for i in contador_total.keys()], contador_total.values()
        )
        # poner encima de las barras el valor de cada una
        for i, v in enumerate(contador_total.values()):
            ax.text(i, v + 3, str(v), ha="center", va="bottom")
        # poner titulo al grafico
        ax.set_title("histograma de clases")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(BASE_FINAL, "histograma_clases.png"))

        # guardar los datos del histograma en un archivo csv con pandas
        df = pd.DataFrame(contador_total.items(), columns=["clase", "cantidad"])
        df.to_csv(os.path.join(BASE_FINAL, "histograma_clases.csv"), index=False)

        """
            Comprimir el dataset final en un archivo ZIP
        """
        carpeta_comprimir = os.path.join(temp_dir, "dataset")
        shutil.make_archive(NOMBRE_DATASET_YOLO, "zip", carpeta_comprimir)
        print(f"El dataset {NOMBRE_DATASET_YOLO}.zip ha sido creado exitosamente.")


if __name__ == "__main__":
    main()

    # ejemplo:
    # python3 CVAT_to_YOLO.py "<nombre_archivo>.zip" "<nombre_dataset_yolo>"
    # nombre_archivo.zip es el archivo zip del proyecto CVAT
    # nombre_dataset_yolo es el nombre del dataset YOLO que se creará a partir del proyecto CVAT
    
    # python3 CVAT_to_YOLO.py "project_eyesdcar-damages-2024_02_07_10_07_38-cvat for images 1.1.zip" "damages_seg_eyesDcar"
