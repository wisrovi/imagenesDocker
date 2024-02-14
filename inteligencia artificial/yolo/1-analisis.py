import numpy as np
import glob
import cv2
import os
import yaml
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


folders = ["./NAS", "./CVAT"]

for folder in folders:
    data_yaml_path = "data.yaml"

    images = glob.glob(os.path.join(folder, "train/images/*.jpg"))
    labels = glob.glob(os.path.join(folder, "train/labels/*.txt"))

    # Cargar el archivo data.yaml
    class_names = []
    with open(data_yaml_path, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        # Obtener los nombres de las etiquetas
        class_names = data["names"]

    datos_validos = []
    numeros_float = []
    # Leer las anotaciones y visualizar las imágenes
    for i in tqdm(
        range(len(labels)),
        desc=f"Analizando {folder}...",
        unit="imagen",
        unit_scale=True,
    ):
        image_file = None
        txt_file = labels[i]
        with open(txt_file, "r") as file:
            annotations = file.readlines()

        if len(annotations) == 0:
            print(f"No se encontraron anotaciones en el archivo {txt_file}")
            continue

        # encontrar la imagen correspondiente buscando por el nombre del archivo
        image_name = (
            os.path.basename(txt_file).replace(".txt", ".jpg").split(os.sep)[-1]
        )

        # buscar "image_name" en "images" para ver cual es su index en la lista, con find
        # con ese index, buscar la imagen correspondiente en "images"
        # y leer la imagen con cv2
        image = None
        for j, img in enumerate(images):
            if img.find(image_name) != -1:
                image = cv2.imread(img)
                image_file = img
                break

        if image is None:
            print(f"No se encontró la imagen correspondiente al archivo {txt_file}")
            continue

        height, width, _ = image.shape
        for annotation_original in annotations:
            annotation = annotation_original
            annotation = annotation.strip().split()
            class_index = int(annotation[0])
            class_label = class_names[class_index]

            # hallar los puntos de la segmentación en numpy array
            segmentation_points = [float(f) for f in map(float, annotation[1:])]
            points_array = np.array(segmentation_points, np.float32)
            points_array_normalized = points_array.reshape((-1, 2))
            points_array = (points_array_normalized * np.array([width, height])).astype(
                np.int32
            )

            # Crear un polígono con los puntos de la segmentación
            polygon = Polygon(points_array)
            area = polygon.area

            datos_validos.append(
                {
                    "folder": folder,
                    "txt_file": txt_file,
                    "image_file": image_file,
                    "area": area,
                    "class_label": class_label,
                    "annotation_original": annotation_original,
                }
            )
            # print(
            #     f"El área de la segmentación de la clase {class_label} es {area:.2f} píxeles cuadrados"
            # )
            numeros_float.append(area)


min_area = min(numeros_float)
max_area = max(numeros_float)
mean_area = np.mean(numeros_float)
median_area = np.median(numeros_float)
std_area = np.std(numeros_float)

print(f"El área mínima es {min_area:.2f} píxeles cuadrados")
print(f"El área máxima es {max_area:.2f} píxeles cuadrados")
print(f"El área promedio es {mean_area:.2f} píxeles cuadrados")
print(f"La mediana del área es {median_area:.2f} píxeles cuadrados")
print(f"La desviación estándar del área es {std_area:.2f} píxeles cuadrados")


datos = np.array(numeros_float).reshape(-1, 1)

# crear dataframe con datos_validos
df = pd.DataFrame(datos_validos)
df.to_csv("analisis.csv", index=False)

print(df.head(5))


# crear histograma de la columna "class_label" del dataframe

clases_data = df["class_label"]
clases_data = clases_data.value_counts()
clases_data = clases_data.to_dict()


# crear histograma de la columna "class_label" del dataframe
plt.bar([nombre_clase for nombre_clase in clases_data.keys()], clases_data.values())

for i, v in enumerate(clases_data.values()):
    plt.text(i, v + 3, str(v), ha="center", va="bottom")

plt.xlabel("Clase")
plt.ylabel("Cantidad de Muestras")
plt.title("Histograma de Muestras por Clase en Dataset de Entrenamiento")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# guardar el histograma
plt.savefig("histograma.png")
plt.show()
