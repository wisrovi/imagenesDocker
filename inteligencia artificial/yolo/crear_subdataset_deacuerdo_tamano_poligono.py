import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import glob
import cv2
import os
import yaml
from tqdm import tqdm

from shapely.geometry import Polygon


MIN_AREA = 5000  # 109910.0 # 44514.0 # 0.0
MAX_AREA = 2000000  # np.inf1


data_yaml_path = "data.yaml"

images_train = glob.glob("train/images/*.jpg")
images_test = glob.glob("test/images/*.jpg")
images_valid = glob.glob("val/images/*.jpg")

labels_train = glob.glob("train/labels/*.txt")
labels_test = glob.glob("test/labels/*.txt")
labels_valid = glob.glob("val/labels/*.txt")

images = images_train + images_test + images_valid
labels = labels_train + labels_test + labels_valid

# Cargar el archivo data.yaml
class_names = []
with open(data_yaml_path, "r") as file:
    data = yaml.load(file, Loader=yaml.FullLoader)
    # Obtener los nombres de las etiquetas
    class_names = data["names"]

if len(class_names) == 0:
    print("No se encontraron nombres de clases en el archivo data.yaml")
    exit()


datos_validos = []
numeros_float = []
# Leer las anotaciones y visualizar las imágenes
for i in tqdm(range(len(labels))):
    txt_file = labels[i]
    with open(txt_file, "r") as file:
        annotations = file.readlines()

    if len(annotations) == 0:
        print(f"No se encontraron anotaciones en el archivo {txt_file}")
        continue

    # encontrar la imagen correspondiente buscando por el nombre del archivo
    image_name = os.path.basename(txt_file).replace(".txt", ".jpg").split(os.sep)[-1]

    # buscar "image_name" en "images" para ver cual es su index en la lista, con find
    # con ese index, buscar la imagen correspondiente en "images"
    # y leer la imagen con cv2
    image = None
    for j, img in enumerate(images):
        if img.find(image_name) != -1:
            image = cv2.imread(img)
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

        to_save = False
        if MIN_AREA is not None:
            if area < MIN_AREA or area > MAX_AREA:
                # print(
                #     f"El área de la segmentación es menor que el área mínima permitida"
                # )
                continue
            else:
                to_save = True
        else:
            to_save = True

        if to_save:
            datos_validos.append(
                {
                    "txt_file": txt_file,
                    "image_file": image,
                    "area": area,
                    "class_label": class_label,
                    "annotation_original": annotation_original,
                }
            )
            # print(
            #     f"El área de la segmentación de la clase {class_label} es {area:.2f} píxeles cuadrados"
            # )
            numeros_float.append(area)


print("\n" * 5)


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


print("\n" * 5)


# Convertir la lista a un arreglo numpy
datos = np.array(numeros_float).reshape(-1, 1)


# Definir el rango de opciones de número de clusters a probar
num_clusters_opciones = range(1, 11)  # Por ejemplo, de 1 a 5 clusters
inercias = []

# Probar diferentes opciones de número de clusters
for num_clusters in num_clusters_opciones:
    print(f"Probando con {num_clusters} clusters...")

    # Inicializar y ajustar el modelo KMeans
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(datos)

    # Obtener las etiquetas de cluster y los centroides
    etiquetas = kmeans.labels_
    centroides = kmeans.cluster_centers_

    # Calcular la inercia y agregarla a la lista
    inercia = kmeans.inertia_
    inercias.append(inercia)

    # Visualizar los clusters
    # plt.figure()
    # plt.scatter(datos, np.zeros_like(datos), c=etiquetas, cmap='viridis')
    # plt.scatter(centroides, np.zeros_like(centroides), c='red', marker='x')
    # plt.title(f"Número de Clusters: {num_clusters}")
    # plt.show()


print("q0")

# Graficar la curva de la "ley del codo"
plt.figure()
plt.plot(num_clusters_opciones, inercias, marker="o")
plt.xlabel("Número de Clusters")
plt.ylabel("Inercia")
plt.title("Ley del Codo")

# Encontrar el codo (punto de inflexión)
deltas = np.diff(inercias, 2)  # Segunda derivada de las inercias
k_optimo = num_clusters_opciones[np.argmax(deltas) + 1]  # El máximo después del mínimo
plt.axvline(x=k_optimo, color="red", linestyle="--", label="Número óptimo de clusters")
plt.legend()


print("q1")
plt.savefig(f"ley_del_codo (min_area={MIN_AREA:.2f} - max_area={MAX_AREA:.2f}).png")


print(f"Número óptimo de clusters según el método del codo: {k_optimo}")


kmeans_optimo = KMeans(n_clusters=k_optimo)
kmeans_optimo.fit(datos)

# Obtener las etiquetas de cluster
etiquetas_optimas = kmeans_optimo.labels_

# Contar cuántos datos hay en cada cluster
conteo_clusters = Counter(etiquetas_optimas)

# Imprimir el conteo de datos en cada cluster
for cluster, conteo in conteo_clusters.items():
    print(f"Cluster {cluster}: {conteo} datos")


# Inicializar listas para almacenar las estadísticas de cada cluster
medias_clusters = []
maximos_clusters = []
minimos_clusters = []
promedios_clusters = []
medianas_clusters = []
desviaciones_clusters = []

# Iterar sobre cada cluster
for cluster_id in range(k_optimo):
    # Filtrar los datos pertenecientes al cluster actual
    datos_cluster = datos[etiquetas_optimas == cluster_id].flatten()

    # Calcular la media, la moda, el máximo y el mínimo para el cluster actual
    media_cluster = np.mean(datos_cluster)
    maximo_cluster = np.max(datos_cluster)
    minimo_cluster = np.min(datos_cluster)
    promedio_cluster = np.average(datos_cluster)
    mediana_cluster = np.median(datos_cluster)
    desviacion_cluster = np.std(datos_cluster)

    # Agregar las estadísticas a las listas correspondientes
    medias_clusters.append(media_cluster)
    maximos_clusters.append(maximo_cluster)
    minimos_clusters.append(minimo_cluster)
    promedios_clusters.append(promedio_cluster)
    medianas_clusters.append(mediana_cluster)
    desviaciones_clusters.append(desviacion_cluster)

# Imprimir las estadísticas de cada cluster
for cluster_id in range(k_optimo):
    print(f"Cluster {cluster_id}:")
    print(f"  Media: {medias_clusters[cluster_id]}")
    print(f"  Máximo: {maximos_clusters[cluster_id]}")
    print(f"  Mínimo: {minimos_clusters[cluster_id]}")
    print(f"  Promedio: {promedios_clusters[cluster_id]}")
    print(f"  Mediana: {medianas_clusters[cluster_id]}")
    print(f"  Desviación estándar: {desviaciones_clusters[cluster_id]}")
    print()

# Definir los nombres de los clusters
nombres_clusters = [f"Cluster {i}" for i in range(k_optimo)]

# Definir las posiciones en el eje x para las barras
posiciones_clusters = np.arange(k_optimo)

# Definir los anchos de las barras
ancho_barra = 0.15

# Calcular el promedio, mediana, desviación estándar, máximo y mínimo para cada cluster
promedios_clusters = [np.mean(datos[etiquetas_optimas == i]) for i in range(k_optimo)]
medianas_clusters = [np.median(datos[etiquetas_optimas == i]) for i in range(k_optimo)]
desviaciones_clusters = [np.std(datos[etiquetas_optimas == i]) for i in range(k_optimo)]
maximos_clusters = [np.max(datos[etiquetas_optimas == i]) for i in range(k_optimo)]
minimos_clusters = [np.min(datos[etiquetas_optimas == i]) for i in range(k_optimo)]

# Crear el gráfico de barras
plt.figure(figsize=(12, 8))

# Graficar el promedio
plt.bar(
    posiciones_clusters - 2 * ancho_barra,
    promedios_clusters,
    width=ancho_barra,
    label="Promedio",
)

# Graficar la mediana
plt.bar(
    posiciones_clusters - ancho_barra,
    medianas_clusters,
    width=ancho_barra,
    label="Mediana",
)

# Graficar la desviación estándar
plt.bar(
    posiciones_clusters,
    desviaciones_clusters,
    width=ancho_barra,
    label="Desviación Estándar",
)

# Graficar el máximo
plt.bar(
    posiciones_clusters + ancho_barra,
    maximos_clusters,
    width=ancho_barra,
    label="Máximo",
)

# Graficar el mínimo
plt.bar(
    posiciones_clusters + 2 * ancho_barra,
    minimos_clusters,
    width=ancho_barra,
    label="Mínimo",
)

# Agregar etiquetas en la parte superior de cada barra
for i in range(k_optimo):
    plt.text(
        posiciones_clusters[i] - 2 * ancho_barra,
        promedios_clusters[i] + 0.05,
        f"{promedios_clusters[i]:.2f}",
        ha="center",
    )
    plt.text(
        posiciones_clusters[i] - ancho_barra,
        medianas_clusters[i] + 0.05,
        f"{medianas_clusters[i]:.2f}",
        ha="center",
    )
    plt.text(
        posiciones_clusters[i],
        desviaciones_clusters[i] + 0.05,
        f"{desviaciones_clusters[i]:.2f}",
        ha="center",
    )
    plt.text(
        posiciones_clusters[i] + ancho_barra,
        maximos_clusters[i] + 0.05,
        f"{maximos_clusters[i]:.2f}",
        ha="center",
    )
    plt.text(
        posiciones_clusters[i] + 2 * ancho_barra,
        minimos_clusters[i] + 0.05,
        f"{minimos_clusters[i]:.2f}",
        ha="center",
    )

# Agregar etiquetas, título y leyenda
plt.xlabel("Clusters")
plt.ylabel("Valor")
plt.title("Estadísticas por cluster")
plt.xticks(posiciones_clusters, nombres_clusters)
plt.legend()

# Mostrar el gráfico
plt.tight_layout()
plt.savefig(
    f"estadisticas_por_cluster (min_area={MIN_AREA:.2f} - max_area={MAX_AREA:.2f}).png"
)


# validar si existe una carpeta llamada "dataset" y si no, crearla
# si ya existe se debe borrar el contenido de la carpeta
# usar shutil
import shutil

if os.path.exists("dataset"):
    shutil.rmtree("dataset")

# repartir los datos en 3 carpetas: "train", "test" y "val"
os.makedirs("dataset/train/images", exist_ok=True)
os.makedirs("dataset/train/labels", exist_ok=True)
os.makedirs("dataset/test/images", exist_ok=True)
os.makedirs("dataset/test/labels", exist_ok=True)
os.makedirs("dataset/val/images", exist_ok=True)
os.makedirs("dataset/val/labels", exist_ok=True)

# copiar el data.yaml a la carpeta "dataset"
shutil.copy("data.yaml", "dataset/data.yaml")


random.shuffle(datos_validos)
# repartir 85% de los datos a "train", 10% a "test" y 5% a "val" usando train_test_split


train, test = train_test_split(datos_validos, test_size=0.15)
test, val = train_test_split(test, test_size=0.33)

dataset = {"train": train, "test": test, "val": val}

# copiar las imágenes y los archivos de texto a las carpetas correspondientes
for folder in ["train", "test", "val"]:
    for data in dataset[folder]:
        image_file = data["image_file"]
        txt_file = data["txt_file"]
        image_name = (
            os.path.basename(txt_file).replace(".txt", ".jpg").split(os.sep)[-1]
        )
        label_name = os.path.basename(txt_file).split(os.sep)[-1]

        # copiar la imagen a la carpeta correspondiente
        cv2.imwrite(f"dataset/{folder}/images/{image_name}", image_file)
        #shutil.copy(image_file, f"dataset/{folder}/images/{image_name}")

        # crear un archivo de texto con las anotaciones de la imagen
        # si ya existe un archivo con el mismo nombre, se agrega el contenido al final
        with open(f"dataset/{folder}/labels/{label_name}", "a") as file:
            file.write(
                data["annotation_original"] + "\n"
            )  # agregar un salto de línea al final


print("Proceso terminado")
