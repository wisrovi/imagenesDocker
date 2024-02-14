import glob
import cv2
import os
import yaml
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon

# Definir constantes
MIN_AREA = 5000
MAX_AREA = 2000000
REDUCCION = 0.25


def cargar_nombres_clase(data_yaml_path):
    with open(data_yaml_path, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        return data.get("names", [])


def obtener_colores_clase(class_names):
    colores = {}
    for i, class_name in enumerate(class_names):
        # Generar un color único para cada clase
        np.random.seed(i)  # Semilla para reproducibilidad
        colores[class_name] = (
            np.random.randint(0, 256),
            np.random.randint(0, 256),
            np.random.randint(0, 256),
        )
    return colores


def procesar_anotaciones(labels, images, class_names, colores):
    for label_file in tqdm(labels):
        with open(label_file, "r") as file:
            annotations = file.readlines()

        if not annotations:
            print(f"No se encontraron anotaciones en el archivo {label_file}")
            continue

        image_name = os.path.basename(label_file).replace(".txt", ".jpg")
        image_path = next((img for img in images if image_name in img), None)
        if not image_path:
            print(f"No se encontró la imagen correspondiente al archivo {label_file}")
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"No se pudo leer la imagen: {image_path}")
            continue

        height, width, _ = image.shape

        count = 0
        for annotation in annotations:
            try:
                class_index, *segmentation_points = map(
                    float, annotation.strip().split()
                )
            except Exception as e:
                print(e)
                continue

            try:
                class_label = class_names[int(class_index)]
            except:
                print(class_index)
                continue

            count += 1

            # Convertir puntos de segmentación a un array numpy
            points_array = np.array(segmentation_points, np.float32)
            points_array_normalized = points_array.reshape((-1, 2))
            points_array = (points_array_normalized * np.array([width, height])).astype(
                np.int32
            )

            # Crear un polígono con los puntos de la segmentación
            house = Polygon(points_array)
            points = np.array(house.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                image, [points], isClosed=True, color=colores[class_label], thickness=2
            )

            # Colocar el nombre de la clase sobre el polígono
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            thickness = 8
            text_size = cv2.getTextSize(class_label, font, font_scale, thickness)[0]
            text_x = (
                points[0][0][0] - text_size[0] if points[0][0][0] >= text_size[0] else 0
            )
            text_y = (
                points[0][0][1] - text_size[1]
                if points[0][0][1] >= text_size[1]
                else text_size[1]
            )
            cv2.putText(
                image,
                class_label,
                (text_x, text_y),
                font,
                font_scale,
                colores[class_label],
                thickness,
            )

        # Redimensionar la imagen al 25%
        resized_image = cv2.resize(
            image, (int(width * REDUCCION), int(height * REDUCCION))
        )

        if count > 0:
            cv2.imshow("Image", resized_image)

            # Esperar a que el usuario oprima "q" para continuar con la siguiente imagen
            key = cv2.waitKey(0)
            if key == ord("q"):
                break

    cv2.destroyAllWindows()


def main():
    data_yaml_path = "data.yaml"
    images_train = glob.glob("train/images/*.jpg")
    images_test = glob.glob("test/images/*.jpg")
    images_valid = glob.glob("val/images/*.jpg")
    labels_train = glob.glob("train/labels/*.txt")
    labels_test = glob.glob("test/labels/*.txt")
    labels_valid = glob.glob("val/labels/*.txt")
    images = images_train + images_test + images_valid
    labels = labels_train + labels_test + labels_valid

    class_names = cargar_nombres_clase(data_yaml_path)
    if not class_names:
        print("No se encontraron nombres de clases en el archivo data.yaml")
        return

    colores = obtener_colores_clase(class_names)

    procesar_anotaciones(labels, images, class_names, colores)


if __name__ == "__main__":

    main()
