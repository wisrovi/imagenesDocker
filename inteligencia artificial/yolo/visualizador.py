import os
import cv2
import numpy as np
import yaml

BASE_PATH = "./"

PROYECTO = "damages-car-1/"
PROYECTO = "damages-car-4/"
# PROYECTO = "dent-1/"
# PROYECTO = "dent-detection-1/"
# PROYECTO = "proj-1/"

# Ruta al archivo data.yaml
data_yaml_path = BASE_PATH + PROYECTO + "data.yaml"

# Ruta al directorio que contiene las imágenes
images_dir = BASE_PATH + PROYECTO + "train/images/"

# Ruta al directorio que contiene las anotaciones YOLO en formato txt
annotations_dir = BASE_PATH + PROYECTO + "train/labels/"

# Cargar el archivo data.yaml
with open(data_yaml_path, "r") as file:
    data = yaml.load(file, Loader=yaml.FullLoader)

# Obtener los nombres de las etiquetas
class_names = data["names"]

# Leer las anotaciones y visualizar las imágenes
for txt_file in os.listdir(annotations_dir):
    # Construir la ruta completa al archivo de anotaciones
    txt_file_path = os.path.join(annotations_dir, txt_file)

    # Leer las líneas de anotación del archivo
    with open(txt_file_path, "r") as file:
        annotations = file.readlines()

    # Construir la ruta completa al archivo de imagen
    image_path = os.path.join(images_dir, txt_file.replace(".txt", ".jpg"))

    # Leer la imagen
    image = cv2.imread(image_path)
    original_image = image.copy()

    height, width, _ = image.shape

    class_to_ignore = []

    # Iterar sobre cada anotación y dibujar la segmentación en la imagen
    for annotation in annotations:
        annotation = annotation.strip().split()
        class_index = int(annotation[0])

        segmentation_points = [float(f) for f in map(float, annotation[1:])]

        # Convertir la lista de puntos a un array de tipo numpy
        points_array = np.array(segmentation_points, np.float32)
        # Reshape el array para que tenga filas y columnas
        points_array_normalized = points_array.reshape((-1, 2))

        points_array = (points_array_normalized * np.array([width, height])).astype(
            np.int32
        )

        # Obtener la clase de la etiqueta
        class_label = class_names[class_index]

        if class_label in class_to_ignore:
            continue

        # Dibujar la segmentación en la imagen
        cv2.polylines(
            image, [points_array], isClosed=True, color=(0, 255, 0), thickness=2
        )

        # Mostrar la clase cerca de la primera coordenada de la segmentación
        cv2.putText(
            image,
            class_label,
            tuple(points_array[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        # breakpoint()

    # Mostrar la imagen con las segmentaciones y etiquetas
    cv2.imshow("YOLO Segmentation Visualization", image)

    # Esperar a que el usuario presione una tecla
    key_user = cv2.waitKey(0)
    if key_user & 0xFF == ord("q"):
        break
    elif key_user & 0xFF == ord("s"):
        print("Guardando imagen...")
        # crear un nuevo folder para almacenar las imagenes a guardar
        os.makedirs(BASE_PATH + "Final_images/images/", exist_ok=True)
        # guardar la imagen
        cv2.imwrite(
            BASE_PATH + "Final_images/images/" + txt_file.replace(".txt", ".jpg"),
            original_image,
        )

        # crear un nuevo folder para almacenar las anotaciones a guardar
        os.makedirs(BASE_PATH + "Final_images/labels/", exist_ok=True)
        # guardar las anotaciones
        with open(BASE_PATH + "Final_images/labels/" + txt_file, "w") as file:
            file.writelines(annotations)
    elif key_user & 0xFF == ord("n"):
        print("No guardando imagen...")
        # borrar la imagen
        os.remove(image_path)
        # borrar la anotacion
        os.remove(txt_file_path)

# Cerrar la ventana al finalizar
cv2.destroyAllWindows()
