from glob import glob
import tempfile
import os
import zipfile
import argparse
import yaml
import ultralytics
import shutil
import datetime


from ultralytics import YOLO


def descomprimir_zip(archivo_zip, directorio_destino):
    try:
        with zipfile.ZipFile(archivo_zip, "r") as zip_ref:
            zip_ref.extractall(directorio_destino)
        print("¡Archivo ZIP descomprimido correctamente!")
    except zipfile.BadZipFile:
        print("Error: El archivo proporcionado no es un archivo ZIP válido.")
    except Exception as e:
        print(f"Error al descomprimir el archivo ZIP: {e}")


def leer_parametros():
    parser = argparse.ArgumentParser(
        description="Entrena un modelo de segmentación de YOLO"
    )
    parser.add_argument(
        "--dataset_yolo",
        type=str,
        default="20240208 damages_seg_eyesDcar.zip",
        help="Ruta del dataset YOLO",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="yolov8s-seg.pt",
        help="Modelo base para el entrenamiento",
    )

    args = parser.parse_args()

    # Accede a los argumentos pasados por línea de comandos
    yolo_dataset_path = args.dataset_yolo
    base_model = args.base_model

    return yolo_dataset_path, base_model


def get_date_time_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    DATASET, BASE_MODEL = leer_parametros()

    BASE = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        # copiar el modelo base a la carpeta temporal
        shutil.copy(BASE_MODEL, temp_dir)

        os.chdir(temp_dir)

        ultralytics.checks()
        PROYECTO = DATASET.split(" ")[-1].split(".")[0]
        YOLO_PATH = os.path.join(temp_dir, PROYECTO)
        ZIP_OUT = os.path.join(BASE, "results", "valid", f"{get_date_time_str()}_{PROYECTO}")
        os.makedirs(os.path.join(BASE, "results", "valid"), exist_ok=True)

        print(f"Descomprimiendo {DATASET} en {YOLO_PATH}")
        os.makedirs(YOLO_PATH, exist_ok=True)
        descomprimir_zip(os.path.join(BASE, DATASET), YOLO_PATH)

        # abrir el archivo data.yaml y cambiar la ruta de train y val
        with open(f"{YOLO_PATH}/data.yaml", "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            data["train"] = f"{YOLO_PATH}/train"
            data["val"] = f"{YOLO_PATH}/val/images"
        with open(f"{YOLO_PATH}/data.yaml", "w") as file:
            yaml.dump(data, file)

        model = YOLO(BASE_MODEL)

        # evaluar el modelo
        try:
            metrics = model.val(
                data=f"{YOLO_PATH}/data.yaml",
                plots=True,
                project=os.path.join("runs", PROYECTO),
            )
        except Exception as e:
            print(f"Error al evaluar el modelo: {e}")

        # la carpeta "runs" que esta en temp_dir se debe comprimir y mover a la carpeta de ejecución BASE
        print("Comprimiendo carpeta runs")
        os.chdir(BASE)

        carpeta_comprimir = os.path.join(temp_dir, "runs")
        shutil.make_archive(ZIP_OUT, "zip", carpeta_comprimir)
        print("Carpeta runs comprimida!!")


if __name__ == "__main__":
    main()

    # ejemplo de ejecución
    # python3 train.py <ruta_dataset_yolo>

    # python3 valid.py --dataset_yolo "20240208 damages_seg_eyesDcar_VALID.zip" --base_model best.pt
