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


def get_date_time_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


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
        "--epocas",
        type=int,
        default=1,
        help="Número de epocas para el entrenamiento",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=-1,
        help="Tamaño del batch para el entrenamiento, número de imágenes por lote (-1 para AutoLote)",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="yolov8s-seg.pt",
        help="Modelo base para el entrenamiento",
    )
    parser.add_argument(
        "--horas",
        type=int,
        default=None,
        help="Tiempo máximo en horas para el entrenamiento, esto anula las épocas si se proporciona",
    )

    args = parser.parse_args()

    # Accede a los argumentos pasados por línea de comandos
    yolo_dataset_path = args.dataset_yolo
    epocas = args.epocas
    batch = args.batch
    base_model = args.base_model
    horas = args.horas

    return yolo_dataset_path, epocas, batch, base_model, horas


def main():
    DATASET, EPOCH, BATCH, BASE_MODEL, HORAS = leer_parametros()
    if not os.path.exists(DATASET):
        print(f"Error: El directorio {DATASET} no existe.")
        return

    BASE = os.getcwd()

    # Crear un directorio results dentro de BASE para guardar los resultados
    os.makedirs(os.path.join(BASE, "results"), exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        # moverse de carpeta de ejecucion a la carpeta temporal
        if os.path.exists(BASE_MODEL):
            shutil.copy(BASE_MODEL, temp_dir)

        FOLDER_TRAINED = os.path.join(temp_dir, "runs")
        PROYECTO = DATASET.split(" ")[-1].split(".")[0]
        YOLO_PATH = os.path.join(temp_dir, PROYECTO)
        ZIP_OUT = os.path.join(
            BASE, "results", "train", f"{get_date_time_str()}_{PROYECTO}"
        )

        os.chdir(temp_dir)
        ultralytics.checks()

        def preparar_dataset():
            # descomprimir el archivo ZIP en la carpeta temporal
            print(f"Descomprimiendo {DATASET} en {YOLO_PATH}")
            os.makedirs(YOLO_PATH, exist_ok=True)
            descomprimir_zip(os.path.join(BASE, DATASET), YOLO_PATH)

            # abrir el archivo data.yaml y cambiar la ruta de train y val
            with open(f"{YOLO_PATH}/data.yaml", "r") as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
                data["train"] = f"{YOLO_PATH}/train"
                data["val"] = f"{YOLO_PATH}/val"
            with open(f"{YOLO_PATH}/data.yaml", "w") as file:
                yaml.dump(data, file)

            return glob(f"{YOLO_PATH}/val/images/*.jpg")

        imagenes_para_predecir = preparar_dataset()

        model = YOLO(BASE_MODEL)

        results = model.train(
            data=f"{YOLO_PATH}/data.yaml",
            epochs=EPOCH,
            imgsz=640,
            batch=BATCH,
            time=(
                HORAS if HORAS is not None and HORAS > 0 else None
            ),  # número de horas para entrenar, anula las épocas si se proporciona
            save=True,  # guardar los puntos de control del tren y predecir los resultados
            project=os.path.join("runs", PROYECTO),
            # name=NAME,
            warmup_epochs=EPOCH * 0.15,  # épocas de calentamiento, 10% del epochs
            plots=True,  # guardar gráficos e imágenes durante el entrenamiento/val
        )

        try:
            metrics = model.val(
                data=f"{YOLO_PATH}/data.yaml",
                plots=True,
                save_json=True,
                save_hybrid=True,
            )
        except Exception as e:
            print(f"Error al evaluar el modelo: {e}")

        # Predict with the model

        try:
            results_predict = model.predict(
                imagenes_para_predecir,
                # # Inference arguments:
                # #conf=0.20,
                # half=True,
                # #visualize=True,
                retina_masks=True,
                # # Visualization arguments:
                save=True,
                # save_txt=True,
                # save_conf=True,
                # save_crop=True,
            )  # predict on an image
        except Exception as e:
            print(f"Error al predecir con el modelo: {e}")

        # print(results)
        # print(metrics)
        # print(results_predict)

        print("Entrenamiento finalizado")
        print(os.listdir(YOLO_PATH))
        print(os.listdir(temp_dir))

        shutil.make_archive(ZIP_OUT, "zip", FOLDER_TRAINED)
        print("Carpeta runs comprimida!!")

        os.chdir(BASE)


if __name__ == "__main__":
    main()

    # ejemplo de ejecución
    # python train.py <ruta_dataset_yolo>

    # python3 train.py --dataset_yolo "20240208 damages_seg_eyesDcar.zip" --epocas 1 --batch 1 --base_model yolov8s-seg.pt
    # python3 train.py --dataset_yolo "20240208_3 damages_seg_eyesDcar.zip" --epocas 150 --batch -1 --base_model yolov8s-seg.pt --horas 12
