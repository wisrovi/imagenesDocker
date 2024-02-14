import os
import argparse


CAMBIOS = None


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
    parser.add_argument("--class_input", type=int, help="Clase a cambiar (input)")
    parser.add_argument("--class_output", type=int, help="Clase a cambiar (output)")

    args = parser.parse_args()

    # Accede a los argumentos pasados por línea de comandos
    yolo_dataset_path = args.dataset_yolo
    class_input = args.class_input
    class_output = args.class_output

    return yolo_dataset_path, class_input, class_output


# Función para cambiar las clases en un archivo de texto
def cambiar_clases_en_archivo(ruta_archivo):
    with open(ruta_archivo, "r") as archivo:
        lineas = archivo.readlines()

    print("\t" * 2, "Cantidad de registros:", len(lineas))

    with open(ruta_archivo, "w") as archivo:
        for linea in lineas:
            # Dividir la línea en partes
            partes = linea.strip().split()

            if len(partes) > 0:
                clase_leida = int(partes[0])

                if clase_leida == CAMBIOS[0]:
                    partes[0] = str(CAMBIOS[1])
                    print("\t" * 3, clase_leida, CAMBIOS[0], " -> cambio")

            # Escribir la línea modificada en el archivo
            archivo.write(" ".join(partes) + "\n")


# Función para recorrer una carpeta y cambiar las clases en todos los archivos de texto
def cambiar_clases_en_carpeta(ruta_carpeta):
    ruta_carpeta = os.path.join(ruta_carpeta, "labels")
    print(ruta_carpeta)
    for archivo in os.listdir(ruta_carpeta):
        if archivo.endswith(".txt"):
            ruta_archivo = os.path.join(ruta_carpeta, archivo)
            print("\t", ruta_archivo)

            cambiar_clases_en_archivo(ruta_archivo)


# recorrer las carpetas train, test y valid
# en cada carpeta labels estan los txt con las detecciones de segmentacion (clases y coordenadas)

# necesito cambiar todas las clases de 0 a 1
# las coordenadas se dejan igual

CAMBIOS = ("7", "1")
DATASET = "./Car-Damage-V5-6/"


# Rutas de las carpetas train, test y valid
carpetas = ["train", "test", "val"]


def main():
    global CAMBIOS

    DATASET, INPUT, OUTPUT = leer_parametros()
    CAMBIOS = (INPUT, OUTPUT)

    # Cambiar clases en cada carpeta
    for carpeta in carpetas:
        ruta_carpeta = os.path.join(DATASET, carpeta)
        try:
            cambiar_clases_en_carpeta(ruta_carpeta)
        except FileNotFoundError:
            print(f"No se encontró la carpeta {ruta_carpeta}")


if __name__ == "__main__":
    main()
    
    # ejemplo:
    
    # python3 redefinir-classes.py  --dataset_yolo "./" --class_input 0 --class_output 50

