import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


def leer_etiquetas_y_contar_clases(directorio):
    try:
        contador_clases = Counter()
        contenido_folder = os.listdir(directorio)
        for archivo in contenido_folder:
            if archivo.endswith(".txt"):
                ruta_archivo = os.path.join(directorio, archivo)
                with open(ruta_archivo, "r") as file:
                    lineas = file.readlines()
                    for linea in lineas:
                        try:
                            linea = linea.strip()
                            clase = int(linea[0])
                            contador_clases[clase] += 1
                        except:
                            pass
        return contador_clases, len(contenido_folder)
    except Exception as e:
        print(f"Error al leer las etiquetas en el directorio {directorio}: {e}")
        return Counter(), 0


# Ajusta estas rutas según la estructura de tu directorio
directorio_train = "train/labels"
directorio_valid = "val/labels"
directorio_test = "test/labels"

# Contar clases
contador_train, size_train = leer_etiquetas_y_contar_clases("train/labels")
contador_valid, size_valid = leer_etiquetas_y_contar_clases("val/labels")
contador_test, size_test = leer_etiquetas_y_contar_clases("test/labels")

# Sumar los contadores
contador_total = contador_train + contador_valid + contador_test
print(contador_total)


# Cargar los nombres de las clases desde data.yaml
try:
    ruta_yaml = "data.yaml"
    with open(ruta_yaml) as file:
        data_yaml = yaml.safe_load(file)
        nombres_clases = data_yaml["names"]
except:
    nombres_clases = list(range(len(contador_total.keys()) + 1))

    # crear data.yaml
    data_yaml = dict(
        train="train/images",
        val="val/images",
        nc=len(nombres_clases),
        names=nombres_clases,
    )
    with open(ruta_yaml, "w") as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=True)


plt.figure(figsize=(10, 6))
# Crear el histograma usando los nombres de las clases
plt.bar([nombres_clases[i] for i in contador_total.keys()], contador_total.values())
# poner encima de las barras el valor de cada una
for i, v in enumerate(contador_total.values()):
    plt.text(i, v + 3, str(v), ha="center", va="bottom")

plt.xlabel("Clase")
plt.ylabel("Cantidad de Muestras")
plt.title("Histograma de Muestras por Clase en Dataset de Entrenamiento")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# guardar el histograma
plt.savefig("histograma.png")
plt.show()
















plt.figure(figsize=(10, 6))
# Crear una lista para almacenar los datos
datos = []

# Llenar la lista con los datos
for i, nombre_clase in enumerate(nombres_clases):
    datos.append(
        {
            "Clase": nombre_clase,
            "Train": contador_train.get(i, 0),
            "Valid": contador_valid.get(i, 0),
            "Test": contador_test.get(i, 0),
        }
    )

# Crear el DataFrame a partir de la lista
df = pd.DataFrame(datos)

# Mostrar el DataFrame
print(df)

# guardar el DataFrame
df.to_csv("estadisticas.csv", index=False)

# crear un histograma de cuantas imagenes hay en train, test y val
# usar los datos de size_train, size_valid, size_test
plt.bar(["Train", "Valid", "Test"], [size_train, size_valid, size_test])

# poner los valores encima de la barra
for i, v in enumerate([size_train, size_test, size_valid]):
    plt.text(i, v + 3, str(v), ha="center", va="bottom")


plt.xlabel("Conjunto de Datos")
plt.ylabel("Cantidad de Muestras")
plt.title("Histograma de Muestras por Conjunto de Datos")
plt.tight_layout()

# guardar el histograma
plt.savefig("histograma_conjuntos.png")










"""
    Histograma por cantidad de datos por clase y por folder
"""



# Configurar el gráfico
plt.figure(figsize=(10, 6))

# Crear las barras para cada clase
for i, clase in enumerate(df["Clase"]):
    plt.bar(
        i,
        df.loc[i, "Train"],
        color="blue",
        label="Train (85%)" if i == 0 else None,
    )
    plt.bar(
        i,
        df.loc[i, "Valid"],
        bottom=df.loc[i, "Train"],
        color="orange",
        label="Valid (5%)" if i == 0 else None,
    )
    plt.bar(
        i,
        df.loc[i, "Test"],
        bottom=df.loc[i, "Train"] + df.loc[i, "Valid"],
        color="green",
        label="Test (10%)" if i == 0 else None,
    )

    # Añadir textos para cada barra y color
    total = df.loc[i, "Train"] + df.loc[i, "Valid"] + df.loc[i, "Test"]
    plt.text(
        i,
        (df.loc[i, "Train"] + df.loc[i, "Valid"] + df.loc[i, "Test"] / 2) + 20,
        str(total),
        ha="center",
        va="center",
        color="black",
    )

# Añadir etiquetas y leyenda
plt.xlabel("Cantidad de Datos")
plt.ylabel("Clase")
plt.title("Cantidad de Datos por Clase y Conjunto")
plt.legend()
plt.xticks(range(len(df["Clase"])), df["Clase"], rotation=45, ha="right")
plt.tight_layout()

# guardar el grafico
plt.savefig("histograma de clases por paquete")


# Mostrar el gráfico


plt.show()
