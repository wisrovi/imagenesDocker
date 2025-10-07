import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans


df = pd.read_csv("analisis.csv")

# extraigo los nombres de class_label sin repetir
class_label = df["class_label"]
class_label = class_label.unique()


print(class_label)


plt.figure()
fig, axs = plt.subplots(2, 4, figsize=(24, 12))

all_medians = []
all_maximos = []
all_minimos = []
all_atipicos = []
for i, ax in enumerate(axs.flat):
    if i < len(class_label):
        clase = class_label[i]
        areas = df[df["class_label"] == clase]["area"]
        # sns.boxplot(data=areas, ax=ax)
        sns.violinplot(data=areas, ax=ax, cut=0)

        mediana = areas.median()
        maximo = areas.max()
        minimo = np.array([area for area in areas if area > 0]).min()

        # buscar los valores atípicos para determinar el punto de corte  y no considerarlos
        Q1 = areas.quantile(0.25)
        Q3 = areas.quantile(0.75)
        IQR = Q3 - Q1
        valores_atipicos = (areas < (Q1 - 1.5 * IQR)) | (areas > (Q3 + 1.5 * IQR))
        valores_atipicos_ordenados = np.sort(areas[valores_atipicos])
        punto_inicio_inferior = (
            valores_atipicos_ordenados[0]
            if valores_atipicos_ordenados.size > 0
            else None
        )
        all_atipicos.append(punto_inicio_inferior)

        ax.axhline(
            y=punto_inicio_inferior,
            color="black",
            linestyle="--",
            label=f"Punto inicio inferior {punto_inicio_inferior:.2f}",
        )

        ax.axhline(
            y=mediana, color="red", linestyle="--", label=f"Mediana {mediana:.2f}"
        )
        ax.axhline(
            y=maximo, color="green", linestyle="--", label=f"Máximo {maximo:.2f}"
        )
        ax.axhline(y=minimo, color="blue", linestyle="--", label=f"Mínimo {minimo:.2f}")

        # ax.set_xticklabels(["Área"])
        ax.set_title(clase)
        ax.legend()
        ax.grid(True)

        all_medians.append(mediana)
        all_maximos.append(maximo)
        all_minimos.append(minimo)
    else:
        fig.delaxes(ax)

plt.tight_layout()


min_mediana = np.array(all_medians).min()
max_maximo = np.array(all_maximos).max()
min_minimo = np.array(all_minimos).min()

plt.savefig("boxplot_areas_por_clase.png")


MIN_AREA = min_mediana - min_minimo
MAX_AREA = max(all_atipicos)

plt.figure(figsize=(12, 6))
sns.violinplot(
    data=df,
    y="class_label",
    x="area",
    hue="class_label",
    inner="point",
    cut=0,
)
plt.grid(True)
plt.axvline(
    x=MIN_AREA,
    color="green",
    linestyle="--",
    label=f"Area minima de corte minimo en {MIN_AREA:.2f}",
)
plt.axvline(
    x=MAX_AREA,
    color="red",
    linestyle="--",
    label=f"Area maxima de corte {MAX_AREA:.2f}",
)
plt.xlabel("identificacion distribucion de areas por clase y puntos de corte sugeridos")
plt.legend()
plt.title("Puntos de corte por clase de las areas")
plt.savefig("boxplot_areas_por_clase2.png")

# filtrar el dataframe por el rango de areas calculado
df = df[(df["area"] >= MIN_AREA) & (df["area"] <= MAX_AREA)]

# hacer un histograma de las clases
plt.figure()
clases_data = df["class_label"]
clases_data = clases_data.value_counts()
clases_data = clases_data.to_dict()


# crear histograma de la columna "class_label" del dataframe
plt.barh([nombre_clase for nombre_clase in clases_data.keys()], clases_data.values())

for i, v in enumerate(clases_data.values()):
    # poner los valores al final de la barra
    plt.text(v, i, str(v), ha="right", va="center")


plt.xlabel("Clase")
plt.ylabel("Cantidad de Muestras")
plt.title("Histograma de Clases (min_area={MIN_AREA:.2f} - max_area={MAX_AREA:.2f})")
plt.grid(True)
plt.savefig(f"histograma_clases_filtrada_por_area.png")




"""
    Clustering de las clases para elegir las que tienen mayor area 
    y filtrar el dataframe con esas clases para usarlas en el entrenamiento
"""

print(clases_data)

X = np.array(list(clases_data.values())).reshape(-1, 1)
kmeans_elegidos = KMeans(n_clusters=2, random_state=0)
kmeans_elegidos.fit(X)


# Etiquetas de cluster asignadas a cada dato
etiquetas = kmeans_elegidos.labels_

# Muestra las etiquetas asignadas a cada dato
maximo_en = max(X)
cluster_elegido = None
clases_por_etiqueta = {}
for etiqueta, (nombre, valor) in zip(etiquetas, clases_data.items()):
    if valor == maximo_en:
        cluster_elegido = etiqueta

    if etiqueta not in clases_por_etiqueta:
        clases_por_etiqueta[etiqueta] = []
    clases_por_etiqueta[etiqueta].append(nombre)

    # print(f"{nombre}: {valor} - Cluster: {etiqueta}")


print(
    f"La seleccion por cluster eligio las clases {clases_por_etiqueta[cluster_elegido]}"
)

df = df[df["class_label"].isin(clases_por_etiqueta[cluster_elegido])]
df.to_csv("analisis_filtrado.csv", index=False)
