from collections import Counter
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

df = pd.read_csv("analisis.csv")

MIN_AREA = 0.0
MAX_AREA = np.inf

areas = df["area"].values
datos = np.array(areas).reshape(-1, 1)


num_clusters_opciones = range(2, 8)
inercias = []
valores_silueta = []
# Probar diferentes opciones de número de clusters
for num_clusters in tqdm(num_clusters_opciones):
    # print(f"Probando con {num_clusters} clusters...")

    # Inicializar y ajustar el modelo KMeans
    kmeans = KMeans(n_clusters=num_clusters, n_init=10)
    kmeans.fit(datos)

    # Obtener las etiquetas de cluster y los centroides
    etiquetas = kmeans.labels_
    centroides = kmeans.cluster_centers_

    # Calcular la inercia y agregarla a la lista
    inercia = kmeans.inertia_
    inercias.append(inercia)
    valores_silueta.append(silhouette_score(datos, etiquetas))

plt.figure()
plt.plot(num_clusters_opciones, inercias, marker="o")
plt.xlabel("Número de Clusters")
plt.ylabel("Inercia")
plt.title("Ley del Codo")
plt.grid(True)

# Encontrar el codo (punto de inflexión)
deltas = np.diff(inercias, 2)  # Segunda derivada de las inercias
k_optimo = (
    num_clusters_opciones[np.argmax(deltas) + 1]  # El máximo después del mínimo
)
plt.axvline(x=k_optimo, color="red", linestyle="--", label="Número óptimo de clusters")
plt.legend()

plt.savefig(f"ley_del_codo (min_area={MIN_AREA:.2f} - max_area={MAX_AREA:.2f}).png")
# plt.show()

# Grafica la curva de la silueta
plt.figure()
plt.plot(num_clusters_opciones, valores_silueta, marker="o")
plt.xlabel("Número de clusters")
plt.ylabel("Valor de la silueta")
plt.title("Método de la Silueta")
plt.grid(True)
plt.savefig(f"silueta (min_area={MIN_AREA:.2f} - max_area={MAX_AREA:.2f}).png")
# plt.show()



plt.figure()


kmeans_optimo = KMeans(n_clusters=k_optimo, n_init=10)
kmeans_optimo.fit(datos)

etiquetas_optimas = kmeans_optimo.labels_
conteo_clusters = Counter(etiquetas_optimas)

datos_cluster_summary = {}

for cluster, conteo in conteo_clusters.items():
    datos_cluster_summary[cluster] = {
        "cantidad_datos": conteo,
    }

for cluster_id in range(k_optimo):
    datos_cluster = datos[etiquetas_optimas == cluster_id].flatten()

    datos_cluster_summary[cluster_id]["media"] = np.mean(datos_cluster)
    datos_cluster_summary[cluster_id]["maximo"] = np.max(datos_cluster)
    datos_cluster_summary[cluster_id]["minimo"] = np.min(datos_cluster)
    datos_cluster_summary[cluster_id]["promedio"] = np.average(datos_cluster)
    datos_cluster_summary[cluster_id]["mediana"] = np.median(datos_cluster)
    datos_cluster_summary[cluster_id]["desviacion"] = np.std(datos_cluster)
    datos_cluster_summary[cluster_id]["var"] = datos_cluster

posiciones_clusters = np.arange(k_optimo)
nombres_clusters = [f"Cluster {i}" for i in range(k_optimo)]

data = {}
for cluster_id in range(k_optimo):
    datos_cluster = datos[etiquetas_optimas == cluster_id].flatten()
    data[f"Grupo {cluster_id}"] = datos_cluster.tolist()

max_length = max(len(data[f"Grupo {i}"]) for i in range(k_optimo))
data = {
    key: value + [np.nan] * (max_length - len(value)) for key, value in data.items()
}

df_caja = pd.DataFrame(data)
ax = sns.boxplot(data=df_caja)

# Agrega los datos relevantes en texto
for i, column in enumerate(df_caja.columns):
    valores = df_caja[column].dropna()
    mediana = valores.median()
    maximo = valores.max()
    minimo = valores.min()
    ax.text(i, maximo, f'Máximo: {maximo}', ha='center', va='bottom', fontsize=8)
    ax.text(i, mediana, f'Mediana: {mediana}', ha='center', va='bottom', fontsize=8)
    ax.text(i, minimo, f'Mínimo: {minimo}', ha='center', va='top', fontsize=8)


plt.savefig(f'distribucion_cada_cluster_con_koptimo ({k_optimo}).png')

