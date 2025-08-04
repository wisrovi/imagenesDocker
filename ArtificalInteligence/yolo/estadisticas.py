import glob
import os
import random

import pandas as pd
from library import *

validation_videos = "../data/2812223 Videos Validacion 1/"

all_videos = []

# Search for files with .mp4 extension
for file in glob.glob(os.path.join(validation_videos, "*.mp4")):
    all_videos.append(file)

# Search for files with .MOV extension
for file in glob.glob(os.path.join(validation_videos, "*.MOV")):
    all_videos.append(file)

print("Videos found: ", len(all_videos))
random.shuffle(all_videos)
print(all_videos)


class Tracking_damages(Yolo_mask, TrackingBase):
    def yolo_detector(self, frame):
        detections, frame_damage, results_bbox = self.detect(frame)
        return results_bbox, frame_damage, detections

    def execute(self, **kwargs: dict) -> dict:
        # run the YOLO model on the frame
        frame = kwargs["frame"]
        angle = kwargs.get("angle", 0)

        to_track = kwargs.get("to_track", False)
        self.to_track = to_track

        results_bbox, frame_damage, detections_mask = self.yolo_detector(frame)

        kwargs["frame_damage"] = frame_damage

        if to_track and len(results_bbox) > 0:
            damage_track_detected = self.tracking(
                frame, results_bbox, self.class_names_with_ids, angle
            )
            frame_damage_track = self.graph_track(frame, damage_track_detected)

            kwargs["frame_damage_track"] = frame_damage_track
            kwargs["damage_track_detected"] = damage_track_detected

            __results = []
            for result in results_bbox:
                bbox, acc, cls = result
                xmin, ymin, w, h = bbox
                xmax, ymax = xmin + w, ymin + h
                __results.append(
                    [
                        [xmin, ymin, xmax, ymax],
                        acc,
                        cls,
                    ]
                )

            kwargs["results"] = __results

        return kwargs

    def get_summary(self):
        return self.summary


# models
print("Loading models...")
model_damage = Tracking_damages("../models/knok/weights/best.pt")
model_parts = Yolo_mask("../models/parts/weights/best.pt")
modelo_orient = Yolo_mask2("../models/orient/weights/best.pt")


def bbox_to_polygon(bbox):
    x, y, w, h = bbox
    return Polygon([(x, y), (x + w, y), (x + w, y - h), (x, y - h)])


def porcentaje_interseccion(poligono1, poligono2):
    if not isinstance(poligono1, Polygon) and not isinstance(poligono2, Polygon):
        return 0

    if poligono1.intersects(poligono2):
        interseccion = poligono1.intersection(poligono2).area
        return (interseccion / poligono1.area) * 100 if poligono1.area > 0 else 0
    else:
        return 0


def expandir_y_unir(poligonos, porcentaje_expansion):
    poligonos_expandidos = [
        poligono.buffer(poligono.area * porcentaje_expansion / 100)
        for poligono in poligonos
    ]
    poligono_unido = unary_union(poligonos_expandidos)
    return poligono_unido


def poligono_a_puntos(poligono):
    if poligono.is_empty:
        return np.array([])
    exterior = np.array(list(poligono.exterior.coords))
    return exterior.reshape((-1, 1, 2)).astype(np.int32)


class EyesDcar:
    def __init__(self, model_damage, model_parts, modelo_orient):
        self.model_damage = model_damage
        self.model_parts = model_parts
        self.modelo_orient = modelo_orient


resultados = []
for video_path in all_videos:
    if not os.path.isfile(video_path):
        print(f"Video file {video_path} does not exist")
        continue

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    bodega_partes_golpes_por_angulo = {}

    for _ in tqdm(range(total_frames), desc="Procesando frames", unit="frame"):
        start = datetime.datetime.now()

        # Read a frame from the video
        success, frame = cap.read()

        if not success:
            break

        scale = 0.5
        frame_original = frame.copy()
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        if True:  # orient
            (
                frame_orient,  # frame filtered with principal car
                _,
                _,
                angle_orient,
            ) = modelo_orient.detector(frame)
            angle_orient = angle_orient * 15

        partes_golpes = {}

        if True:  # parts
            annotated_parts = model_parts.detect(frame)[1]
            PARTS, names_PARTS = model_parts.graph_poligons()[1:]

            # masked = model_parts.graph_poligons(PARTS)[0]

            # recortar en el area de la carroceria
            # porcentaje_expansion = 0.05
            # masked = np.zeros_like(frame)
            # for polygon in PARTS:
            #     distancia_buffer = polygon.length * porcentaje_expansion
            #     polygon = polygon.buffer(distancia_buffer)

            #     exterior_coords = np.array(polygon.exterior.coords, dtype=np.int32)
            #     cv2.fillPoly(masked, [exterior_coords], (255, 255, 255))
            # masked = cv2.bitwise_and(frame, masked)
            # frame_orient = masked

        if True:  # damages
            predicho = {"frame": frame_orient, "to_track": True, "angle": angle_orient}
            result_damage = model_damage.execute(**predicho)
            DAMAGEs_polig, name_DAMAGE = (
                model_damage.result_model_to_poligons(),
                model_damage.get_category_poligon(),
            )

            annotated_damages = result_damage.get("frame_damage", None)

        if DAMAGEs_polig is not None:
            partes_golpes["damages"] = {
                f"{name}_{id}": damage
                for damage, name, id in zip(
                    DAMAGEs_polig, name_DAMAGE, [i for i in range(len(name_DAMAGE))]
                )
            }
            partes_golpes["parts"] = {
                name: part for part, name in zip(PARTS, names_PARTS)
            }
            if angle_orient not in bodega_partes_golpes_por_angulo:
                bodega_partes_golpes_por_angulo[angle_orient] = partes_golpes

        if True:  # split damages for every part of damage touched
            if DAMAGEs_polig is not None:
                frame_damages = graph_damage_part(
                    DAMAGEs_polig, PARTS, frame_orient.copy()
                )
                # cv2.imshow("YOLOv8 frame_damages", frame_damages)

        end = datetime.datetime.now()
        fps_count = 1 / (end - start).total_seconds()

        cv2.putText(
            annotated_damages,
            f"FPS: {int(fps_count)}/{fps} - angle: {angle_orient}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        # cv2.imshow("YOLOv8 annotated_damages", annotated_damages)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for key, value in model_damage.get_summary().items():
        # frame = value["frame"] # for to save frame
        bbox = bbox_to_polygon(value["bbox"])
        name = value["name"]
        class_id = value["class_id"]
        angle = value["angle"]

        try:
            bodega_parts = bodega_partes_golpes_por_angulo[angle]["parts"]
            bodega_golpes = bodega_partes_golpes_por_angulo[angle]["damages"]
        except KeyError:
            continue

        # revisar porque algunos bboxes no se logran asociar a golpes aun cuando deberian
        id_mayor_porcentaje, mayor_porcentaje = -1, 0
        for j, golpe_este_angulo in bodega_golpes.items():
            porcentaje_intepceccion = porcentaje_interseccion(bbox, golpe_este_angulo)
            if porcentaje_intepceccion > mayor_porcentaje:
                mayor_porcentaje = porcentaje_intepceccion
                id_mayor_porcentaje = j

        if id_mayor_porcentaje == -1:
            golpe = bbox
        else:
            golpe = bodega_golpes[id_mayor_porcentaje]

        PORCENTAJE_MINIMA_AFECTACION = 5
        partes_afectadas = {}
        for nombre_parte, poligono_parte in bodega_parts.items():
            try:
                porcentaje_afectacion_parte = porcentaje_interseccion(
                    golpe, poligono_parte
                )
            except:
                continue

            if porcentaje_afectacion_parte > PORCENTAJE_MINIMA_AFECTACION:
                partes_afectadas[nombre_parte] = porcentaje_afectacion_parte

        for nombre_parte, porcentaje_afectacion_parte in partes_afectadas.items():
            resultados.append(
                {
                    "class_id": class_id,
                    "name": name,
                    "parte": nombre_parte,
                    "porcentaje": porcentaje_afectacion_parte,
                    "angle": angle,
                    "video": video_path,
                }
            )
        #     print(
        #         f"Name: {class_id}.{name} - Parte: {nombre_parte} - Porcentaje: {porcentaje_afectacion_parte} - Angulo: {angle} - Video: {video_path}"
        #     )
        # print()


resultados = pd.DataFrame(resultados)

# ordenar primero por video, luego por angulo
resultados = resultados.sort_values(by=["video", "angle"])

# guardar resultados
resultados.to_csv("resultados.csv", index=False)



import os
import pandas as pd


"""
        INVENTARIO
"""


def leer_inventario(path: str):
    # Cargando el archivo Excel
    inventario = pd.read_excel(path)

    # Agrupando por video, parte y tipo de golpe, y contando las ocurrencias
    inventario = (
        inventario.groupby(["Video", "Parte", "tipo daño"])
        .size()
        .reset_index(name="conteo")
    )

    # Reorganizando el DataFrame para tener una mejor visualización
    inventario = inventario.pivot_table(
        index=["Video", "Parte"], columns="tipo daño", values="conteo", fill_value=0
    )

    videos = {
        "Coche 1": "../data/2812223 Videos Validacion 1/20231116_103809.mp4",
        "Coche 2": "../data/2812223 Videos Validacion 1/20231116_105633.mp4",
        "Coche 3": "../data/2812223 Videos Validacion 1/IMG_4806.MOV",
        "Coche 4": "../data/2812223 Videos Validacion 1/IMG_4812.MOV",
        "Coche 5": "../data/2812223 Videos Validacion 1/IMG_4814.MOV",
        "Coche 6": "../data/2812223 Videos Validacion 1/IMG_4816.MOV",
        "Coche 7": "../data/2812223 Videos Validacion 1/IMG_4817.MOV",
        "Coche 8": "../data/2812223 Videos Validacion 1/IMG_4818.MOV",
        "Coche 9": "../data/2812223 Videos Validacion 1/IMG_4819.MOV",
        "Coche 10": "../data/2812223 Videos Validacion 1/j1.mp4",
        "Coche 11": "../data/2812223 Videos Validacion 1/j2.mp4",
        "Coche 12": "../data/2812223 Videos Validacion 1/j3.mp4",
    }

    inventario = inventario.reset_index()
    inventario["Video"] = inventario["Video"].map(videos)

    return inventario


inventario = leer_inventario(
    "../data/2812223 Videos Validacion 1/EtiquetadoFran/20231227 Inventario para estadisticas.xlsx"
)


"""
        PREDICHO
"""


def prepare_predichas(path: str):
    # Cargando el archivo CSV
    predicho = pd.read_csv(path)

    # renombrando columna name a tipo daño
    predicho = predicho.rename(columns={"name": "tipo daño"})

    # Agrupando por video, parte y tipo de golpe, y contando las ocurrencias
    predicho = (
        predicho.groupby(["Video", "Parte", "tipo daño"])
        .size()
        .reset_index(name="conteo")
    )

    # Reorganizando el DataFrame para tener una mejor visualización
    predicho = predicho.pivot_table(
        index=["Video", "Parte"], columns="tipo daño", values="conteo", fill_value=0
    )

    # Guardando los resultados en un archivo CSV
    predicho.to_csv("predicho.csv")
    predicho = pd.read_csv("predicho.csv")
    os.remove("predicho.csv")

    return predicho


predicho = prepare_predichas("resultados.csv")

# Identificando las columnas comunes entre ambos DataFrames
columnas_comunes = set(inventario.columns).intersection(set(predicho.columns))
columnas_comunes = [f for f in columnas_comunes]


# Filtrando los DataFrames para que solo contengan las columnas comunes
original_comun = inventario[columnas_comunes]
predicho_comun = predicho[columnas_comunes]

# Asegurándose de que ambos DataFrames estén alineados para la comparación
original_comun = original_comun.sort_values(by=["Video", "Parte"]).reset_index(
    drop=True
)
predicho_comun = predicho_comun.sort_values(by=["Video", "Parte"]).reset_index(
    drop=True
)


# Realizando la comparación por cada video
columnas_relevantes = [
    "faro roto",
    "roce",
    "pintura raspada",
    "bollo",
    "deformacion",
    "oxido",
]

matrix_confusion = {
    golpe: {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
    }
    for golpe in columnas_relevantes
}


for video in original_comun["Video"].unique():
    original_video = original_comun[original_comun["Video"] == video]
    predicho_video = predicho_comun[predicho_comun["Video"] == video]

    # borrar la columna Video
    original_video = original_video.drop(columns=["Video"])
    predicho_video = predicho_video.drop(columns=["Video"])

    # borrar el indice
    original_video = original_video.reset_index(drop=True)
    predicho_video = predicho_video.reset_index(drop=True)

    # guardar el original y el predicho en un archivo CSV
    original_video.to_csv(f"{video}_original.csv")
    predicho_video.to_csv(f"{video}_redicho.csv")

    for parte in original_video["Parte"].unique():
        original_video_parte = original_video[original_video["Parte"] == parte]
        predicho_video_parte = predicho_video[predicho_video["Parte"] == parte]

        for col in columnas_relevantes:
            conteo_original_parte_golpe = (
                int(original_video_parte[col].iloc[0])
                if len(original_video_parte.get(col, [])) > 0
                else 0
            )
            conteo_predicho_parte_golpe = (
                int(predicho_video_parte[col].iloc[0])
                if len(predicho_video_parte.get(col, [])) > 0
                else 0
            )

            if conteo_original_parte_golpe > conteo_predicho_parte_golpe:
                # 8 - 3
                TP = conteo_predicho_parte_golpe
                FP = 0
                TN = 0
                FN = conteo_original_parte_golpe - conteo_predicho_parte_golpe
            elif conteo_original_parte_golpe < conteo_predicho_parte_golpe:
                # 3 - 8
                TP = conteo_original_parte_golpe
                FP = conteo_predicho_parte_golpe - conteo_original_parte_golpe
                TN = 0
                FN = 0
            elif conteo_original_parte_golpe == conteo_predicho_parte_golpe:
                # 3 - 3
                TP = conteo_original_parte_golpe
                FP = 0
                TN = 0
                FN = 0

            matrix_confusion[col]["TP"] += TP
            matrix_confusion[col]["FP"] += FP
            matrix_confusion[col]["TN"] += TN
            matrix_confusion[col]["FN"] += FN


print(matrix_confusion)


metricas = {
    golpe: {
        "deteccion_global": 0,  # accuracy
        "acierto_en_inventario": 0,  # precision
    }
    for golpe in columnas_relevantes
}

for golpe, conteos in matrix_confusion.items():
    TP = conteos["TP"]
    FP = conteos["FP"]
    TN = conteos["TN"]
    FN = conteos["FN"]

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    metricas[golpe]["deteccion_global"] = f"{int(accuracy * 1000)/10}%"
    metricas[golpe]["acierto_en_inventario"] = f"{int(precision * 1000)/10}%"


metricas = pd.DataFrame.from_dict(metricas, orient="index")
matrix_confusion = pd.DataFrame.from_dict(matrix_confusion, orient="index")

print()
print(metricas)

TEMPLATE = "TEMPLATE_ESTADISTICAS.xlsx"
HOJA_METRICAS = "metricas"
HOJA_MATRIX_CONFUSION = "matrix_confusion"


with pd.ExcelWriter(
    TEMPLATE, engine="openpyxl", mode="a", if_sheet_exists="replace"
) as writer:
    matrix_confusion.to_excel(writer, sheet_name=HOJA_MATRIX_CONFUSION)
    metricas.to_excel(writer, sheet_name=HOJA_METRICAS)

    inventario.to_excel(writer, sheet_name="inventario")
