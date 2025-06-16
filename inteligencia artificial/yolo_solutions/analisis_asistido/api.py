import gradio as gr
import cv2
from ultralytics import solutions
from ultralytics import YOLO
import os
import tempfile


# --- Configuración del modelo y rutas ---
# Asegúrate de que 'yolo11n.pt' esté en el mismo directorio
# o proporciona la ruta completa aquí.
model_path = "yolo11n.pt"


# --- Inicialización del estimador de velocidad ---
# El modelo debe cargarse una sola vez, fuera de la función de Gradio,
# para evitar recargarlo en cada inferencia.
# Sin embargo, SpeedEstimator requiere un 'model' (que es un objeto YOLO o ruta).
# Si `create_solution` se llama dentro de `speed_estimation_gradio_fn`,
# el modelo se cargará en cada llamada a la función, lo cual es ineficiente.
# Si el modelo se cargara directamente en `create_solution`,
# la solución óptima es instanciar SpeedEstimator fuera de la función.
# Pero la clase SpeedEstimator de Ultralytics (v8.2.19) espera la ruta del modelo
# o un objeto modelo YOLO directamente en su constructor.
# La forma actual en `create_solution` es válida, pero significa que el modelo
# se recarga para cada video. Para una app con mucho tráfico, esto no es ideal.
# Sin embargo, para simplicidad y evitar problemas de concurrencia o estado compartido,
# es aceptable en este contexto de un solo video por llamada.
# Si usaras un objeto `model` de YOLO ya cargado:
# model_yolo = YOLO(model_path) # Cargar solo una vez si SpeedEstimator acepta el objeto


# --- Inicialización del estimador de velocidad ---
# El modelo debe cargarse una sola vez, fuera de la función de Gradio,
# para evitar recargarlo en cada inferencia si SpeedEstimator pudiera aceptarlo así.
# Sin embargo, SpeedEstimator de Ultralytics (v8.2.19) espera la ruta del modelo
# o un objeto modelo YOLO directamente en su constructor, y lo carga internamente.
# Para evitar la recarga en cada llamada a `speed_estimation_gradio_fn`,
# podemos cargar el modelo YOLO una vez y pasarlo al SpeedEstimator.
try:
    # Cargar el modelo una vez al inicio del script
    # Esto también manejará la descarga si el modelo no existe.
    yolo_model = YOLO(model_path)
    print(f"Modelo YOLO '{model_path}' cargado exitosamente.")
except Exception as e:
    print(f"Error al cargar el modelo YOLO '{model_path}': {e}")
    print("Asegúrate de que el nombre del modelo sea correcto (ej. 'yolov8n.pt').")
    # Si el modelo no se puede cargar, la aplicación no funcionará correctamente.
    # Podrías considerar salir o manejar esto de otra manera.
    yolo_model = None  # Asegúrate de que yolo_model sea None si falla la carga inicial


def create_solution(model_path_arg, fps_arg):
    """
    Crea y configura una instancia de SpeedEstimator.
    Nota: El modelo se carga internamente por SpeedEstimator cada vez que se crea una solución.
    """
    # Initialize analytics object
    analytics = solutions.Analytics(
        show=True,  # display the output
        analytics_type="line",  # pass the analytics type, could be "pie", "bar" or "area".
        model=model_path_arg,  # path to the YOLO11 model file
        verbose=False,  # display analytics in console
        # classes=[0, 2],  # display analytics for specific detection classes
    )

    # Initialize object counter object
    region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # rectangle region
    counter = solutions.ObjectCounter(
        show=True,  # display the output
        region=region_points,  # pass region points
        fps=fps_arg,  # adjust speed based on frame per second
        verbose=False,  # display analytics in console
        model=model_path_arg,  # model="yolo11n-obb.pt" for object counting with OBB model.
        # classes=[0, 2],  # count specific classes i.e. person and car with COCO pretrained model.
        # tracker="botsort.yaml",  # choose trackers i.e "bytetrack.yaml"
    )
    return analytics, counter


# Define la función que procesará el video.
# Esta función encapsula la lógica de tu script original.
def speed_estimation_gradio_fn(video_file):
    """
    Procesa un archivo de video para estimar la velocidad de los objetos
    usando un modelo YOLO y devuelve la ruta al video procesado.
    Args:
        video_file: El objeto de archivo de video de Gradio (gr.File o gr.Video).
    Returns:
        str: La ruta al archivo de video procesado.
    Raises:
        gr.Error: Si ocurre un error durante el procesamiento.
    """
    global yolo_model  # Acceder al modelo YOLO cargado globalmente

    if yolo_model is None:
        raise gr.Error("Error: El modelo YOLO no se cargó correctamente al inicio.")

    video_path = video_file
    if not video_path:
        raise gr.Error("Error: No se proporcionó ningún archivo de video.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error(
            f"Error: No se pudo abrir el archivo de video '{video_path}'. "
            "Verifica que el archivo sea válido y que los códecs estén instalados."
        )

    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )

    if fps <= 0:
        fps = 30
        print(
            f"Advertencia: FPS del video no válido o cero. Usando FPS por defecto: {fps}"
        )

    # Es recomendable ajustar 'reg_pts' y 'meter_per_pixel' para tu caso de uso específico.
    # Aquí, `reg_pts` se pasa en la inicialización de `create_solution`.
    # Puedes exponer estos parámetros en la interfaz de Gradio para que el usuario los ajuste.
    # Por ejemplo, una línea horizontal en el centro del video:
    # reg_pts=((0, h // 2), (w, h // 2))

    temp_output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    analitics_output_video_path = temp_output_file.name
    counted_output_video_path = temp_output_file.name.replace(".mp4", "_counter.mp4")
    temp_output_file.close()

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )  # 'mp4v' for .mp4, more compatible than 'avc1'
    video_analytics_writer = cv2.VideoWriter(
        analitics_output_video_path, fourcc, fps, (w, h)
    )
    video_counter_writer = cv2.VideoWriter(
        counted_output_video_path, fourcc, fps, (w, h)
    )

    if not video_analytics_writer.isOpened() or not video_counter_writer.isOpened():
        cap.release()
        if os.path.exists(analitics_output_video_path):
            os.remove(analitics_output_video_path)
        raise gr.Error(
            "Error: No se pudo crear el archivo de video de salida. "
            "Asegúrate de que los códecs de video necesarios (ej. 'mp4v' o 'H.264') estén instalados. "
            "En un contenedor Docker, esto a menudo significa tener `ffmpeg`."
            "\nConsejo Dockerfile: `apt-get install -y ffmpeg`"
        )

    # Inicializar la solución de estimación de velocidad con el objeto YOLO ya cargado
    try:
        analytics, counter = create_solution(model_path_arg=yolo_model, fps_arg=fps)
        gr.Info("Procesando frames del video... Por favor espera.")
    except Exception as e:
        cap.release()
        video_analytics_writer.release()
        video_counter_writer.release()
        if os.path.exists(analitics_output_video_path):
            os.remove(analitics_output_video_path)
        raise gr.Error(f"Error al inicializar el estimador de velocidad: {e}")

    try:
        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break  # El video ha terminado o el frame está vacío

            # Procesar el frame con SpeedEstimator.
            # La instancia `solution` se llama directamente, lo que procesa el frame
            # y actualiza `solution.annotator.im0` con la imagen anotada.
            analytics_results = analytics(im0, frame_count)
            analytics_annotated_frame = analytics_results.plot_im

            if (
                analytics_annotated_frame is not None
                and analytics_annotated_frame.shape[0] > 0
                and analytics_annotated_frame.shape[1] > 0
            ):
                video_analytics_writer.write(analytics_annotated_frame)
            else:
                print(
                    f"Advertencia: Frame procesado ({cap.get(cv2.CAP_PROP_POS_FRAMES)}) es inválido. Saltando."
                )

            counter_results = counter(im0)
            counter_annotated_frame = counter_results.plot_im

            if (
                counter_annotated_frame is not None
                and counter_annotated_frame.shape[0] > 0
                and counter_annotated_frame.shape[1] > 0
            ):
                video_counter_writer.write(counter_annotated_frame)
            else:
                print(
                    f"Advertencia: Frame procesado ({cap.get(cv2.CAP_PROP_POS_FRAMES)}) es inválido. Saltando."
                )

            frame_count += 1
            # Actualizar el progreso en Gradio (opcional, si hay soporte para ello o para depuración)
            # gr.Info(f"Procesando frame {frame_count}...")

    except Exception as e:
        print(f"Ocurrió un error durante el procesamiento del video: {e}")
        # Asegura la liberación de recursos y la limpieza del archivo temporal
        if cap.isOpened():
            cap.release()
        if video_analytics_writer.isOpened():
            video_analytics_writer.release()
        if video_counter_writer.isOpened():
            video_counter_writer.release()
        if os.path.exists(analitics_output_video_path):
            os.remove(analitics_output_video_path)
        raise gr.Error(f"Ocurrió un error durante el procesamiento del video: {e}")
    finally:
        if cap.isOpened():
            cap.release()
        if video_analytics_writer.isOpened():
            video_analytics_writer.release()
        if video_counter_writer.isOpened():
            video_counter_writer.release()
        cv2.destroyAllWindows()

    gr.Info("Procesamiento de video completado.")
    return analitics_output_video_path, counted_output_video_path


# --- Interfaz de Gradio (Usando gr.Blocks()) ---
with gr.Blocks(title="Analisis asistido usando IA") as interface:
    gr.Markdown(
        """
        # 🏎️ **Análisis Asistido usando IA** 🚀
        Sube un archivo de video para detectar objetos.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="Sube tu Video (MP4, AVI, etc.)", height=480)
            process_button = gr.Button("Procesar Video")
        with gr.Column(scale=1):
            analitics_output_video = gr.Video(
                label="Video Procesado con Estimación de Velocidad", height=480
            )
            counter_output_video = gr.Video(
                label="Video Procesado con Conteo de Objetos", height=480
            )

    # Conectar el botón a la función
    process_button.click(
        fn=speed_estimation_gradio_fn,
        inputs=input_video,
        outputs=(
            analitics_output_video,
            counter_output_video,
        ),
    )

    gr.Markdown(
        """
        ---
        **Consejos:**
        * Para mejores resultados, el video debe tener una **perspectiva fija** (cámara estática).
        * Unos ejemplos de videos de prueba:
        * [Autovia 1](https://drive.google.com/file/d/1YLV0bvvLwcsCyBshFw-w0jmRXaTNdAtY/view?usp=sharing)
        * [Autovia 2](https://drive.google.com/file/d/1PXrc4WNWsTrlKuSJodFU-MqHk_czUDE8/view?usp=sharing)
        """
    )


# Lanza la interfaz de Gradio
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, debug=True)
