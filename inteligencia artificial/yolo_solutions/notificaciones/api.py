import gradio as gr
import cv2
from ultralytics import solutions
from ultralytics import YOLO
import os
import tempfile


from_email = "wisrovi.rodriguez@gmail.com"  # the sender email address
password = "cwca qwho pkyu bnui"  # 16-digits password generated via: https://myaccount.google.com/apppasswords
to_email = "wisrovi.rodriguez@gmail.com"  # the receiver email address


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

    securityalarm = solutions.SecurityAlarm(
        show=False,  # display the output
        model=model_path_arg,  # i.e. yolo11s.pt, yolo11m.pt
        records=4,  # total detections count to send an email
        fps=fps_arg,  # ajustar la velocidad según los fotogramas por segundo
        classes=[0],  # estimate speed of specific classes (0: person, 2: car)
    )

    securityalarm.authenticate(
        from_email, password, to_email
    )  # authenticate the email server

    return securityalarm


# Define la función que procesará el video.
# Esta función encapsula la lógica de tu script original.
def process_with_solution(video_file):
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
    output_video_path = temp_output_file.name
    temp_output_file.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    if not video_writer.isOpened():
        cap.release()
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        raise gr.Error(
            "Error: No se pudo crear el archivo de video de salida. "
            "Asegúrate de que los códecs de video necesarios (ej. 'mp4v' o 'H.264') estén instalados. "
            "En un contenedor Docker, esto a menudo significa tener `ffmpeg`."
            "\nConsejo Dockerfile: `apt-get install -y ffmpeg`"
        )

    # Inicializar la solución de estimación de velocidad con el objeto YOLO ya cargado
    try:
        solution = create_solution(model_path_arg=yolo_model, fps_arg=fps)
        gr.Info("Procesando frames del video... Por favor espera.")
    except Exception as e:
        cap.release()
        video_writer.release()
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        raise gr.Error(f"Error al inicializar el estimador de VisionEye: {e}")

    try:
        frame_count = 0
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                break  # El video ha terminado o el frame está vacío

            # Procesar el frame con SpeedEstimator.
            # La instancia `solution` se llama directamente, lo que procesa el frame
            # y actualiza `solution.annotator.im0` con la imagen anotada.
            results = solution(im0)

            # Obtén la imagen con las anotaciones desde el atributo annotator
            annotated_frame = results.plot_im

            if (
                annotated_frame is not None
                and annotated_frame.shape[0] > 0
                and annotated_frame.shape[1] > 0
            ):
                video_writer.write(annotated_frame)
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
        if video_writer.isOpened():
            video_writer.release()
        if os.path.exists(output_video_path):
            os.remove(output_video_path)
        raise gr.Error(f"Ocurrió un error durante el procesamiento del video: {e}")
    finally:
        if cap.isOpened():
            cap.release()
        if video_writer.isOpened():
            video_writer.release()
        cv2.destroyAllWindows()

    gr.Info("Procesamiento de video completado.")
    return output_video_path


# --- Interfaz de Gradio (Usando gr.Blocks()) ---
with gr.Blocks(title="Notificacfión en tiempo real") as interface:
    gr.Markdown(
        """
        # 👁️ **Notificacfión en tiempo real** 🚀
        Sube un archivo de video para ver cómo la IA detecta y rastrea e identifica alertas de seguridad.
        
        Este sistema esta calibrado para detectar alertas de seguridad en videos de tráfico, como:
        * **Personas cruzando la carretera** sin precaución.
        * **Vehículos en sentido contrario** a la dirección permitida.
        * **Vehículos estacionados** en lugares no permitidos.
        * **Vehículos que pasan un semáforo en rojo**.
        * **Vehículos que exceden el límite de velocidad**.
        * **Vehículos que no respetan las señales de tráfico**.
        * **Vehículos que no respetan las líneas de la carretera**.
        * **Vehículos que no respetan las zonas peatonales**.
        * **Vehículos que no respetan las zonas de carga y descarga**.
        * **Vehículos que no respetan las zonas de parada**.
        * **Vehículos que no respetan las zonas de estacionamiento**.
        * **Vehículos que no respetan las zonas de acceso restringido**.
        * **Vehículos que no respetan las zonas de velocidad reducida**.
        * **Vehículos que no respetan las zonas de velocidad máxima**.
        * **Vehículos que no respetan las zonas de velocidad mínima**.
        * **Vehículos que no respetan las zonas de velocidad variable**.
        * **Vehículos que no respetan las zonas de velocidad constante**.
        
        Para efectos de demostración, el sistema está configurado para enviar un correo electrónico
        cuando se detecta una alerta de seguridad. Asegúrate de que la dirección de correo electrónico
        y la contraseña sean correctas en el código fuente antes de ejecutar la aplicación.

        Una alerta de seguridad se envía cuando se detecta un mínimo de 4 personas, pero se peude ajustar según las necesidades.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="Sube tu Video (MP4, AVI, etc.)", height=480)
            process_button = gr.Button("Procesar Video")
        with gr.Column(scale=1):
            output_video = gr.Video(
                label="Video Procesado con detección de alertas", height=480
            )

    # Conectar el botón a la función
    process_button.click(
        fn=process_with_solution, inputs=input_video, outputs=output_video
    )

    gr.Markdown(
        """
        ---
        **Consejos:**
        * Para mejores resultados, el video debe tener una **perspectiva fija** (cámara estática).
        * Unos ejemplos de videos de prueba:
        * [Video1](https://drive.google.com/file/d/1JvlsagtGRBh9miovA6wICXug2it2o1uI/view?usp=sharing)
        * [Video2](https://drive.google.com/file/d/1PXrc4WNWsTrlKuSJodFU-MqHk_czUDE8/view?usp=sharing)
        """
    )


# Lanza la interfaz de Gradio
if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, debug=True)
