import cv2

from ultralytics import solutions

import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="Speed Estimation from Video")
parser.add_argument(
    "--model",
    type=str,
    default="yolo11n.pt",
    help="Path to the YOLO11 model file.",
)
parser.add_argument(
    "--video",
    type=str,
    default="AutoviaDireccionContraria.mkv",
    help="Path to the input video file.",
)
parser.add_argument(
    "--output",
    type=str,
    default="./media/region_counting",
    help="Path to the output video file.",
)

args = parser.parse_args()

model_path = args.model
video_path = args.video
output_path = args.output


cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"
width, height = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
)

# Pass region as list
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

# crear dos regiones, dividimos la pantalla en cuatro cuadrantes
# la primera region es la parte superior izquierda y la segunda region es la parte inferior derecha
# considerar el tamaño de la pantalla, por ejemplo, 1080x720 (width x height), calculando los puntos de las regiones
# de acuerdo a la resolucion de la pantalla


center_x = width // 2
center_y = height // 2


# Pass region as dictionary
region_points = {
    "region-01": [
        (50, 50),
        (center_x - 50, 50),
        (center_x - 50, center_y - 50),
        (50, center_y - 50),
    ],
    "region-02": [
        (center_x + 50, center_y + 50),
        (center_x + 50, center_y - 50),
        (width - 50, height + 50),
        (width - 50, height - 50),
    ],
}

# Video writer
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)
video_writer = cv2.VideoWriter(
    f"{output_path}.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Initialize region counter object
regioncounter = solutions.RegionCounter(
    show=True,  # display the frame
    region=region_points,  # pass region points
    model=model_path,  # model for counting in regions i.e yolo11s.pt
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = regioncounter(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows
