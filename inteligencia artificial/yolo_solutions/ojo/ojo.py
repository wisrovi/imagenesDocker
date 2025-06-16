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
    default="./media/ojo",
    help="Path to the output video file.",
)

args = parser.parse_args()

model_path = args.model
video_path = args.video
output_path = args.output


cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)
video_writer = cv2.VideoWriter(
    f"{output_path}.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Init vision eye object
visioneye = solutions.VisionEye(
    show=True,  # display the output
    model=model_path,  # use any model that Ultralytics support, i.e, YOLOv10
    # classes=[0, 2],  # generate visioneye view for specific classes
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break

    results = visioneye(im0)

    print(results)  # access the output

    video_writer.write(results.plot_im)  # write the video file

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows
