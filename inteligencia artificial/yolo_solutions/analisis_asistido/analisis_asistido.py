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
    default="./media/analytics_output",
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


out = cv2.VideoWriter(
    f"{output_path}.avi",
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (1280, 720),  # this is fixed
)

out2 = cv2.VideoWriter(
    f"{output_path}2.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
)

# Initialize analytics object
analytics = solutions.Analytics(
    show=True,  # display the output
    analytics_type="line",  # pass the analytics type, could be "pie", "bar" or "area".
    model=model_path,  # path to the YOLO11 model file
    # classes=[0, 2],  # display analytics for specific detection classes
)

# Initialize object counter object
region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # rectangle region
counter = solutions.ObjectCounter(
    show=True,  # display the output
    region=region_points,  # pass region points
    model="yolo11n.pt",  # model="yolo11n-obb.pt" for object counting with OBB model.
    # classes=[0, 2],  # count specific classes i.e. person and car with COCO pretrained model.
    # tracker="botsort.yaml",  # choose trackers i.e "bytetrack.yaml"
)

# Process video
frame_count = 0
while cap.isOpened():
    success, im0 = cap.read()
    if success:
        frame_count += 1
        results = analytics(im0, frame_count)  # update analytics graph every frame

        # print(results)  # access the output

        out.write(results.plot_im)  # write the video file

        results = counter(im0)
        out2.write(results.plot_im)  # write the processed frame.

    else:
        break

cap.release()
out.release()
out2.release()
cv2.destroyAllWindows()  # destroy all opened windows
