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
    default="./media/speed_management",
    help="Path to the output video file.",
)

args = parser.parse_args()

model_path = args.model
video_path = args.video
output_path = args.output


cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(f"{output_path}.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

from_email = "wisrovi.rodriguez@gmail.com"  # the sender email address
password = "cwca qwho pkyu bnui"  # 16-digits password generated via: https://myaccount.google.com/apppasswords
to_email = "wisrovi.rodriguez@gmail.com"  # the receiver email address

# Initialize security alarm object
securityalarm = solutions.SecurityAlarm(
    show=False,  # display the output
    model=model_path,  # i.e. yolo11s.pt, yolo11m.pt
    records=1,  # total detections count to send an email
)

securityalarm.authenticate(from_email, password, to_email)  # authenticate the email server

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    results = securityalarm(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)  # write the processed frame.

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows
