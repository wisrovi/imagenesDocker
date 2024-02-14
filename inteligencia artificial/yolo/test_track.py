from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("../models/knok/weights/best.pt")
model = YOLO("../models/knok/knok/damage_20230929.pt")
model = YOLO("../models/parts/weights/best.pt")

# Open the video file
video_path = "2812223 Videos Validacion 1/20231116_103809.mp4"
video_path = "2812223 Videos Validacion 1/IMG_4806.MOV"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

number_frame = 0

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    number_frame += 1

    if success:
        # resize the frame at 50%
        frame = cv2.resize(frame, (0, 0), fx=0.4, fy=0.4)

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(
            frame,
            persist=True,
            tracker="custom_tracker.yaml",
            show=True,
            save=True,
            save_txt=True,
        )

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()

        if results[0].boxes.id is None:
            track_ids = []
        else:
            track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            class_name = results.names[int(box[5])]
            track.append(
                (float(x), float(y), int(number_frame), str(class_name))
            )  # x, y center point and frame number
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            # points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            # cv2.polylines(
            #     annotated_frame,
            #     [points],
            #     isClosed=False,
            #     color=(230, 230, 230),
            #     thickness=10,
            # )

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()

# breakpoint()
print(track_history)
