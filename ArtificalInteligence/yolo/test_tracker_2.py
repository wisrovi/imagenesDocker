import datetime
from typing import List, Tuple
import numpy as np
import cv2
import yaml
from shapely.geometry import Polygon, mapping
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO
import numpy as np


class Yolo_bbox:
    """
    Class to use YOLOv5 to detect objects in a frame.

    CONFIDENCE_THRESHOLD: float
        confidence threshold to filter out weak detections

    PLOT: bool
        if True, the model will return the frame with the detections

    MODEL: YOLO
        model to use

    BASIC_PARAMS: dict
        basic parameters to use in the model

    TRACK_PARAMS: dict
        track parameters to use in the model

    EXTRA_PARAMS: dict
        extra parameters to use in the model

    results: list
        list of results of the model when the frame is processed

    is_bbox: bool
        if True, the model will return the bounding boxes

    all_trained_names: list
        list of all the trained names of the model, readed from the model file
    """

    CONFIDENCE_THRESHOLD = 0.2

    PLOT = False
    MODEL = None

    BASIC_PARAMS = dict(
        verbose=False,
    )
    TRACK_PARAMS = dict()
    EXTRA_PARAMS = dict()

    results = []
    is_bbox = False
    all_trained_names = []

    def __init__(self, path_model: str, is_bbox: bool = True):
        """
        Initialize the model.

        @type path_model: str
        @param path_model: path of the model

        @type is_bbox: bool
        @param is_bbox: if True, the model will return the bounding boxes
        """

        self.MODEL = YOLO(path_model)
        self.is_bbox = is_bbox

        try:
            self.all_trained_names = self.MODEL.names
        except:
            pass

    def detect(self, frame: np.array) -> tuple:
        """
        Process a frame and return the detections.

        @type frame: np.array
        @param frame: frame to process

        @rtype: np.array
        @return: frame with the detections
        """

        if len(self.TRACK_PARAMS) == 0:
            # if the model is not a tracker, use the basic params
            PARAMS = dict(**self.BASIC_PARAMS, **self.EXTRA_PARAMS)
            detections = self.MODEL(frame, **PARAMS)
        else:
            # if the model is a tracker, use the track params
            PARAMS = dict(**self.BASIC_PARAMS, **self.TRACK_PARAMS, **self.EXTRA_PARAMS)
            detections = self.MODEL.track(frame, **PARAMS)

        if self.is_bbox:
            detections = detections[0]

        if self.PLOT:
            if not self.is_bbox:
                new_frame = detections[0].plot()
            else:
                new_frame = detections.plot()
        else:
            new_frame = frame

        if not self.is_bbox:
            detections_to_extract_coords = detections[0]
        else:
            detections_to_extract_coords = detections

        results = self.__YoloDetec2ListCoord(detections_to_extract_coords)

        return detections, new_frame, results

    def __YoloDetec2ListCoord(self, detections: np.array) -> list:
        """
        Convert the detections to a list of coordinates.

        @type detections: np.array
        @param detections: detections of the frame

        @rtype: list
        @return: list of coordinates
        """

        # initialize the list of bounding boxes and confidences
        results = []

        ######################################
        # DETECTION
        ######################################

        # loop over the detections
        for data in detections.boxes.data.tolist():
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = data[4]

            # filter out weak detections by ensuring the
            # confidence is greater than the minimum confidence
            if float(confidence) < self.CONFIDENCE_THRESHOLD:
                continue

            # if the confidence is greater than the minimum confidence,
            # get the bounding box and the class id
            xmin, ymin, xmax, ymax = (
                int(data[0]),
                int(data[1]),
                int(data[2]),
                int(data[3]),
            )
            class_id = int(data[5])

            # add the bounding box (x, y, w, h), confidence and class id to the results list
            results.append(
                [[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id]
            )

        return results


class Yolo_mask(Yolo_bbox):
    """
    This class is used to process the frames and return the masks.

    @type CONFIDENCE_THRESHOLD: float
    @param CONFIDENCE_THRESHOLD: confidence threshold

    @type PLOT: bool
    @param PLOT: if True, the model will return the bounding boxes
    """

    annotated_frame = None

    def detector(self, frame: np.array) -> tuple:
        """
        Process a frame and return the detections.

        @type frame: np.array
        @param frame: frame to process

        @rtype: tuple(np.array, float, List[Polygon])
        @return: frame with the detections, confidence and polygons of the detections
        """

        detections_seg, annotated_frame, results_bbox = self.detect(frame)
        self.annotated_frame = annotated_frame

        polygons_bbox = self.__result_to_polygons(results_bbox)
        if len(polygons_bbox) == 0:
            return frame, 0, None

        results_bbox = results_bbox[0]

        self.frame = frame
        self.results = detections_seg

        return (
            self.graph_frame_without_area_out_polygon(frame, polygons_bbox),
            round(results_bbox[1], 2),
            results_bbox[2],
        )

    def result_model_to_polygons(self) -> Tuple[List[Polygon], List[str]]:
        """
        This function is used to convert the results of the model to polygons.

        @rtype: Tuple(List(Polygon), List(str))
        @return: list of polygons and list of categories of the polygons
        """

        all_polygons, all_categories_of_polygons = [], []

        try:
            # validate if the results is a iterable object
            for result in self.results:
                pass
        except:
            self.results = [self.results]

        for result in self.results:
            if self.is_bbox:
                predicted_object = result.boxes.data.cpu().numpy()
            elif result.masks is not None:
                predicted_object = result.masks.data.cpu().numpy()
            else:
                continue

            predicted_classes = [int(c) for c in result.boxes.cls.tolist()]

            # loop over the masks of the objects detected
            for object_mask_id, object_points in enumerate(predicted_object):
                # resize the mask to the original frame size
                if not self.is_bbox:
                    object_points = cv2.resize(
                        object_points, (self.frame.shape[1], self.frame.shape[0])
                    )

                    # find the contours of the mask
                    all_contours_of_object_points, _ = cv2.findContours(
                        object_points.astype(np.uint8),
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE,
                    )

                    # convert the contours to polygons
                    for this_contour in all_contours_of_object_points:
                        # one polygon have to have at least 3 points, if not, continue
                        if len(this_contour) < 3:
                            continue

                        # convert the contour to a polygon and append to the list of polygons
                        this_polygon = Polygon([point[0] for point in this_contour])
                        all_polygons.append(this_polygon)
                else:
                    all_contours_of_object_points = [object_points]

                    # convert the contours to polygons
                    for this_contour in all_contours_of_object_points:
                        xmin, ymin, w, h = this_contour[0:4]
                        xmax, ymax = xmin + w, ymin + h
                        this_polygon = Polygon(
                            [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                        )
                        all_polygons.append(this_polygon)

                for this_contour in all_contours_of_object_points:
                    # append the category of the polygon
                    if self.all_trained_names:
                        id_name_of_this_mask = predicted_classes[object_mask_id]
                        all_categories_of_polygons.append(
                            self.all_trained_names[id_name_of_this_mask]
                        )
                    else:
                        all_categories_of_polygons.append(
                            predicted_classes[object_mask_id]
                        )

        return all_polygons, all_categories_of_polygons

    @staticmethod
    def __result_to_polygons(results: list) -> list:
        """
        Convert the results to a list of polygons.

        @type results: list
        @param results: list of results

        @rtype: list
        @return: list of polygons
        """

        polygons = []
        for result in results:
            xmin, ymin, w, h = result[0]
            xmax, ymax = xmin + w, ymin + h

            polygons.append(
                Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
            )

        return polygons

    @staticmethod
    def graph_frame_without_area_out_polygon(
        frame: np.array, polygons: list
    ) -> np.array:
        """
        Graph the frame without the area outside the poligon.

        @type frame: np.array
        @param frame: frame to process

        @type polygons: list
        @param polygons: list of poligons

        @rtype: np.array
        @return: frame with the detections
        """

        masked = np.zeros_like(frame)

        for polygon in polygons:
            exterior_coords = np.array(polygon.exterior.coords, dtype=np.int32)
            cv2.fillPoly(masked, [exterior_coords], (255, 255, 255))

        masked = cv2.bitwise_and(frame, masked)

        return masked

    @staticmethod
    def polygon2string(polygons: list) -> List[Polygon]:
        """
        This function is used to convert the poligons to string.

        @type polygons: list
        @param polygons: list of poligons

        @rtype: list
        @return: list of poligons in string format
        """

        poligons_str = []
        for pol in polygons:
            poligons_str.append(mapping(pol))

        return poligons_str

    @staticmethod
    def string2polygon(polygons_str: List[str]) -> list:
        """
        This function is used to convert the polygons in string format
        to polygons.

        @type polygons_str: list
        @param polygons_str: list of poligons in string format

        @rtype: list
        @return: list of poligons
        """

        polygons = []
        for pol in polygons_str:
            restored_polygon = Polygon(pol["coordinates"][0])
            polygons.append(restored_polygon)

        return polygons

    def graph_contorn(self, polygons: List[Polygon]) -> np.array:
        """
        This function is used to graph the contours of the polygons.

        @type polygons: List(Polygon)
        @param polygons: list of poligons

        @rtype: np.array
        @return: frame with the contours
        """

        masked = np.zeros_like(self.frame)

        for polygon in polygons:
            cv2.polylines(
                masked, np.int32([polygon.exterior.coords]), True, (0, 0, 255), 3
            )

        return masked


class Yolo_track(Yolo_mask):
    CONFIDENCE_THRESHOLD = 0.4

    TRACK_PARAMS = dict(
        persist=True,  # necessary to track for take the last frame how comparetion frame for looking for the emmbeding
        tracker="custom_tracker.yaml",  # custom tracker for tracking the objects
    )

    MAXIMUM_TRACKING_VECTOR_SIZE = (
        20  # maximum number of frames to retain for each track, recommended 30
    )

    track_history = defaultdict(lambda: [])
    track_history_ids = defaultdict(lambda: [])

    @property
    def tracking_history(self):
        # filter for get only the tracks with more than MAXIMUM_TRACKING_VECTOR_SIZE/2 registers
        tracking_history = [
            (track_id, track)
            for track_id, track in self.track_history.items()
            if len(track) > int(self.MAXIMUM_TRACKING_VECTOR_SIZE * 0.5)
        ]

        # for every track, get all names of the track and get the most common name
        # and replace every name of the track for the most common name
        for track_id, track in tracking_history:
            if not track:
                continue

            names_of_tracked_objects = [
                track_data[5] for track_data in track
            ]  # get the name of the track in position 5 of every data
            most_common_name = max(
                set(names_of_tracked_objects), key=names_of_tracked_objects.count
            )
            for i in range(len(track)):
                track[i] = (
                    track[i][0],
                    track[i][1],
                    track[i][2],
                    track[i][3],
                    track[i][4],
                    most_common_name,
                )

            self.track_history_ids[track_id] = most_common_name

        # get the names of every track
        names_of_tracked_objects = [track[0][5] for _, track in tracking_history]
        # count the repetitions of every name
        number_of_repetitions_of_the_same_object = {
            name_of_the_tracked_object: names_of_tracked_objects.count(
                name_of_the_tracked_object
            )
            for name_of_the_tracked_object in set(names_of_tracked_objects)
        }

        return tracking_history, number_of_repetitions_of_the_same_object, self.track_history_ids

    def predict(self, frame: np.array, extra_data: dict = {}) -> np.array:
        self.detector(frame)[1]

        results = self.results
        if len(results) == 0:
            return frame

        predicted_boxes = results[0].boxes.xywh.cpu()
        confidences_boxes = results[0].boxes.conf.cpu()

        polygons_parts, category_of_polygons_of_parts = self.result_model_to_polygons()
        polygons_parts_str = self.polygon2string(polygons_parts)

        if len(predicted_boxes) > 0:
            ids = results[0].boxes.id
            if ids is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, confidence, polygon, polygon_str, track_id, class_name in zip(
                    predicted_boxes,
                    confidences_boxes,
                    polygons_parts,
                    polygons_parts_str,
                    track_ids,
                    category_of_polygons_of_parts,
                ):
                    x, y, w, h = box
                    xmin, ymin, xmax, ymax = (
                        int(x - w / 2),
                        int(y - h / 2),
                        int(x + w / 2),
                        int(y + h / 2),
                    )

                    self.track_history[track_id].append(
                        (
                            float(xmin),
                            float(ymin),
                            float(xmax),
                            float(ymax),
                            extra_data,
                            str(class_name),
                            round(float(confidence), 2),
                        )
                    )

                    # limit the track history to 30 frames for each track, if not, pop the first element of the list
                    if (
                        len(self.track_history[track_id])
                        > self.MAXIMUM_TRACKING_VECTOR_SIZE
                    ):  # retain 90 tracks for 90 frames
                        self.track_history[track_id].pop(0)

                    if not self.PLOT:
                        random_color = np.random.randint(0, 255, size=3).tolist()

                        # graph circle in (x, y) and put the text class_name
                        cv2.circle(
                            self.annotated_frame, (int(x), int(y)), 10, random_color, -1
                        )
                        cv2.putText(
                            self.annotated_frame,
                            f"{track_id} {class_name}",
                            (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            random_color,
                            2,
                        )

        annotated_frame = self.annotated_frame
        return annotated_frame


class Custom_yolo_track(Yolo_track):
    EXTRA_PARAMS = dict(
        # classes=[0],  # filter by classes
        conf=0.40,  # minimal confidence threshold (recommended 0.25 for faster and 0.5 for higher accuracy)
        iou=0.8,  # intersection over union (IoU) threshold for NMS
        retina_masks=True,  # use high-resolution segmentation masks
        half=True,  # use FP16 half-precision inference
    )

    MAXIMUM_TRACKING_VECTOR_SIZE = (
        10  # maximum number of frames to retain for each track, recommended 30
    )

    CONFIDENCE_THRESHOLD = 0.4  # confidence threshold to filter out weak detections


# **********************************************************************************************************************


def process(configuration, verbose: bool = False):
    # change configuration
    # model_knok.EXTRA_PARAMS["classes"] = [1, 6, 7, 13]
    model_knok.EXTRA_PARAMS["conf"] = configuration.get("conf", 0.4)
    model_knok.EXTRA_PARAMS["iou"] = configuration.get("iou", 0.8)
    model_knok.EXTRA_PARAMS["retina_masks"] = configuration.get("retina_masks", True)
    model_knok.EXTRA_PARAMS["half"] = configuration.get("half", True)

    # model_knok.EXTRA_PARAMS["classes"] = [14]
    # model_knok.PLOT = True

    # process configuration
    summary = {}

    for video_path, real_data in DATA.items():
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for frame_number in tqdm(
            range(total_frames),
            desc=f"\033[94mProcesando frames: {video_path}\033[0m",
            unit="frame",
            bar_format="{l_bar}\033[92m{bar}\033[0m{r_bar}",
        ):
            success, frame = cap.read()
            if not success:
                break

            actual_width = frame.shape[1]
            RESIZE = BASE_WIDTH / actual_width
            frame = cv2.resize(frame, (0, 0), fx=RESIZE, fy=RESIZE)

            annotated_frame = model_knok.predict(
                frame,
                extra_data={
                    "frame_number": frame_number,
                },
            )

            if GRAPH:
                cv2.imshow(f"{video_path}", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        (
            tracking_history,
            number_of_repetitions_of_the_same_object,
        ) = model_knok.tracking_history

        if verbose:
            print(number_of_repetitions_of_the_same_object)
            print(DATA[video_path])

        summary[video_path] = number_of_repetitions_of_the_same_object

    cap.release()
    cv2.destroyAllWindows()

    return summary


def find_score(individual, real):
    total_real = sum(real.values())
    total_individual = sum(individual.values())

    # total_individual = 0
    # for key in real.keys():
    #     if key in individual.keys():
    #         ind_key = individual[key]
    #         real_key = real[key]

    #         if ind_key > real_key:
    #             try:
    #                 # si supera el real, se calcula una diferencia con el real y se le resta al real, de esta forma se obtiene el porcentaje de error del individuo
    #                 ind_key = int(abs(real_key - int(abs(real_key - (ind_key / real_key)))))
    #                 if ind_key == 0:
    #                     ind_key = 1
    #                 elif ind_key > real_key:
    #                     # si aun supera el real, se pone el real menos el 25%
    #                     ind_key = int(real_key * 0.75)
    #             except:
    #                 ind_key = -1

    #         diff = real_key - abs(real_key - ind_key)

    #         total_individual += diff

    #         # f"key: {key} - real: {real_key} - count: {ind_key} - tmp_score: {diff}, total_score: {total_individual}"

    # score = int((total_individual / total_real) * 100) / 100

    score = total_real - total_individual

    return score, total_real, total_individual


def evaluate(configuration):
    start = datetime.datetime.now()

    # change configuration in yaml file
    # read yaml file: custom_tracker.yaml with yaml library

    with open("custom_tracker.yaml", "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    data["track_high_thresh"] = configuration.get("track_high_thresh", 0.5)
    data["track_low_thresh"] = configuration.get("track_low_thresh", 0.08)
    data["new_track_thresh"] = configuration.get("new_track_thresh", 0.75)
    data["track_buffer"] = configuration.get("track_buffer", 40)
    data["match_thresh"] = configuration.get("match_thresh", 0.85)
    data["proximity_thresh"] = configuration.get("proximity_thresh", 0.5)
    data["appearance_thresh"] = configuration.get("appearance_thresh", 0.25)
    data["with_reid"] = configuration.get("with_reid", False)

    with open("custom_tracker.yaml", "w") as file:
        yaml.dump(data, file)

    # process configuration
    summary = process(configuration)

    # evaluate configuration
    total_score = 0
    total_real = 0
    total_individual = 0
    for individual, real in zip(summary.values(), DATA.values()):
        score, real, individual = find_score(individual, real)
        total_score += score
        total_real += real
        total_individual += individual

    percentage_closeness = round(total_score / len(DATA), 2)
    print()
    print(
        f"score: {percentage_closeness} - real: {total_real} - individual: {total_individual}"
    )

    end = datetime.datetime.now()

    print(f"Time: {(end - start).total_seconds()} seconds")

    return (int(percentage_closeness * 100),)


GRAPH = False
BASE_WIDTH = 600
DATA = {
    "2812223 Videos Validacion 1/IMG_4806.MOV": {
        "Faro": 4,
        "Paragolpes delantero": 1,
        "Paragolpes Trasero": 1,
        "Capó": 1,
        "Puerta": 6,
        "Cristal": 12,
        "Luna": 2,
        "Techo": 1,
        "Maletero": 1,
        "Retrovisor": 3,
        "Matricula": 2,
        "Ruedas": 6,
        "Aleta": 2,
    },
    "2812223 Videos Validacion 1/20231116_103809.mp4": {
        "Faro": 6,
        "Paragolpes delantero": 1,
        "Paragolpes Trasero": 1,
        "Capó": 1,
        "Puerta": 6,
        "Cristal": 9,
        "Luna": 2,
        "Techo": 1,
        "Maletero": 1,
        "Retrovisor": 3,
        "Matricula": 2,
        "Ruedas": 6,
        "Aleta": 2,
    },
    "2812223 Videos Validacion 1/20231116_105633.mp4": {
        "Retrovisor": 3,
        "Matricula": 2,
        "Ruedas": 6,
        "Puerta": 6,
        "Techo": 1,
        "Paragolpes delantero": 1,
        "Paragolpes Trasero": 1,
        "Maletero": 1,
        "Capó": 1,
        "Faro": 6,
        "Cristal": 10,
        "Luna": 2,
        "Aleta": 2,
    },
    "2812223 Videos Validacion 1/IMG_4812.MOV": {
        "Retrovisor": 3,
        "Matricula": 2,
        "Ruedas": 6,
        "Puerta": 6,
        "Techo": 1,
        "Paragolpes delantero": 1,
        "Paragolpes Trasero": 1,
        "Maletero": 1,
        "Capó": 1,
        "Faro": 6,
        "Cristal": 8,
        "Luna": 2,
        "Aleta": 2,
    },
    "2812223 Videos Validacion 1/IMG_4814.MOV": {
        "Retrovisor": 3,
        "Matricula": 2,
        "Ruedas": 6,
        "Puerta": 6,
        "Techo": 1,
        "Paragolpes delantero": 1,
        "Paragolpes Trasero": 1,
        "Maletero": 1,
        "Capó": 1,
        "Faro": 6,
        "Cristal": 4,
        "Luna": 2,
        "Aleta": 2,
    },
    "2812223 Videos Validacion 1/IMG_4816.MOV": {
        "Retrovisor": 3,
        "Matricula": 2,
        "Ruedas": 6,
        "Puerta": 6,
        "Techo": 1,
        "Paragolpes delantero": 1,
        "Paragolpes Trasero": 1,
        "Maletero": 1,
        "Capó": 1,
        "Faro": 6,
        "Cristal": 6,
        "Luna": 2,
        "Aleta": 2,
    },
    "2812223 Videos Validacion 1/IMG_4817.MOV": {
        "Retrovisor": 4,
        "Matricula": 2,
        "Ruedas": 4,
        "Puerta": 4,
        "Techo": 1,
        "Paragolpes delantero": 1,
        "Paragolpes Trasero": 1,
        "Maletero": 1,
        "Capó": 1,
        "Faro": 6,
        "Cristal": 6,
        "Luna": 2,
        "Aleta": 2,
    },
    "2812223 Videos Validacion 1/IMG_4818.MOV": {
        "Retrovisor": 3,
        "Matricula": 2,
        "Ruedas": 6,
        "Puerta": 6,
        "Techo": 1,
        "Paragolpes delantero": 1,
        "Paragolpes Trasero": 1,
        "Maletero": 1,
        "Capó": 1,
        "Faro": 6,
        "Cristal": 12,
        "Luna": 2,
        "Aleta": 2,
    },
    "2812223 Videos Validacion 1/IMG_4819.MOV": {
        "Retrovisor": 3,
        "Matricula": 2,
        "Ruedas": 6,
        "Puerta": 3,
        "Techo": 1,
        "Paragolpes delantero": 1,
        "Paragolpes Trasero": 1,
        "Maletero": 1,
        "Capó": 1,
        "Faro": 6,
        "Cristal": 6,
        "Luna": 3,
        "Aleta": 2,
    },
    "2812223 Videos Validacion 1/j1.mp4": {
        "Retrovisor": 1,
        "Matricula": 1,
        "Ruedas": 2,
        "Puerta": 2,
        "Techo": 1,
        "Paragolpes delantero": 0,
        "Paragolpes Trasero": 1,
        "Maletero": 1,
        "Capó": 0,
        "Faro": 2,
        "Cristal": 3,
        "Luna": 1,
        "Aleta": 1,
    },
}

# model_knok = Custom_yolo_track("../models/licence_plate/weights/best.pt", is_bbox=True)
model_knok = Custom_yolo_track("../models/parts/weights/best.pt", is_bbox=False)


if __name__ == "__main__":
    configuration = dict(
        conf=0.35,
        iou=0.8,
        retina_masks=True,
        half=True,
        #
        track_high_thresh=0.5,
        track_low_thresh=0.08,
        new_track_thresh=0.75,
        track_buffer=30,
        match_thresh=0.85,
        #
        proximity_thresh=0.5,
        appearance_thresh=0.25,
        with_reid=False,
    )

    percentage_closeness = evaluate(configuration)
    print(f"This configuration has a {percentage_closeness}% of closeness")

    print()
    print(model_knok.all_trained_names)
