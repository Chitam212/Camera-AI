import argparse
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# ===================== CONFIG =====================
SOURCE = np.array([
    [1252, 787],
    [2298, 803],
    [5039, 2159],
    [-550, 2159]
])

TARGET_WIDTH = 25     # mét
TARGET_HEIGHT = 250   # mét

TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])

LINE_Y = 100  # vị trí line trong hệ TARGET (m)
# =================================================


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32),
            target.astype(np.float32)
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        reshaped = points.reshape(-1, 1, 2).astype(np.float32)
        transformed = cv2.perspectiveTransform(reshaped, self.m)
        return transformed.reshape(-1, 2)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vehicle Speed + Line Crossing (IN/OUT)"
    )
    parser.add_argument("--source_video_path", required=True, type=str)
    parser.add_argument("--target_video_path", required=True, type=str)
    parser.add_argument("--confidence_threshold", default=0.3, type=float)
    parser.add_argument("--iou_threshold", default=0.7, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    model = YOLO("yolo11x.pt")

    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps,
        track_activation_threshold=args.confidence_threshold
    )

    thickness = sv.calculate_optimal_line_thickness(video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(video_info.resolution_wh)

    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER
    )
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER
    )

    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(SOURCE, TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    passed_ids = set()
    vehicle_logs = {}

    count_in = 0
    count_out = 0

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)

            detections = detections[detections.confidence > args.confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(args.iou_threshold)
            detections = byte_track.update_with_detections(detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points)

            for tracker_id, (_, y) in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)

                if len(coordinates[tracker_id]) < 2:
                    continue

                y_prev = coordinates[tracker_id][-2]
                y_curr = coordinates[tracker_id][-1]

                if tracker_id not in passed_ids:
                    if y_prev < LINE_Y <= y_curr:
                        direction = "IN"
                        count_in += 1
                    elif y_prev > LINE_Y >= y_curr:
                        direction = "OUT"
                        count_out += 1
                    else:
                        continue

                    passed_ids.add(tracker_id)

                    ys = list(coordinates[tracker_id])
                    time = len(ys) / video_info.fps
                    speed = abs(ys[-1] - ys[0]) / time * 3.6

                    vehicle_logs[tracker_id] = {
                        "speed": speed,
                        "direction": direction
                    }

            labels = []
            for tid in detections.tracker_id:
                if tid in vehicle_logs:
                    v = vehicle_logs[tid]
                    labels.append(f"#{tid} {int(v['speed'])}km/h {v['direction']}")
                else:
                    labels.append(f"#{tid}")

            annotated = frame.copy()
            annotated = trace_annotator.annotate(annotated, detections)
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels)

            cv2.putText(
                annotated,
                f"IN: {count_in} | OUT: {count_out}",
                (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            sink.write_frame(annotated)
            cv2.imshow("frame", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

    # ===================== SAVE TXT =====================
    with open("vehicle_log.txt", "w", encoding="utf-8") as f:
        f.write("ID\tSpeed(km/h)\tDirection\n")
        for vid, data in vehicle_logs.items():
            f.write(f"{vid}\t{data['speed']:.2f}\t{data['direction']}\n")
