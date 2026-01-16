import argparse
import cv2
from ultralytics import YOLO
import supervision as sv

from config import SOURCE, TARGET, LINE_Y
from view_transformer import ViewTransformer
from speed_utils import SpeedEstimator


def parse_arguments():
    parser = argparse.ArgumentParser("Vehicle Speed + Line Checkpoint")
    parser.add_argument("--source_video_path", required=True, type=str)
    parser.add_argument("--target_video_path", required=True, type=str)
    parser.add_argument("--confidence_threshold", default=0.3, type=float)
    parser.add_argument("--iou_threshold", default=0.7, type=float)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    model = YOLO("yolo11x.pt")

    tracker = sv.ByteTrack(
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

    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(SOURCE, TARGET)
    speed_estimator = SpeedEstimator(video_info.fps, LINE_Y)

    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)

            detections = detections[detections.confidence > args.confidence_threshold]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(args.iou_threshold)
            detections = tracker.update_with_detections(detections)

            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points)

            for tid, (_, y) in zip(detections.tracker_id, points):
                speed_estimator.update_speed(tid, y)
                speed_estimator.check_line(tid)

            labels = []
            for tid in detections.tracker_id:
                spd = speed_estimator.current_speeds.get(tid, 0)
                if tid in speed_estimator.vehicle_logs:
                    direction = speed_estimator.vehicle_logs[tid]["direction"]
                    labels.append(f"#{tid} {int(spd)} km/h {direction}")
                else:
                    labels.append(f"#{tid} {int(spd)} km/h")

            annotated = frame.copy()
            annotated = trace_annotator.annotate(annotated, detections)
            annotated = box_annotator.annotate(annotated, detections)
            annotated = label_annotator.annotate(annotated, detections, labels)

            cv2.putText(
                annotated,
                f"IN: {speed_estimator.count_in} | OUT: {speed_estimator.count_out}",
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

    with open("vehicle_log.txt", "w", encoding="utf-8") as f:
        f.write("ID\tSpeed(km/h)\tDirection\n")
        for vid, data in speed_estimator.vehicle_logs.items():
            f.write(f"{vid}\t{data['speed']:.2f}\t{data['direction']}\n")
