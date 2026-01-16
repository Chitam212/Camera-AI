from collections import defaultdict, deque


class SpeedEstimator:
    def __init__(self, fps: int, line_y: float):
        self.fps = fps
        self.line_y = line_y

        self.coordinates = defaultdict(lambda: deque(maxlen=fps))
        self.current_speeds = {}
        self.passed_ids = set()
        self.vehicle_logs = {}

        self.count_in = 0
        self.count_out = 0

    def update_speed(self, tracker_id: int, y: float):
        self.coordinates[tracker_id].append(y)

        if len(self.coordinates[tracker_id]) >= 2:
            y_start = self.coordinates[tracker_id][0]
            y_end = self.coordinates[tracker_id][-1]
            time = len(self.coordinates[tracker_id]) / self.fps
            speed = abs(y_end - y_start) / time * 3.6
            self.current_speeds[tracker_id] = speed

    def check_line(self, tracker_id: int):
        if tracker_id in self.passed_ids:
            return

        if len(self.coordinates[tracker_id]) < 2:
            return

        y_prev = self.coordinates[tracker_id][-2]
        y_curr = self.coordinates[tracker_id][-1]

        if y_prev < self.line_y <= y_curr:
            direction = "IN"
            self.count_in += 1
        elif y_prev > self.line_y >= y_curr:
            direction = "OUT"
            self.count_out += 1
        else:
            return

        self.passed_ids.add(tracker_id)

        self.vehicle_logs[tracker_id] = {
            "speed": self.current_speeds.get(tracker_id, 0),
            "direction": direction
        }
