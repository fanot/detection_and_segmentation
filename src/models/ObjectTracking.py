import cv2
from src.logger import logger
import numpy as np

class ObjectTracker:
    def __init__(self):
        self.trackers = []
        self.next_id = 1  # Инициализация идентификатора для новых объектов
        self.tracking_counter = 0  # Счетчик трекинга

    def start_tracking(self, frame, detected_objects):
        """ Инициализирует трекеры для новых обнаруженных объектов. """
        for obj in detected_objects:
            x_min = int(np.min(obj['x_coords']))
            y_min = int(np.min(obj['y_coords']))
            x_max = int(np.max(obj['x_coords']))
            y_max = int(np.max(obj['y_coords']))
            width = x_max - x_min
            height = y_max - y_min

            if width > 0 and height > 0:
                bbox = (x_min, y_min, width, height)
                if not any(self.bboxes_overlap(bbox, existing_bbox) for _, existing_bbox, _ in self.trackers):
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, bbox)
                    self.trackers.append((tracker, bbox, self.next_id))
                    self.next_id += 1
                    self.tracking_counter += 1  # Увеличиваем счетчик трекинга
            else:
                logger.error(f"Invalid bbox with width {width} and height {height} from coordinates x_min={x_min}, y_min={y_min}, x_max={x_max}, y_max={y_max}")

    def update_tracking(self, frame):
        """ Обновляет трекеры и возвращает список активных трекеров и их bounding boxes. """
        active_trackers = []
        for tracker, bbox, cloud_id in list(self.trackers):
            success, new_bbox = tracker.update(frame)
            if success:
                active_trackers.append((tracker, new_bbox, cloud_id))
            else:
                logger.info(f"Tracker {cloud_id} failed to update and will be removed.")
        # Обновляем список трекеров
        self.trackers = active_trackers
        # Возвращаем список кортежей, содержащих идентификатор облака, новый bbox, и идентификатор облака.
        return [(cloud_id, bbox, cloud_id) for _, bbox, cloud_id in active_trackers]

    def bboxes_overlap(self, bbox1, bbox2):
        """ Проверяет пересечение двух bounding boxes. """
        x1_min, y1_min, w1, h1 = bbox1
        x2_min, y2_min, w2, h2 = bbox2
        x1_max, y1_max = x1_min + w1, y1_min + h1
        x2_max, y2_max = x2_min + w2, y2_min + h2

        return (x1_min <= x2_max and x1_max >= x2_min and y1_min <= y2_max and y1_max >= y2_min)
