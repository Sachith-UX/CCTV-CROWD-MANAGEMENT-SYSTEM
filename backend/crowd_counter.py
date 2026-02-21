import torch
import cv2

class CrowdCounter:
    def __init__(self, model_path='models/yolov5s.pt'):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = 0.3  # confidence threshold

    def count_people(self, frame):
        results = self.model(frame)
        detections = results.xyxy[0]  # bounding boxes
        count = 0

        for *box, conf, cls in detections:
            if int(cls) == 0:  # class 0 corresponds to person in COCO dataset
                count += 1
                # Draw rectangle on frame
                frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
        
        # Put count text
        cv2.putText(frame, f'People Count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        return count, frame
