import pandas as pd
import os
import json

class InferenceAuditor:
    def __init__(self, model_source):
        self.model = None # Placeholder for your company model
        print(f"Auditor initialized for: {model_source}")

    def calculate_iou(self, box1, box2):
        """Calculates Intersection over Union (IoU) between two boxes [x1, y1, x2, y2]."""
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection_area / float(area1 + area2 - intersection_area)

    def validate_prediction(self, pred_results, gt_json_path):
        """
        Validates if the prediction matches the Ground Truth.
        pred_results: list of {'box': [x1,y1,x2,y2], 'class': 0, 'conf': 0.9}
        """
        with open(gt_json_path, 'r') as f:
            gt_data = json.load(f) # Assuming standard format: {"boxes": [[...]], "labels": [0]}
        
        best_valid_conf = 0.0
        
        for pred in pred_results:
            for gt_box, gt_label in zip(gt_data['boxes'], gt_data['labels']):
                iou = self.calculate_iou(pred['box'], gt_box)
                # Check if it's the right object in the right place
                if iou > 0.5 and pred['class'] == gt_label:
                    best_valid_conf = max(best_valid_conf, pred['conf'])
        
        return best_valid_conf

def run_systematic_audit():
    # ... previous logic to loop through data ...
    # Now includes: conf = auditor.validate_prediction(raw_preds, "data/input/sample.json")
    pass
