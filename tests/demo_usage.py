import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pv_audit import AuditEngine, AuditReporter, BaseAdapter
from ultralytics import YOLO

class YoloAdapter(BaseAdapter):
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def run_inference(self, image_path: str) -> float:
        results = self.model.predict(image_path, verbose=False)
        return float(results[0].boxes.conf[0].item()) if len(results[0].boxes) > 0 else 0.0

# Setup workspace
BASE_DIR = "/Users/shef9432/Desktop/PV"
audit_engine = AuditEngine(YoloAdapter(os.path.join(BASE_DIR, "yolov8n.pt")))

# Execution Pipeline
print("--- Starting stress testing... ---")
audit_engine.apply_stress(os.path.join(BASE_DIR, "bus.jpg"))
results = audit_engine.run_audit()

# Reporting
reporter = AuditReporter(results)
df = reporter.get_dataframe()
reporter.save_csv(os.path.join(BASE_DIR, "audit_report.csv"))
reporter.save_json(os.path.join(BASE_DIR, "audit_report.json"))

# Visualization
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x="level", y="metric", hue="dimension")
plt.title("Model Robustness: Stress vs Confidence")
plt.savefig(os.path.join(BASE_DIR, "robustness_summary.png"))
print("Audit complete. Report and chart saved to Desktop/PV.")
