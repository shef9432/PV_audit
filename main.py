import cv2
import numpy as np
from core.corruptor import PhysicalValidator
from utils.logger import AuditLogger

def generate_samples():
    img = cv2.imread("data/input/sample.jpg")
    if img is None: raise FileNotFoundError("Please put sample.jpg in data/input/")
    
    pv = PhysicalValidator()
    uid = "PRO_AUDIT"

    # Extreme range for survivability boundary
    pv.apply_illumination(img, uid, np.linspace(0.005, 0.5, 40))
    pv.apply_resolution(img, uid, np.linspace(0.01, 0.2, 40))
    pv.apply_motion_blur(img, uid, np.linspace(5, 200, 40))
    pv.apply_defocus(img, uid, np.linspace(0, 50, 40))
    pv.apply_sensor_noise(img, uid, np.linspace(0, 300, 40))

    AuditLogger().build_template()
    print("🚀 Extreme stress-test payload generated.")

if __name__ == "__main__":
    generate_samples()
