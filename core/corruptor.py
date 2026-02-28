import cv2
import numpy as np
import os

class PhysicalValidator:
    def __init__(self, output_dir="data/audit_payload"):
        self.output_dir = output_dir
        self.dimensions = ['Lux', 'DPI', 'MBlur', 'Defocus', 'ISO']
        os.makedirs(output_dir, exist_ok=True)
        for d in self.dimensions:
            os.makedirs(os.path.join(self.output_dir, d), exist_ok=True)

    def _save(self, img, uid, param_name, val):
        filename = f"{uid}_{param_name}_{val:.3f}.jpg" 
        cv2.imwrite(os.path.join(self.output_dir, param_name, filename), img)

    def apply_illumination(self, img, uid, steps):
        for alpha in steps:
            inv_gamma = 1.0 / alpha if alpha > 0 else 1.0
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            self._save(cv2.LUT(img, table), uid, "Lux", alpha)

    def apply_resolution(self, img, uid, steps):
        h, w = img.shape[:2]
        for scale in steps:
            temp = cv2.resize(img, (max(1, int(w*scale)), max(1, int(h*scale))), interpolation=cv2.INTER_AREA)
            res = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
            self._save(res, uid, "DPI", scale)

    def apply_motion_blur(self, img, uid, steps):
        for size in steps:
            k = int(size) if int(size) % 2 != 0 else int(size) + 1
            if k < 1: k = 1
            kernel = np.zeros((k, k)); kernel[int((k-1)/2), :] = 1; kernel /= kernel.sum()
            self._save(cv2.filter2D(img, -1, kernel), uid, "MBlur", size)

    def apply_defocus(self, img, uid, steps):
        for sigma in steps:
            res = img if sigma == 0 else cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
            self._save(res, uid, "Defocus", sigma)

    def apply_sensor_noise(self, img, uid, steps):
        for intensity in steps:
            noise = np.random.normal(0, intensity, img.shape).astype('float32')
            res = np.clip(img.astype('float32') + noise, 0, 255).astype('uint8')
            self._save(res, uid, "ISO", intensity)
