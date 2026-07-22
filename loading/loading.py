import cv2
import numpy as np

def get_accurate_hsv(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None, None
    if len(img.shape) == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        mask = alpha > 0
    else:
        bgr = img
        mask = np.ones(img.shape[:2], dtype=bool)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 2], mask

v_45, mask2 = get_accurate_hsv("images/LedPhoto.jpg")
v_15, mask1 = get_accurate_hsv("images/LedPhoto2.jpg")

if v_45 is not None and v_15 is not None:
    target_height, target_width = v_45.shape[:2]
    v_15 = cv2.resize(v_15, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    
    h, w = v_45.shape
    crop_45 = v_45[h//2-150:h//2+150, w//2-150:w//2+150]
    crop_15 = v_15[h//2-150:h//2+150, w//2-150:w//2+150]
    
    hist_45, _ = np.histogram(crop_45.ravel(), bins=256, range=(0, 256))
    hist_15, _ = np.histogram(crop_15.ravel(), bins=256, range=(0, 256))
    
    p_45 = hist_45.astype("float32") / (hist_45.sum() + 1e-6)
    p_15 = hist_15.astype("float32") / (hist_15.sum() + 1e-6)
    
    entropy_45 = -np.sum(p_45 * np.log2(p_45 + 1e-6))
    entropy_15 = -np.sum(p_15 * np.log2(p_15 + 1e-6))
    
    entropy_shift = np.abs(entropy_45 - entropy_15)
    print(f"Lighting Entropy Shift Metric: {entropy_shift:.4f}")
    
    if entropy_shift < 0.28:
        print("SMOOTH")
    else:
        print("ROUGH")
else:
    print("Error: Could not find image assets.")