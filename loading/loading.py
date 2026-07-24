import argparse
import os
import re
import sys
import cv2
import numpy as np


OS_IMAGES_DIR = "images"
os.makedirs(OS_IMAGES_DIR, exist_ok=True)


def get_next_sequence_numbers():
    files = os.listdir(OS_IMAGES_DIR)
    existing_nums = []
    for f in files:
        match = re.search(r"LedPhoto(\d+)\.jpg", f)
        if match:
            existing_nums.append(int(match.group(1)))
    next_bf = max(existing_nums) + 1 if existing_nums else 1
    next_df = next_bf + 1
    return next_bf, next_df


def capture_astra_photo(camera_index=1, window_name="Astra Preview"):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
   
    if not cap.isOpened():
        print(f"Error: Could not open camera on device index {camera_index}.", file=sys.stderr)
        return None
       
    for _ in range(30):
        cap.read()
       
    print(f"[{window_name}] Camera ready. Press SPACE to capture, 'q' to exit.")
    captured_frame = None
   
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("Error: Camera stream failed to yield a valid frame matrix.", file=sys.stderr)
                    break
           
            preview = frame.copy()
            h, w = preview.shape[:2]
            start_y, end_y = h // 2 - 150, h // 2 + 150
            start_x, end_x = w // 2 - 150, w // 2 + 150
           
            cv2.rectangle(preview, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
            cv2.putText(preview, "TARGET ANALYSIS ZONE", (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(preview, "SPACE: Capture | q: Quit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
           
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Capture canceled by user.")
                break
            if key == ord(" "):
                captured_frame = frame
                print("Frame captured successfully!")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
       
    return captured_frame


def get_luminance_and_mask(img):
    """Extracts mathematically accurate luminance (Y) and mask matrix."""
    if img is None:
        return None, None
       
    if len(img.shape) == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        mask = alpha > 0
    else:
        bgr = img
        mask = np.ones(img.shape[:2], dtype=bool)
       
    # Convert BGR to Grayscale using standard ITU-R BT.601 luminance weights
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return gray, mask


def calculate_fft_roughness(crop_img):
    """Calculates physical surface roughness index using a 2D Fourier Transform on darkfield assets."""
    # Compute 2D Fast Fourier Transform
    f_transform = np.fft.fft2(crop_img.astype(np.float32))
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
   
    # Isolate high frequencies (eliminate low-frequency macro illumination variations)
    h, w = magnitude_spectrum.shape
    cy, cx = h // 2, w // 2
   
    # Construct a high-pass mask blocking central low frequencies (radius of 15 pixels)
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    low_freq_mask = (x**2 + y**2) <= 15**2
    magnitude_spectrum[low_freq_mask] = 0
   
    # Calculate the normalized high-frequency energy index
    total_high_freq_energy = np.sum(magnitude_spectrum)
    normalized_roughness_score = total_high_freq_energy / (h * w)
   
    return normalized_roughness_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=2, help="Camera device index (Try 1 or 2 for Astra color feed)")
    args = parser.parse_args()
   
    bf_num, df_num = get_next_sequence_numbers()
   
    print(f"\n--- STEP 1: Capture Brightfield Image (Using Camera Index {args.camera}) ---")
    input("Prepare LED 1 (Brightfield) lighting setup and press Enter...")
    bf_raw = capture_astra_photo(camera_index=args.camera, window_name="LED 1 - Brightfield")
   
    if bf_raw is not None:
        bf_filename = f"LedPhoto{bf_num}.jpg"
        bf_path = os.path.join(OS_IMAGES_DIR, bf_filename)
        cv2.imwrite(bf_path, bf_raw)
        print(f"Saved Brightfield capture to {bf_path}")
    else:
        print("Error: Step 1 capture failed or was canceled.")
        sys.exit(1)
       
    print("\n--- STEP 2: Capture Darkfield Image ---")
    input("Prepare LED 2 (Darkfield) lighting setup (Keep target sample stationary) and press Enter...")
    df_raw = capture_astra_photo(camera_index=args.camera, window_name="LED 2 - Darkfield")
   
    if df_raw is not None:
        df_filename = f"LedPhoto{df_num}.jpg"
        df_path = os.path.join(OS_IMAGES_DIR, df_filename)
        cv2.imwrite(df_path, df_raw)
        print(f"Saved Darkfield capture to {df_path}")
    else:
        print("Error: Step 2 capture failed or was canceled.")
        sys.exit(1)
       
    v_bf, mask_bf = get_luminance_and_mask(bf_raw)
    v_df, mask_df = get_luminance_and_mask(df_raw)
   
    if v_bf is not None and v_df is not None:
        target_height, target_width = v_df.shape[:2]
       
        v_bf = cv2.resize(v_bf, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
       
        h, w = v_df.shape
        start_y, end_y = h // 2 - 150, h // 2 + 150
        start_x, end_x = w // 2 - 150, w // 2 + 150
       
        crop_bf = v_bf[start_y:end_y, start_x:end_x]
        crop_df = v_df[start_y:end_y, start_x:end_x]
       
        cv2.imwrite(os.path.join(OS_IMAGES_DIR, f"crop_brightfield_{bf_num}.jpg"), crop_bf)
        cv2.imwrite(os.path.join(OS_IMAGES_DIR, f"crop_darkfield_{df_num}.jpg"), crop_df)
       
        # Perform Spatial Frequency Analysis on the Darkfield sample
        roughness_index = calculate_fft_roughness(crop_df)
        print(f"\nCalculated High-Frequency Scattering Index: {roughness_index:.2f}")
       
        # Classification thresholds calibrated for high-frequency optical energy
        if roughness_index < 120.0:
            print("Surface Classification: SMOOTH")
        elif roughness_index < 650.0:
            print("Surface Classification: Semi-Rough")
        else:
            print("Surface Classification: VERY ROUGH")
    else:
        print("Error: Could not process live camera image assets.")



