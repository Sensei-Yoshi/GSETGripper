import argparse
import time
from typing import Optional

import cv2
import numpy as np
import serial
from serial.tools import list_ports

from pyorbbecsdk import Config, OBSensorType, Pipeline


LOWEST_HEIGHT_MM = 2500
SERIAL_PORT = None
BAUD_RATE = 9600
ON_MESSAGE = "LED_ON\n"
OFF_MESSAGE = "LED_OFF\n"
SERIAL_STARTUP_DELAY_SECONDS = 2.0
FRAME_TIMEOUT_MS = 500
MIN_VALID_DEPTH_MM = 20
MAX_VALID_DEPTH_MM = 10000
SEND_ONLY_ON_CHANGE = True
SHOW_DEPTH_WINDOW = True
WINDOW_NAME = "Orbbec Depth Serial Trigger"
ESC_KEY = 27
CENTER_ROI_RADIUS_PIXELS = 120


def lowest_height_is_below_threshold(depth_map_mm: np.ndarray) -> bool:
    lowest_height = get_lowest_valid_height_mm(depth_map_mm)
    if lowest_height is None:
        return False

    return lowest_height < LOWEST_HEIGHT_MM


def get_lowest_valid_height_mm(depth_map_mm: np.ndarray) -> Optional[float]:
    valid_depth = depth_map_mm[
        np.isfinite(depth_map_mm)
        & (depth_map_mm >= MIN_VALID_DEPTH_MM)
        & (depth_map_mm <= MAX_VALID_DEPTH_MM)
    ]

    if valid_depth.size == 0:
        return None

    return float(valid_depth.min())


def get_valid_depth_mask(depth_map_mm: np.ndarray) -> np.ndarray:
    return (
        np.isfinite(depth_map_mm)
        & (depth_map_mm >= MIN_VALID_DEPTH_MM)
        & (depth_map_mm <= MAX_VALID_DEPTH_MM)
    )


def get_center_depth_mm(depth_map_mm: np.ndarray) -> Optional[float]:
    height, width = depth_map_mm.shape
    center_depth = float(depth_map_mm[height // 2, width // 2])
    if MIN_VALID_DEPTH_MM <= center_depth <= MAX_VALID_DEPTH_MM:
        return center_depth

    return None


def get_center_roi_mask(depth_map_mm: np.ndarray) -> np.ndarray:
    height, width = depth_map_mm.shape
    center_y = height // 2
    center_x = width // 2
    y_indices, x_indices = np.ogrid[:height, :width]
    distance_squared = (x_indices - center_x) ** 2 + (y_indices - center_y) ** 2
    return distance_squared <= CENTER_ROI_RADIUS_PIXELS ** 2


def get_trigger_depth_map(depth_map_mm: np.ndarray) -> np.ndarray:
    center_roi_mask = get_center_roi_mask(depth_map_mm)
    return depth_map_mm[center_roi_mask]


def find_arduino_port() -> str:
    if SERIAL_PORT:
        return SERIAL_PORT

    candidates = []
    for port in list_ports.comports():
        description = f"{port.description} {port.manufacturer or ''}".lower()
        device = port.device.lower()
        if (
            "arduino" in description
            or "ch340" in description
            or "usb serial" in description
            or "usbmodem" in device
            or "usbserial" in device
        ):
            candidates.append(port.device)

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) > 1:
        raise RuntimeError(
            "Multiple possible Arduino serial ports found. Set SERIAL_PORT or pass --port. "
            f"Candidates: {', '.join(candidates)}"
        )

    available = ", ".join(port.device for port in list_ports.comports()) or "none"
    raise RuntimeError(
        "No Arduino serial port found. Set SERIAL_PORT or pass --port. "
        f"Available ports: {available}"
    )


def send_led_state(serial_connection: serial.Serial, led_on: bool) -> None:
    message = ON_MESSAGE if led_on else OFF_MESSAGE
    serial_connection.write(message.encode("ascii"))
    serial_connection.flush()


def read_depth_map_mm(frames) -> Optional[np.ndarray]:
    depth_frame = frames.get_depth_frame()
    if depth_frame is None:
        return None

    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()
    depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
    return depth.astype(np.float32) * scale


def render_depth_window(depth_map_mm: np.ndarray, lowest_height: Optional[float], triggered: bool) -> bool:
    valid_mask = get_valid_depth_mask(depth_map_mm)
    clipped_depth = np.clip(depth_map_mm, MIN_VALID_DEPTH_MM, MAX_VALID_DEPTH_MM)
    normalized_depth = (
        255
        - (
            (clipped_depth - MIN_VALID_DEPTH_MM)
            * 255
            / (MAX_VALID_DEPTH_MM - MIN_VALID_DEPTH_MM)
        )
    ).astype(np.uint8)
    normalized_depth[~valid_mask] = 0
    depth_image = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
    depth_image[~valid_mask] = (0, 0, 0)

    min_depth_text = "Min depth: none" if lowest_height is None else f"Min depth: {lowest_height:.1f} mm"
    center_depth = get_center_depth_mm(depth_map_mm)
    center_text = "Center depth: invalid" if center_depth is None else f"Center depth: {center_depth:.1f} mm"
    trigger_text = f"Trigger: {triggered}"
    threshold_text = f"Threshold: {LOWEST_HEIGHT_MM} mm"
    roi_text = f"ROI radius: {CENTER_ROI_RADIUS_PIXELS}px"
    valid_text = f"Valid pixels: {int(valid_mask.sum())}/{valid_mask.size}"
    status_color = (0, 255, 0) if triggered else (255, 255, 255)

    height, width = depth_map_mm.shape
    center = (width // 2, height // 2)
    cv2.circle(depth_image, center, CENTER_ROI_RADIUS_PIXELS, (255, 255, 255), 1)
    cv2.drawMarker(
        depth_image,
        center,
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=14,
        thickness=1,
    )

    cv2.rectangle(depth_image, (10, 10), (390, 180), (0, 0, 0), -1)
    cv2.putText(depth_image, min_depth_text, (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(depth_image, center_text, (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(depth_image, trigger_text, (24, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
    cv2.putText(depth_image, threshold_text, (24, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(depth_image, roi_text, (24, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(depth_image, valid_text, (24, 172), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imshow(WINDOW_NAME, depth_image)
    key = cv2.waitKey(1)
    return key not in (ord("q"), ESC_KEY)


def run(port: str, baud_rate: int, show_window: bool) -> None:
    pipeline = Pipeline()
    config = Config()

    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    with serial.Serial(port, baud_rate, timeout=1) as serial_connection:
        time.sleep(SERIAL_STARTUP_DELAY_SECONDS)

        pipeline.start(config)
        last_led_state = None
        try:
            while True:
                frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
                if frames is None:
                    continue

                depth_map_mm = read_depth_map_mm(frames)
                if depth_map_mm is None:
                    continue

                trigger_depth_map = get_trigger_depth_map(depth_map_mm)
                lowest_height = get_lowest_valid_height_mm(trigger_depth_map)
                led_on = lowest_height is not None and lowest_height < LOWEST_HEIGHT_MM
                if not SEND_ONLY_ON_CHANGE or led_on != last_led_state:
                    send_led_state(serial_connection, led_on)
                    print("Sent:", ON_MESSAGE.strip() if led_on else OFF_MESSAGE.strip())
                    last_led_state = led_on

                if show_window and not render_depth_window(depth_map_mm, lowest_height, led_on):
                    break
        finally:
            pipeline.stop()
            if show_window:
                cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send LED_ON or LED_OFF to an Arduino based on the closest valid depth value."
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Arduino serial port, such as /dev/cu.usbmodem1101. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=BAUD_RATE,
        help=f"Arduino serial baud rate. Default: {BAUD_RATE}.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run serial triggering without showing the OpenCV depth window.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    port = args.port or find_arduino_port()
    print(f"Using Arduino serial port: {port}")
    print(f"Depth threshold: {LOWEST_HEIGHT_MM} mm")
    run(port, args.baud, show_window=SHOW_DEPTH_WINDOW and not args.no_window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
