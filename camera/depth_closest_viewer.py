import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from pyorbbecsdk import Config, OBSensorType, Pipeline

try:
    from GSETGripper.camera.depth_closest import (
        CENTER_ROI_RADIUS_PIXELS,
        MAX_VALID_DEPTH_MM,
        MIN_VALID_DEPTH_MM,
        find_closest_point,
        get_center_roi_mask,
        read_depth_map_mm,
    )
    from GSETGripper.camera.depth_closest_read import (
        DEFAULT_BAUD_RATE,
        SERIAL_STARTUP_DELAY_SECONDS,
        compute_object_height_mm,
        find_arduino_port,
    )
except ImportError:
    # Launched as `python camera/depth_closest_viewer.py` (e.g. via the macos
    # shell script): the camera/ dir itself is on sys.path, not the repo parent.
    from depth_closest import (
        CENTER_ROI_RADIUS_PIXELS,
        MAX_VALID_DEPTH_MM,
        MIN_VALID_DEPTH_MM,
        find_closest_point,
        get_center_roi_mask,
        read_depth_map_mm,
    )
    from depth_closest_read import (
        DEFAULT_BAUD_RATE,
        SERIAL_STARTUP_DELAY_SECONDS,
        compute_object_height_mm,
        find_arduino_port,
    )

FRAME_TIMEOUT_MS = 500
WINDOW_NAME = "Orbbec Closest Point Viewer"
ESC_KEY = 27
CAPTURE_KEY = ord("c")
MARKER_COLOR = (0, 255, 0)
ROI_COLOR = (255, 255, 255)
CAPTURE_FLASH_COLOR = (0, 255, 255)
CAPTURE_FLASH_SECONDS = 1.0
SIDEBAR_WIDTH = 260
SMOOTHING_WINDOW_FRAMES = 5
CAPTURE_FILE_PATH = Path(__file__).parent / "captures" / "latest_capture.json"
REPLY_OK_COLOR = (0, 255, 0)
REPLY_WARN_COLOR = (0, 165, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live closest-depth viewer. Press 'c' to capture AND send the "
        "computed object height straight to the Arduino over serial."
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Arduino serial port, such as /dev/cu.usbmodem1101. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD_RATE,
        help=f"Arduino serial baud rate. Default: {DEFAULT_BAUD_RATE}.",
    )
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help="Capture to file only; do not connect to the Arduino.",
    )
    return parser.parse_args()


def open_arduino(port_preference: Optional[str], baud: int):
    """Open one long-lived serial connection. The board resets when the port
    opens, so the carriage must be parked at the top of travel at this moment;
    every later capture sends an absolute height against that zero."""
    import serial

    port = find_arduino_port(port_preference)
    connection = serial.Serial(port, baud, timeout=0)
    time.sleep(SERIAL_STARTUP_DELAY_SECONDS)  # allow the board to reset
    return connection, port


def send_object_height(connection, height_mm: float) -> None:
    connection.write(f"Z {height_mm:.1f}\n".encode("ascii"))
    connection.flush()


def drain_arduino_replies(connection, buffer: bytearray) -> list:
    """Non-blocking read of complete reply lines (DONE Z / WARN ... / ERR ...)."""
    if connection is None:
        return []
    waiting = connection.in_waiting
    if waiting:
        buffer.extend(connection.read(waiting))
    lines = []
    while b"\n" in buffer:
        raw, _, rest = bytes(buffer).partition(b"\n")
        buffer[:] = rest
        text = raw.decode("ascii", errors="ignore").strip()
        if text:
            lines.append(text)
    return lines


def depth_map_to_color_image(depth_map_mm: np.ndarray) -> np.ndarray:
    valid_mask = (
        np.isfinite(depth_map_mm)
        & (depth_map_mm >= MIN_VALID_DEPTH_MM)
        & (depth_map_mm <= MAX_VALID_DEPTH_MM)
    )
    clipped_depth = np.clip(depth_map_mm, MIN_VALID_DEPTH_MM, MAX_VALID_DEPTH_MM)
    normalized_depth = (
        255
        - (clipped_depth - MIN_VALID_DEPTH_MM) * 255 / (MAX_VALID_DEPTH_MM - MIN_VALID_DEPTH_MM)
    ).astype(np.uint8)
    normalized_depth[~valid_mask] = 0

    depth_image = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
    depth_image[~valid_mask] = (0, 0, 0)
    return depth_image


def smooth_closest_point(
    history: "deque[Tuple[float, Tuple[int, int]]]",
) -> Optional[Tuple[float, Tuple[int, int]]]:
    if not history:
        return None

    depths = [depth for depth, _ in history]
    rows = [position[0] for _, position in history]
    cols = [position[1] for _, position in history]
    return float(np.median(depths)), (int(np.median(rows)), int(np.median(cols)))


def save_capture(closest: Tuple[float, Tuple[int, int]]) -> None:
    closest_depth_mm, (row, col) = closest
    CAPTURE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "depth_mm": closest_depth_mm,
        "row": row,
        "col": col,
        "captured_at_epoch": time.time(),
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    CAPTURE_FILE_PATH.write_text(json.dumps(payload, indent=2))


def render_frame(
    depth_map_mm: np.ndarray,
    closest: Optional[Tuple[float, Tuple[int, int]]],
    capture_flash_active: bool = False,
    last_sent_text: Optional[str] = None,
    last_reply_text: Optional[str] = None,
) -> np.ndarray:
    depth_image = depth_map_to_color_image(depth_map_mm)
    height, width = depth_map_mm.shape

    roi_center = (width // 2, height // 2)
    cv2.circle(depth_image, roi_center, CENTER_ROI_RADIUS_PIXELS, ROI_COLOR, 1)

    if closest is not None:
        closest_depth_mm, (row, col) = closest
        cv2.drawMarker(
            depth_image,
            (col, row),
            MARKER_COLOR,
            markerType=cv2.MARKER_CROSS,
            markerSize=20,
            thickness=2,
        )
        cv2.circle(depth_image, (col, row), 10, MARKER_COLOR, 2)
        distance_text = f"{closest_depth_mm:.0f} mm"
    else:
        distance_text = "no signal"

    sidebar = np.zeros((height, SIDEBAR_WIDTH, 3), dtype=np.uint8)
    cv2.putText(sidebar, "Closest point", (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(sidebar, distance_text, (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, MARKER_COLOR, 2)

    if closest is not None:
        _, (row, col) = closest
        cv2.putText(sidebar, f"row: {row}", (16, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(sidebar, f"col: {col}", (16, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if last_sent_text is not None:
        cv2.putText(sidebar, f"sent: {last_sent_text}", (16, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CAPTURE_FLASH_COLOR, 1)
    if last_reply_text is not None:
        reply_color = REPLY_OK_COLOR if last_reply_text.startswith("DONE") else REPLY_WARN_COLOR
        cv2.putText(sidebar, last_reply_text, (16, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, reply_color, 1)

    hint_text = "CAPTURED" if capture_flash_active else "Press 'c' to capture"
    hint_color = CAPTURE_FLASH_COLOR if capture_flash_active else (160, 160, 160)
    cv2.putText(sidebar, hint_text, (16, height - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hint_color, 2)

    return np.hstack((depth_image, sidebar))


def run(args: argparse.Namespace) -> None:
    connection = None
    port = None
    if not args.no_serial:
        try:
            connection, port = open_arduino(args.port, args.baud)
            print(f"Arduino connected on {port}. The board just reset -- carriage position is now zero, "
                  "so it must be parked at the top of travel.")
        except Exception as error:
            print(
                f"Arduino not connected ({error}). Captures will still be saved, but heights will not be sent.",
                file=sys.stderr,
            )

    pipeline = Pipeline()
    config = Config()

    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    history: "deque[Tuple[float, Tuple[int, int]]]" = deque(maxlen=SMOOTHING_WINDOW_FRAMES)
    capture_flash_until = 0.0
    reply_buffer = bytearray()
    last_sent_text: Optional[str] = None
    last_reply_text: Optional[str] = None

    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)

            for reply in drain_arduino_replies(connection, reply_buffer):
                print(f"[arduino] {reply}")
                last_reply_text = reply

            if frames is None:
                continue

            depth_map_mm = read_depth_map_mm(frames)
            if depth_map_mm is None:
                continue

            roi_mask = get_center_roi_mask(depth_map_mm)
            raw_closest = find_closest_point(depth_map_mm, roi_mask=roi_mask)
            if raw_closest is not None:
                history.append(raw_closest)

            closest = smooth_closest_point(history)
            capture_flash_active = time.time() < capture_flash_until
            frame_image = render_frame(
                depth_map_mm, closest, capture_flash_active, last_sent_text, last_reply_text
            )

            cv2.imshow(WINDOW_NAME, frame_image)
            key = cv2.waitKey(1)
            if key in (ord("q"), ESC_KEY):
                break
            if key == CAPTURE_KEY:
                if closest is None:
                    print("Capture skipped: no valid closest point right now.")
                    continue

                save_capture(closest)
                print(f"Captured: {closest[0]:.1f} mm -> {CAPTURE_FILE_PATH}")
                capture_flash_until = time.time() + CAPTURE_FLASH_SECONDS

                object_height_mm = compute_object_height_mm(closest[0])
                if object_height_mm is None:
                    print(
                        f"Object height undefined: depth {closest[0]:.1f} mm is shorter than the "
                        "fixed horizontal camera distance. Nothing sent.",
                        file=sys.stderr,
                    )
                elif connection is not None:
                    send_object_height(connection, object_height_mm)
                    last_sent_text = f"Z {object_height_mm:.1f}"
                    last_reply_text = None
                    print(f"Sent: Z {object_height_mm:.1f} -> {port}")
                else:
                    print(
                        f"Object height: {object_height_mm:.1f} mm (Arduino not connected; not sent)"
                    )
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        if connection is not None:
            connection.close()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
