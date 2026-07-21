import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

CAPTURE_FILE_PATH = Path(__file__).parent / "captures" / "latest_capture.json"
DEFAULT_MAX_AGE_SECONDS = 30.0
DEFAULT_BAUD_RATE = 9600
SERIAL_STARTUP_DELAY_SECONDS = 2.0

# Fixed rig geometry (right triangle: camera -> horizontal offset x -> object top,
# hypotenuse d = sensor depth reading). h = sqrt(d^2 - x^2) is how far the object's
# top sits below the camera; object height = total camera mount height - h.
CAMERA_HEIGHT_MM = 1193.8  # 3 ft 11 in (47 in) camera mount height above ground
CAMERA_HORIZONTAL_DISTANCE_MM = 444.5  # 17.5 in horizontal camera-to-object distance x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the most recently captured closest-depth value from depth_closest_viewer.py, "
        "and send the computed object height to the Arduino over serial."
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print only the depth value in mm, with no extra text.",
    )
    parser.add_argument(
        "--max-age",
        type=float,
        default=DEFAULT_MAX_AGE_SECONDS,
        help=f"Warn (and exit 2) if the capture is older than this many seconds. Default: {DEFAULT_MAX_AGE_SECONDS}.",
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
        help="Compute and print the object height without connecting to the Arduino.",
    )
    return parser.parse_args()


def find_arduino_port(preferred: Optional[str] = None) -> str:
    """Autodetect an Arduino-like serial port (mirrors force_prediction.hardware)."""
    if preferred:
        return preferred

    from serial.tools import list_ports

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
            "Multiple possible Arduino serial ports found. Pass one explicitly with --port. "
            f"Candidates: {', '.join(candidates)}"
        )

    available = ", ".join(port.device for port in list_ports.comports()) or "none"
    raise RuntimeError(
        f"No Arduino serial port found. Pass one explicitly with --port. Available ports: {available}"
    )


def send_object_height_mm(port: str, baud: int, height_mm: float) -> None:
    import serial

    with serial.Serial(port, baud, timeout=1) as connection:
        time.sleep(SERIAL_STARTUP_DELAY_SECONDS)  # allow the board to reset
        connection.write(f"Z {height_mm:.1f}\n".encode("ascii"))
        connection.flush()


def compute_object_height_mm(slant_distance_mm: float) -> float | None:
    """Return object height above ground, or None if the geometry is invalid
    (slant distance d shorter than the fixed horizontal distance x)."""
    if slant_distance_mm < CAMERA_HORIZONTAL_DISTANCE_MM:
        return None
    height_below_camera_mm = (slant_distance_mm**2 - CAMERA_HORIZONTAL_DISTANCE_MM**2) ** 0.5
    return CAMERA_HEIGHT_MM - height_below_camera_mm


def main() -> int:
    args = parse_args()

    if not CAPTURE_FILE_PATH.exists():
        print(f"No capture found at {CAPTURE_FILE_PATH}. Run depth_closest_viewer.py and press 'c' first.", file=sys.stderr)
        return 1

    try:
        payload = json.loads(CAPTURE_FILE_PATH.read_text())
        depth_mm = float(payload["depth_mm"])
        captured_at_epoch = float(payload["captured_at_epoch"])
    except (json.JSONDecodeError, KeyError, ValueError) as error:
        print(f"Capture file at {CAPTURE_FILE_PATH} is malformed: {error}", file=sys.stderr)
        return 1

    age_seconds = time.time() - captured_at_epoch

    object_height_mm = compute_object_height_mm(depth_mm)

    if args.raw:
        if object_height_mm is None:
            print(f"{depth_mm:.1f} nan")
        else:
            print(f"{depth_mm:.1f} {object_height_mm:.1f}")
    else:
        print(f"Closest depth: {depth_mm:.1f} mm (captured {age_seconds:.1f}s ago)")
        if object_height_mm is None:
            print(
                f"Object height: undefined (depth {depth_mm:.1f} mm is shorter than the "
                f"fixed horizontal distance {CAMERA_HORIZONTAL_DISTANCE_MM:.1f} mm)",
                file=sys.stderr,
            )
        else:
            print(f"Object height: {object_height_mm:.1f} mm")

    if object_height_mm is not None and not args.no_serial:
        try:
            port = find_arduino_port(args.port)
            send_object_height_mm(port, args.baud, object_height_mm)
            print(f"Sent: Z {object_height_mm:.1f} -> {port}", file=sys.stderr)
        except Exception as error:
            print(f"Failed to send height to Arduino: {error}", file=sys.stderr)
            return 1

    if age_seconds > args.max_age:
        print(
            f"Warning: capture is {age_seconds:.1f}s old, older than the {args.max_age:.1f}s freshness threshold.",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
