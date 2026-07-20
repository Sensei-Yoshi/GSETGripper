import time
from typing import Optional, Tuple

import numpy as np

from pyorbbecsdk import Config, OBSensorType, Pipeline

MIN_VALID_DEPTH_MM = 20
MAX_VALID_DEPTH_MM = 10000
FRAME_TIMEOUT_MS = 500


def read_depth_map_mm(frames) -> Optional[np.ndarray]:
    depth_frame = frames.get_depth_frame()
    if depth_frame is None:
        return None

    width = depth_frame.get_width()
    height = depth_frame.get_height()
    scale = depth_frame.get_depth_scale()
    depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
    return depth.astype(np.float32) * scale


def find_closest_point(depth_map_mm: np.ndarray) -> Optional[Tuple[float, Tuple[int, int]]]:
    valid_mask = (
        np.isfinite(depth_map_mm)
        & (depth_map_mm >= MIN_VALID_DEPTH_MM)
        & (depth_map_mm <= MAX_VALID_DEPTH_MM)
    )
    if not valid_mask.any():
        return None

    masked_depth = np.where(valid_mask, depth_map_mm, np.inf)
    flat_index = int(np.argmin(masked_depth))
    row, col = np.unravel_index(flat_index, masked_depth.shape)
    return float(masked_depth[row, col]), (int(row), int(col))


def main() -> int:
    pipeline = Pipeline()
    config = Config()

    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    pipeline.start(config)
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
            if frames is None:
                continue

            depth_map_mm = read_depth_map_mm(frames)
            if depth_map_mm is None:
                continue

            result = find_closest_point(depth_map_mm)
            if result is None:
                print("No valid depth pixels in frame.")
                return 3

            closest_depth_mm, (row, col) = result
            print(f"Closest depth: {closest_depth_mm:.1f} mm at pixel (row={row}, col={col})")
            return 0

        print("Timed out waiting for a depth frame.")
        return 3
    finally:
        pipeline.stop()


if __name__ == "__main__":
    raise SystemExit(main())
