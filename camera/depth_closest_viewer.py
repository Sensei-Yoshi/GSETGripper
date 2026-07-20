from typing import Optional, Tuple

import cv2
import numpy as np

from pyorbbecsdk import Config, OBSensorType, Pipeline

from depth_closest import find_closest_point, read_depth_map_mm, MIN_VALID_DEPTH_MM, MAX_VALID_DEPTH_MM

FRAME_TIMEOUT_MS = 500
WINDOW_NAME = "Orbbec Closest Point Viewer"
ESC_KEY = 27
MARKER_COLOR = (0, 255, 0)
SIDEBAR_WIDTH = 260


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


def render_frame(depth_map_mm: np.ndarray, closest: Optional[Tuple[float, Tuple[int, int]]]) -> np.ndarray:
    depth_image = depth_map_to_color_image(depth_map_mm)
    height = depth_image.shape[0]

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

    return np.hstack((depth_image, sidebar))


def run() -> None:
    pipeline = Pipeline()
    config = Config()

    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(depth_profile)

    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames(FRAME_TIMEOUT_MS)
            if frames is None:
                continue

            depth_map_mm = read_depth_map_mm(frames)
            if depth_map_mm is None:
                continue

            closest = find_closest_point(depth_map_mm)
            frame_image = render_frame(depth_map_mm, closest)

            cv2.imshow(WINDOW_NAME, frame_image)
            key = cv2.waitKey(1)
            if key in (ord("q"), ESC_KEY):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
