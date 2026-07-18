import time

import numpy as np

from pyorbbecsdk import Config, Context, OBSensorType, Pipeline


def print_devices():
    context = Context()
    devices = context.query_devices()
    count = devices.get_count()
    print(f"SDK device count: {count}")

    for index in range(count):
        device = devices.get_device_by_index(index)
        info = device.get_device_info()
        print(f"Device {index}: {info}")

    return count


def main():
    if print_devices() == 0:
        print("No Orbbec device found by the SDK.")
        print("On macOS, try running this same command with sudo.")
        return 2

    pipeline = Pipeline()
    config = Config()

    profile_list = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = profile_list.get_default_video_stream_profile()
    print(f"Depth profile: {depth_profile}")
    config.enable_stream(depth_profile)

    pipeline.start(config)
    try:
        deadline = time.time() + 10
        while time.time() < deadline:
            frames = pipeline.wait_for_frames(500)
            if frames is None:
                continue

            depth_frame = frames.get_depth_frame()
            if depth_frame is None:
                continue

            width = depth_frame.get_width()
            height = depth_frame.get_height()
            scale = depth_frame.get_depth_scale()
            depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
            depth_mm = depth.astype(np.float32) * scale
            valid = depth_mm[depth_mm > 0]

            center = depth_mm[height // 2, width // 2]
            print("Depth frame received")
            print(f"Shape: {height}x{width}")
            print(f"Scale: {scale}")
            print(f"Center depth: {center:.1f} mm")
            print(f"Valid pixels: {valid.size}/{depth_mm.size}")
            if valid.size:
                print(f"Median valid depth: {float(np.median(valid)):.1f} mm")
            return 0

        print("Timed out waiting for a depth frame.")
        return 3
    finally:
        pipeline.stop()


if __name__ == "__main__":
    raise SystemExit(main())
