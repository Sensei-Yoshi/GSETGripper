# GSETGripper

## Orbbec Astra+ Depth Camera Setup on macOS

This project uses an Orbbec Astra+ camera for depth sensing. On modern macOS systems, use Orbbec SDK v1 for Astra+. Astra+ is not supported by Orbbec SDK v2.

The helper scripts in this repository assume this workspace layout:

```text
GSET/
  GSETGripper/
    camera/
      depth_probe.py
      probe_depth_macos.sh
      depth_viewer_macos.sh
  pyorbbecsdk-v1/
```

If the SDK folder is somewhere else, set `ORBBEC_SDK_DIR` before running the scripts.

## Prerequisites

Install the macOS command line tools and Homebrew packages:

```bash
xcode-select --install
brew install cmake libusb
```

Use Python 3.11 if possible. This setup was tested with Python 3.11 on Apple Silicon macOS.

## Clone the Orbbec Python SDK

From the workspace root that contains `GSETGripper`:

```bash
cd /path/to/GSET
git clone https://github.com/orbbec/pyorbbecsdk.git pyorbbecsdk-v1
cd pyorbbecsdk-v1
git checkout main
```

Use the `main` branch for Astra+ support. Do not use the `v2-main` branch for Astra+.

## Create the Python Environment

From inside `pyorbbecsdk-v1`:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install "pybind11==2.11.0" "numpy<2.0" "opencv-python<4.12" "pyserial"
```

The NumPy pin is intentional. The Orbbec Python wrapper expects NumPy 1.x, and newer OpenCV wheels may otherwise install NumPy 2.x.

## Build and Install the SDK Binding

From inside `pyorbbecsdk-v1` with the virtual environment active:

```bash
cmake -S . -B build \
  -DPython3_EXECUTABLE="$PWD/venv/bin/python" \
  -Dpybind11_DIR="$("$PWD/venv/bin/python" -m pybind11 --cmakedir)"

cmake --build build -j8
cmake --install build
```

After installation, the compiled Python module and Orbbec dynamic libraries should exist under:

```text
pyorbbecsdk-v1/install/lib/
```

To make `pyorbbecsdk` importable after activating the virtual environment, add the SDK install folder to the venv path:

```bash
python -c 'import pathlib, sysconfig; pathlib.Path(sysconfig.get_paths()["purelib"], "pyorbbecsdk_local_install.pth").write_text(str(pathlib.Path.cwd() / "install/lib") + "\n")'
```

After that, this should work from the workspace root:

```bash
source pyorbbecsdk-v1/venv/bin/activate
python3 -c "import pyorbbecsdk; print('pyorbbecsdk import ok')"
```

## Validate the Depth Camera

Connect the Astra+ over USB 3.0, then run the probe script from the workspace root:

```bash
cd /path/to/GSET
GSETGripper/camera/probe_depth_macos.sh
```

The script uses `sudo` because the Orbbec SDK may not see the depth device on macOS without elevated USB access.

A successful run should show:

```text
SDK device count: 1
Device 0: DeviceInfo(name=Astra+ ...)
Depth frame received
Shape: 480x640
Valid pixels: ...
Median valid depth: ...
```

The message `Current device does not support frame sync!` can appear with Astra+. It is not fatal if the script still receives a depth frame.

## Run the Live Depth Viewer

After the probe works, run:

```bash
cd /path/to/GSET
GSETGripper/camera/depth_viewer_macos.sh
```

This opens the Orbbec SDK depth viewer using OpenCV. Press `q` or `Esc` to quit.

## Arduino LED Serial Trigger

The camera trigger code shows a live depth window and sends serial messages to an Arduino based on the closest valid value in the depth map.

Files:

```text
GSETGripper/
  camera/
    depth_serial_trigger.py
    depth_serial_trigger_macos.sh
  arduino/
    default_led_serial/
      default_led_serial.ino
```

The threshold is defined at the top of `camera/depth_serial_trigger.py`:

```python
LOWEST_HEIGHT_MM = 2500
```

The trigger region is also defined at the top of `camera/depth_serial_trigger.py`:

```python
CENTER_ROI_RADIUS_PIXELS = 120
```

In this script, depth means camera-to-object distance in millimeters. It does not mean vertical height from the table or floor.

The function `lowest_height_is_below_threshold(depth_map_mm)` returns `True` when the closest valid depth value is less than `LOWEST_HEIGHT_MM`. The running trigger script applies that calculation only inside the circular center region drawn on the depth window. When the result is `True`, the Python script sends `LED_ON` over serial. When the result is `False`, it sends `LED_OFF`.

The depth window overlays:

```text
Min depth: current closest valid depth in millimeters
Center depth: depth at the center crosshair, or invalid
Trigger: True or False
Threshold: configured threshold in millimeters
ROI radius: circular trigger region radius in pixels
Valid pixels: valid depth pixels in the frame
```

Black pixels in the window are invalid depth pixels. If a hand is too close to the camera, too reflective, moving quickly, or near a depth edge, the camera may return invalid pixels for that hand. In that case the closest valid depth may come from the background instead of the hand.

The script only sends a message when the state changes. This prevents unnecessary serial traffic while the camera is running.

### Upload the Arduino Sketch

Open this sketch in the Arduino IDE:

```text
GSETGripper/arduino/default_led_serial/default_led_serial.ino
```

Select the correct board and port, then upload it. The sketch uses `LED_BUILTIN`, which is the default test LED on most Arduino boards.

The Arduino expects:

```text
Baud rate: 9600
Messages: LED_ON, LED_OFF
Line ending: newline
```

### Run the Trigger

With the Astra+ and Arduino connected, run from the workspace root:

```bash
cd /path/to/GSET
GSETGripper/camera/depth_serial_trigger_macos.sh
```

This opens the live depth window. Press `q` or `Esc` to quit.

If auto-detection finds the wrong serial port, pass the Arduino port explicitly:

```bash
GSETGripper/camera/depth_serial_trigger_macos.sh --port /dev/cu.usbmodem1101
```

To run the serial trigger without the depth window:

```bash
GSETGripper/camera/depth_serial_trigger_macos.sh --no-window
```

To test the same camera, ROI, min-depth, visual overlay, and trigger logic without an Arduino connected:

```bash
GSETGripper/camera/depth_serial_trigger_macos.sh --no-serial
```

In `--no-serial` mode, the script prints `Would send: LED_ON` or `Would send: LED_OFF` instead of opening a serial connection.

To list available serial ports on macOS:

```bash
ls /dev/cu.*
```

## Custom SDK Location

If `pyorbbecsdk-v1` is not next to `GSETGripper`, set `ORBBEC_SDK_DIR`:

```bash
export ORBBEC_SDK_DIR=/absolute/path/to/pyorbbecsdk-v1
GSETGripper/camera/probe_depth_macos.sh
GSETGripper/camera/depth_viewer_macos.sh
```

## Troubleshooting

If the SDK reports `SDK device count: 0`, run the provided macOS helper scripts so the SDK is launched with `sudo`.

If the RGB camera works in normal video apps but depth does not work, that is expected until the Orbbec SDK can access the depth endpoint. The RGB stream is exposed as a standard camera interface; the depth stream requires the SDK.

If the build accidentally uses a different Python version, delete the CMake cache and reconfigure with `Python3_EXECUTABLE` pointing at the virtual environment:

```bash
cd /path/to/GSET/pyorbbecsdk-v1
rm -rf build
cmake -S . -B build \
  -DPython3_EXECUTABLE="$PWD/venv/bin/python" \
  -Dpybind11_DIR="$("$PWD/venv/bin/python" -m pybind11 --cmakedir)"
```
