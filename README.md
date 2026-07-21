# GSETGripper

## Orbbec Astra+ Depth Camera Setup on macOS

This project uses an Orbbec Astra+ camera for depth sensing. On modern macOS systems, use Orbbec SDK v1 for Astra+. Astra+ is not supported by Orbbec SDK v2.

The helper scripts in this repository assume this workspace layout:

```text
GSET/
  GSETGripper/
    camera/
      depth_closest.py
      depth_closest_macos.sh
      depth_closest_viewer.py
      depth_closest_viewer_macos.sh
      depth_closest_read.py
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

Connect the Astra+ over USB 3.0, then run the closest-point script from the workspace root:

```bash
cd /path/to/GSET
GSETGripper/camera/depth_closest_macos.sh
```

The script uses `sudo` because the Orbbec SDK may not see the depth device on macOS without elevated USB access.

A successful run should print the closest valid depth pixel in the center ROI, e.g.:

```text
Closest depth: 842.0 mm at pixel (row=241, col=318)
```

The message `Current device does not support frame sync!` can appear with Astra+. It is not fatal if the script still receives a depth frame.

## Find Object Height

Once the camera is validated, use the live viewer to capture a reading, then convert it to a real-world object height.

Files:

```text
GSETGripper/
  camera/
    depth_closest_viewer.py
    depth_closest_viewer_macos.sh
    depth_closest_read.py
```

Run the viewer from the workspace root:

```bash
cd /path/to/GSET
GSETGripper/camera/depth_closest_viewer_macos.sh
```

This opens a live window showing the closest valid depth point in the center ROI. Press `c` to capture the current reading to `camera/captures/latest_capture.json`. Press `q` or `Esc` to quit.

Then read the captured value and convert it to object height above the ground:

```bash
python GSETGripper/camera/depth_closest_read.py
```

`depth_closest_read.py` applies the rig's fixed geometry (camera mount height and horizontal camera-to-object offset, defined at the top of the file) to turn the raw slant-distance depth reading into `compute_object_height_mm()`, the object's height above the ground.

## Custom SDK Location

If `pyorbbecsdk-v1` is not next to `GSETGripper`, set `ORBBEC_SDK_DIR`:

```bash
export ORBBEC_SDK_DIR=/absolute/path/to/pyorbbecsdk-v1
GSETGripper/camera/depth_closest_macos.sh
GSETGripper/camera/depth_closest_viewer_macos.sh
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
