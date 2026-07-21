# Material Segmenter

Take a picture from the camera, cut the objects out of the background at
high quality, label what each object is and what it's made of, and measure
its surface properties with Marigold intrinsics.

- **Hybrid segmentation + analysis:** one Gemini API call
  (`gemini-3-flash` by default) detects every object in the capture and
  returns, per object, a bounding box, a coarse segmentation mask, and a
  full JSON profile: name, category, primary/secondary materials, colors,
  texture, transparency, rigidity, surface finish, condition, estimated
  size, visible text/branding, distinctive features, typical use,
  recyclability, and a confidence score (saved as
  `capture_NN_objects.json`). Each Gemini mask is then intersected with a
  full-resolution rembg (ISNet) matte: Gemini decides *what* the objects
  are — including splitting touching objects — and rembg supplies the
  crisp full-resolution edges. If Gemini misses an object rembg found (or
  the API call fails), the rembg mask is used alone as a fallback.
- **Intrinsics:** Marigold IID appearance model (fp16 on MPS) decomposes
  the capture into albedo, roughness, and metallicity; per-object means are
  computed over each object's segmentation mask and shown in its label.
  Skip with `--no-intrinsics` (faster, much less memory).

## Setup (already done in `.venv`)

```sh
python3.12 -m venv .venv
.venv/bin/pip install opencv-python torch torchvision pillow transformers "rembg[cpu]" diffusers accelerate
```

## API key

Create a `.env` file next to the script (get a key at
https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=your-key-here
# optional, defaults to gemini-3-flash:
# GEMINI_MODEL=gemini-3-flash-preview
```

## Run

```sh
.venv/bin/python material_segmenter.py             # camera preview
.venv/bin/python material_segmenter.py --camera 1  # different camera
.venv/bin/python material_segmenter.py --image photo.jpg  # existing photo
```

In the preview window press **SPACE** to take a picture. The result window
shows the segmented portion (background replaced by a checkerboard) with an
"object: material | rough X metal Y" label per object, above the albedo /
roughness / metallicity maps masked to the objects. Press any key to return
to the preview, **q** to quit. Each capture saves:

- `capture_NN_segmented.png` — the labeled cutout view + intrinsics panel
- `capture_NN_cutout.png` — the objects on a transparent background
- `capture_NN_objects.json` — full Gemini profile per object, plus its
  Marigold roughness/metallicity means and bounding box
- `capture_NN_albedo.png`, `capture_NN_roughness.png`,
  `capture_NN_metallicity.png` — intrinsics maps (objects only)

Notes:
- Marigold runs on the first capture, not at startup (lazy-loaded); expect
  ~20-25 s per capture for the intrinsics pass on an M1.
- Marigold's diffusion sampling is seeded (fixed seed in
  `IntrinsicsAnalyzer.analyze`), so roughness/metallicity values are
  reproducible for the same image; without it they vary a few percent
  between runs.
- First run downloads model weights (cached in `~/.cache` / `~/.u2net`). If
  you hit `CERTIFICATE_VERIFY_FAILED`, prefix the command with
  `SSL_CERT_FILE=$(.venv/bin/python -m certifi)`.
- macOS asks for camera permission for your terminal the first time
  (System Settings → Privacy & Security → Camera).
