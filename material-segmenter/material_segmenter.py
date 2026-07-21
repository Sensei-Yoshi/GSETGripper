"""Snapshot material segmenter: take a picture, cut objects out of the
background at high quality, and label each object's material.

Workflow:
  1. A live preview window shows the camera. Press SPACE to take a picture.
  2. Hybrid segmentation: one Gemini API call detects every object and
     returns, per object, a bounding box, a coarse segmentation mask, and a
     rich JSON profile (name, category, materials, colors, texture,
     transparency, surface finish, condition, size estimate, visible text,
     and more). In parallel, rembg (ISNet matting) computes a full-resolution
     foreground matte. Each Gemini mask is intersected with the matte, so
     Gemini decides WHAT the objects are and rembg decides exactly WHERE
     their edges lie.
  3. Requires a .env file next to this script with GEMINI_API_KEY=<key>
     (get one at https://aistudio.google.com/apikey). Model defaults to
     gemini-3-flash; override with GEMINI_MODEL in the .env.
  4. The capture is decomposed with Marigold IID (appearance model) into
     albedo, roughness, and metallicity; per-object stats are computed over
     each object's segmentation mask.
  5. A result window shows the segmented portion (objects on a checkerboard,
     background removed) with material labels, above the albedo / roughness /
     metallicity maps masked to the objects. The cutout is also saved as a
     transparent PNG.

Usage:
  python material_segmenter.py               # camera preview, SPACE to shoot
  python material_segmenter.py --camera 1    # another camera
  python material_segmenter.py --image path  # segment an existing photo

Keys:  SPACE = capture,  any key in result view = back to preview,  q = quit.
"""

import argparse
import base64
import json
import os
import sys
import time

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from rembg import remove, new_session

# Minimum fraction of the frame an object region must cover to be labeled.
MIN_REGION_FRAC = 0.005
# Alpha threshold for turning the soft matte into object regions.
ALPHA_THRESH = 120


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_env_file(path: str = ".env"):
    """Load KEY=VALUE lines from a .env file into the environment
    (existing environment variables win)."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(),
                                      value.strip().strip("'\""))
    except FileNotFoundError:
        pass


SCENE_PROMPT = """\
Detect and segment every distinct physical object in this photo. Ignore the
background itself (walls, floor, table surface, sky, large furniture that
merely supports the objects).
Return ONLY a JSON array with one entry per object. Each entry must contain
"box_2d": [ymin, xmin, ymax, xmax] normalized to 0-1000, "mask": a
base64-encoded PNG segmentation mask for the box region, and this profile
(use null for anything you cannot determine):
{
  "name": "short object name",
  "category": "broad category, e.g. kitchenware, tool, electronics, plant, animal",
  "primary_material": "the main material it is made of",
  "secondary_materials": ["other visible materials"],
  "colors": ["dominant colors"],
  "texture": "short texture description",
  "transparency": "opaque | translucent | transparent",
  "rigidity": "rigid | flexible | soft",
  "surface_finish": "e.g. matte, glossy, brushed, polished, rough",
  "condition": "e.g. new, worn, scratched, dirty, damaged",
  "estimated_size_cm": {"width": 0, "height": 0},
  "visible_text_or_branding": "any readable text or logos",
  "distinctive_features": ["notable details"],
  "typical_use": "one sentence",
  "recyclable": true,
  "confidence": 0.0
}"""


class GeminiClassifier:
    """Scene analysis via the Gemini API: the whole capture is sent once and
    Gemini returns every object with a bounding box, a segmentation mask,
    and a structured JSON profile."""

    def __init__(self, model: str | None = None):
        self.api_key = (os.environ.get("GEMINI_API_KEY")
                        or os.environ.get("GOOGLE_API_KEY"))
        if not self.api_key:
            sys.exit(
                "No Gemini API key found. Create a .env file next to this "
                "script containing:\n"
                "  GEMINI_API_KEY=your-key-here\n"
                "Get a key at https://aistudio.google.com/apikey")
        self.model = (model or os.environ.get("GEMINI_MODEL")
                      or "gemini-3-flash")

    def analyze_scene(self, frame_rgb: np.ndarray) -> list[dict]:
        """Return a list of detections, each a dict with box_2d, mask, and
        the object profile. Empty list on API failure."""
        ok, jpg = cv2.imencode(".jpg",
                               cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, 92])
        payload = {
            "contents": [{"parts": [
                {"inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(jpg.tobytes()).decode()}},
                {"text": SCENE_PROMPT},
            ]}],
            "generationConfig": {
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        }
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               f"{self.model}:generateContent")
        try:
            resp = requests.post(
                url, json=payload, timeout=120,
                headers={"x-goog-api-key": self.api_key})
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            detections = json.loads(text)
            return detections if isinstance(detections, list) else []
        except requests.HTTPError:
            detail = resp.text[:300]
            print(f"  Gemini API error {resp.status_code}: {detail}",
                  file=sys.stderr)
        except Exception as e:
            print(f"  Gemini request failed: {e}", file=sys.stderr)
        return []


def gemini_mask_to_frame(det: dict, frame_w: int, frame_h: int):
    """Decode a detection's base64 PNG mask and place it into a full-frame
    binary mask (uint8, 255 inside the object). None if undecodable."""
    try:
        b64 = det.get("mask") or ""
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64)
        m = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
        y0, x0, y1, x1 = det["box_2d"]
        x0 = int(x0 / 1000 * frame_w)
        x1 = int(x1 / 1000 * frame_w)
        y0 = int(y0 / 1000 * frame_h)
        y1 = int(y1 / 1000 * frame_h)
        if m is None or x1 <= x0 or y1 <= y0:
            return None
        m = cv2.resize(m, (x1 - x0, y1 - y0))
        full = np.zeros((frame_h, frame_w), np.uint8)
        full[y0:y1, x0:x1] = np.where(m > 127, 255, 0).astype(np.uint8)
        return full
    except Exception as e:
        print(f"  could not decode a Gemini mask: {e}", file=sys.stderr)
        return None


class IntrinsicsAnalyzer:
    """Marigold intrinsic image decomposition: albedo, roughness, metallicity.
    Loaded lazily on first use (the pipeline is ~5 GB of weights)."""

    def __init__(self, device: torch.device, processing_res: int = 640):
        self.device = device
        self.processing_res = processing_res  # below default 768 to save memory
        self.pipe = None

    def _load(self):
        import diffusers
        print("Loading Marigold pipeline (first run downloads ~5 GB) ...")
        name = "prs-eth/marigold-iid-appearance-v1-1"
        try:
            self.pipe = diffusers.MarigoldIntrinsicsPipeline.from_pretrained(
                name, variant="fp16", torch_dtype=torch.float16)
        except Exception:
            self.pipe = diffusers.MarigoldIntrinsicsPipeline.from_pretrained(
                name, torch_dtype=torch.float16)
        self.pipe = self.pipe.to(self.device)
        # Memory savers — important on 8 GB machines.
        self.pipe.enable_attention_slicing()
        self.pipe.vae.enable_tiling()

    def analyze(self, frame_rgb: np.ndarray):
        """Return (albedo_rgb, roughness, metallicity) at processing size.
        albedo is uint8 RGB; roughness/metallicity are float arrays in 0-1."""
        if self.pipe is None:
            self._load()
        image = Image.fromarray(frame_rgb)
        image.thumbnail((self.processing_res, self.processing_res))
        generator = torch.Generator(self.device).manual_seed(2024)
        result = self.pipe(image, num_inference_steps=1, generator=generator)

        pred = np.asarray(result.prediction)
        props = self.pipe.target_properties
        mat_idx = props["target_names"].index("material")
        sub_names = props["material"]["sub_target_names"]
        rough = np.squeeze(pred[mat_idx][..., sub_names.index("roughness")]).astype(float)
        metal = np.squeeze(pred[mat_idx][..., sub_names.index("metallicity")]).astype(float)

        # Albedo via the built-in visualizer (handles sRGB conversion).
        vis = self.pipe.image_processor.visualize_intrinsics(
            result.prediction, props)
        albedo = np.asarray(vis[0]["albedo"].convert("RGB"))
        if self.device.type == "mps":
            torch.mps.empty_cache()
        return albedo, rough, metal


def segment(frame_bgr: np.ndarray, session) -> np.ndarray:
    """Return a soft alpha matte (uint8, 0-255) at full frame resolution."""
    rgba = remove(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
                  session=session)  # RGBA ndarray
    return rgba[:, :, 3]


def extract_regions(alpha: np.ndarray):
    """Split the matte into per-object regions: (x, y, w, h, region_mask)."""
    binary = (alpha >= ALPHA_THRESH).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    frame_area = alpha.shape[0] * alpha.shape[1]
    regions = []
    for i in range(1, num):  # label 0 is background
        x, y, w, h, area = stats[i]
        if area < frame_area * MIN_REGION_FRAC:
            continue
        regions.append((x, y, w, h, (labels == i).astype(np.uint8) * 255))
    return regions


def checkerboard(h: int, w: int, cell: int = 16) -> np.ndarray:
    """Photoshop-style transparency checkerboard."""
    ys, xs = np.mgrid[0:h, 0:w]
    board = (((ys // cell) + (xs // cell)) % 2 * 55 + 180).astype(np.uint8)
    return cv2.merge([board, board, board])


def compose_result(frame_bgr, alpha, labeled_regions):
    """Return (display_bgr, cutout_bgra): the segmented portion over a
    checkerboard for display, and a transparent-background PNG image."""
    h, w = alpha.shape
    a = (alpha.astype(np.float32) / 255.0)[..., None]
    display = (frame_bgr * a + checkerboard(h, w) * (1 - a)).astype(np.uint8)
    cutout = cv2.merge([frame_bgr[:, :, 0], frame_bgr[:, :, 1],
                        frame_bgr[:, :, 2], alpha])

    for (x, y, tw_, th_, region_mask, material) in labeled_regions:
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(display, contours, -1, (40, 160, 30), 2)
        scale = 0.8 if w > display.shape[1] // 3 else 0.6
        (tw, th), _ = cv2.getTextSize(material, cv2.FONT_HERSHEY_SIMPLEX,
                                      scale, 2)
        tx = min(x, display.shape[1] - tw - 10)  # keep the box inside frame
        ty = max(y - 10, th + 6)
        cv2.rectangle(display, (tx, ty - th - 6), (tx + tw + 10, ty + 6),
                      (0, 0, 0), -1)
        cv2.putText(display, material, (tx + 5, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (80, 255, 120), 2)
    return display, cutout


def panel_tile(img_bgr: np.ndarray, width: int, caption: str) -> np.ndarray:
    """Resize a map to a tile width and draw a caption strip on it."""
    h, w = img_bgr.shape[:2]
    tile = cv2.resize(img_bgr, (width, int(h * width / w)))
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(tile, caption, (6, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1)
    return tile


def process_capture(frame_bgr, session, classifier, intrinsics,
                    save_prefix="capture"):
    """Segment a captured frame, label materials, run Marigold intrinsics,
    save + return the composed result panel."""
    print("Segmenting (rembg) ...")
    alpha = segment(frame_bgr, session)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_h, frame_w = alpha.shape
    frame_area = frame_h * frame_w
    rembg_binary = ((alpha >= ALPHA_THRESH).astype(np.uint8)) * 255

    print("Analyzing scene with Gemini (objects + masks + profiles) ...")
    detections = classifier.analyze_scene(frame_rgb)

    # Hybrid: Gemini decides WHAT the objects are, rembg's full-resolution
    # matte decides exactly WHERE their edges are.
    labeled = []
    objects_info = []
    display_alpha = np.zeros_like(alpha)
    for det in detections:
        gemini_mask = gemini_mask_to_frame(det, frame_w, frame_h)
        if gemini_mask is None:
            continue
        inter = cv2.bitwise_and(gemini_mask, rembg_binary)
        gemini_px = np.count_nonzero(gemini_mask)
        if (np.count_nonzero(inter) >= 0.2 * gemini_px
                and np.count_nonzero(inter) >= frame_area * MIN_REGION_FRAC):
            # Normal case: refine the coarse Gemini mask with rembg's edges,
            # keeping rembg's soft alpha inside the object.
            region_mask = inter
            soft = np.where(inter > 0, alpha, 0).astype(np.uint8)
        elif gemini_px >= frame_area * MIN_REGION_FRAC:
            # rembg missed this object entirely; use the Gemini mask as-is.
            region_mask = gemini_mask
            soft = gemini_mask
        else:
            continue  # too small to be a real object

        x, y, w, h = cv2.boundingRect(region_mask)
        info = {k: v for k, v in det.items() if k not in ("mask", "box_2d")}
        name = info.get("name") or "object"
        material = info.get("primary_material") or "unknown"
        info["bbox"] = [int(x), int(y), int(w), int(h)]
        print(f"  {name} at ({x},{y}) {w}x{h}:")
        print("    " + json.dumps(info, indent=2).replace("\n", "\n    "))
        objects_info.append(info)
        labeled.append([x, y, w, h, region_mask, f"{name}: {material}"])
        display_alpha = np.maximum(display_alpha, soft)

    if not labeled:
        # Gemini failed or found nothing: fall back to rembg-only regions.
        print("  no Gemini detections usable; falling back to rembg regions")
        display_alpha = alpha
        for (x, y, w, h, region_mask) in extract_regions(alpha):
            objects_info.append(
                {"name": "object", "primary_material": "unknown",
                 "bbox": [int(x), int(y), int(w), int(h)]})
            labeled.append([x, y, w, h, region_mask, "object"])
    alpha = display_alpha  # downstream masking/compositing uses the hybrid

    maps_row = None
    if intrinsics is not None:
        print("Running Marigold intrinsics ...")
        albedo, rough, metal = intrinsics.analyze(frame_rgb)
        mh, mw = rough.shape
        alpha_small = cv2.resize(alpha, (mw, mh)).astype(np.float32) / 255.0

        # Per-object roughness/metallicity over each segmentation mask.
        for entry, info in zip(labeled, objects_info):
            region_small = cv2.resize(entry[4], (mw, mh)) > 127
            if region_small.any():
                r = float(rough[region_small].mean())
                m = float(metal[region_small].mean())
                print(f"  {entry[5]}: roughness {r:.3f}, metallicity {m:.3f}")
                entry[5] += f" | rough {r:.2f} metal {m:.2f}"
                info["marigold_roughness"] = round(r, 3)
                info["marigold_metallicity"] = round(m, 3)

        # Mask the maps to the segmented portion for display/saving.
        albedo_bgr = cv2.cvtColor(albedo, cv2.COLOR_RGB2BGR)
        albedo_bgr = (albedo_bgr * alpha_small[..., None]).astype(np.uint8)
        rough_img = cv2.cvtColor(
            (np.clip(rough, 0, 1) * 255 * alpha_small).astype(np.uint8),
            cv2.COLOR_GRAY2BGR)
        metal_img = cv2.cvtColor(
            (np.clip(metal, 0, 1) * 255 * alpha_small).astype(np.uint8),
            cv2.COLOR_GRAY2BGR)
        cv2.imwrite(f"{save_prefix}_albedo.png", albedo_bgr)
        cv2.imwrite(f"{save_prefix}_roughness.png", rough_img)
        cv2.imwrite(f"{save_prefix}_metallicity.png", metal_img)

        tile_w = frame_bgr.shape[1] // 3
        maps_row = np.hstack([
            panel_tile(albedo_bgr, tile_w, "albedo"),
            panel_tile(rough_img, tile_w, "roughness (bright = rough)"),
            panel_tile(metal_img, tile_w, "metallicity (bright = metal)"),
        ])

    display, cutout = compose_result(frame_bgr, alpha,
                                     [tuple(e) for e in labeled])
    if maps_row is not None:
        pad = display.shape[1] - maps_row.shape[1]
        if pad > 0:
            maps_row = cv2.copyMakeBorder(maps_row, 0, 0, 0, pad,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
        display = np.vstack([display, maps_row])

    cv2.imwrite(f"{save_prefix}_segmented.png", display)
    cv2.imwrite(f"{save_prefix}_cutout.png", cutout)
    with open(f"{save_prefix}_objects.json", "w") as f:
        json.dump(objects_info, f, indent=2)
    print(f"Saved {save_prefix}_segmented.png, {save_prefix}_cutout.png, "
          f"{save_prefix}_objects.json"
          + ("" if maps_row is None else
             f", {save_prefix}_albedo.png, {save_prefix}_roughness.png, "
             f"{save_prefix}_metallicity.png"))
    return display


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--image", type=str, default=None,
                        help="segment an existing photo instead of the camera")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-intrinsics", action="store_true",
                        help="skip the Marigold albedo/roughness/metallicity "
                             "pass (faster, much less memory)")
    args = parser.parse_args()

    load_env_file(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               ".env"))
    device = pick_device()
    print(f"Loading models ({device}) ...")
    session = new_session("isnet-general-use")
    classifier = GeminiClassifier()
    print(f"Using Gemini model: {classifier.model}")
    intrinsics = None if args.no_intrinsics else IntrinsicsAnalyzer(device)
    print("Models ready.")

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read image: {args.image}", file=sys.stderr)
            return 1
        process_capture(frame, session, classifier, intrinsics)
        return 0

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Warm up: on macOS the first reads fail while the camera initializes
    # (or while the permission prompt is up), so retry before giving up.
    ok = cap.isOpened()
    if ok:
        for _ in range(100):
            ok, _ = cap.read()
            if ok:
                break
            time.sleep(0.1)
    if not ok:
        print(
            f"Could not get frames from camera {args.camera}.\n"
            "On macOS this almost always means the app you ran this from has\n"
            "no camera permission. Fix:\n"
            "  1. System Settings > Privacy & Security > Camera\n"
            "  2. Enable it for your terminal app (Terminal, iTerm, VS Code, ...)\n"
            "     If it isn't listed, run this script once from that app so\n"
            "     macOS shows the permission prompt, and click Allow.\n"
            "  3. Quit and reopen the terminal app, then run again.\n"
            "If permission is already granted, another app may be using the\n"
            "camera, or try a different index: --camera 1",
            file=sys.stderr)
        cap.release()
        return 1

    shot = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed.", file=sys.stderr)
                break
            preview = frame.copy()
            cv2.putText(preview, "SPACE: take picture   q: quit",
                        (10, preview.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.imshow("Camera", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                shot += 1
                result = process_capture(frame, session, classifier, intrinsics,
                                         save_prefix=f"capture_{shot:02d}")
                cv2.imshow("Segmented", result)
                cv2.waitKey(0)  # any key returns to the live preview
                cv2.destroyWindow("Segmented")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
