"""Snapshot material segmenter: take a picture, cut objects out of the
background at high quality, and label each object's material.

Workflow:
  1. A live preview window shows the camera. Press SPACE to take a picture.
  2. The captured frame is segmented with rembg (ISNet matting model) at full
     resolution — this produces clean, soft-edged cutouts.
  3. Each object region is classified for material, open-ended, with BLIP-VQA
     ("what material is this object made of?") — no fixed material list.
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
import sys
import time

import cv2
import numpy as np
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


class MaterialClassifier:
    """Open-ended material classification with BLIP-VQA, in two steps:
    identify the object first, then ask what that object is made of.
    (Asking about "this object" directly tends to get answers about the
    surroundings; masked backgrounds confuse the model even more.)"""

    def __init__(self, device: torch.device):
        from transformers import BlipProcessor, BlipForQuestionAnswering
        name = "Salesforce/blip-vqa-base"
        self.processor = BlipProcessor.from_pretrained(name)
        # Kept on CPU: BLIP is small enough there, and it leaves the whole
        # GPU memory budget to Marigold (matters on 8 GB machines).
        self.device = torch.device("cpu")
        self.model = (BlipForQuestionAnswering.from_pretrained(name)
                      .to(self.device).eval())

    @torch.inference_mode()
    def _ask(self, image: Image.Image, question: str) -> str:
        inputs = self.processor(image, question,
                                return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=10)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()

    def classify(self, crop_rgb: np.ndarray) -> tuple[str, str]:
        """Return (object_name, material) for an RGB crop."""
        image = Image.fromarray(crop_rgb)
        obj = self._ask(image, "what is the main object in this picture?")
        material = self._ask(image, f"what material is the {obj} made of?")
        return obj, material


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
    print("Segmenting ...")
    alpha = segment(frame_bgr, session)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    labeled = []
    for (x, y, w, h, region_mask) in extract_regions(alpha):
        obj, material = classifier.classify(frame_rgb[y:y + h, x:x + w])
        print(f"  {obj} at ({x},{y}) {w}x{h}: {material}")
        labeled.append([x, y, w, h, region_mask, f"{obj}: {material}"])
    if not labeled:
        print("  no objects found")

    maps_row = None
    if intrinsics is not None:
        print("Running Marigold intrinsics ...")
        albedo, rough, metal = intrinsics.analyze(frame_rgb)
        mh, mw = rough.shape
        alpha_small = cv2.resize(alpha, (mw, mh)).astype(np.float32) / 255.0

        # Per-object roughness/metallicity over each segmentation mask.
        for entry in labeled:
            region_small = cv2.resize(entry[4], (mw, mh)) > 127
            if region_small.any():
                r = float(rough[region_small].mean())
                m = float(metal[region_small].mean())
                print(f"  {entry[5]}: roughness {r:.3f}, metallicity {m:.3f}")
                entry[5] += f" | rough {r:.2f} metal {m:.2f}"

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
    print(f"Saved {save_prefix}_segmented.png, {save_prefix}_cutout.png"
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

    device = pick_device()
    print(f"Loading models ({device}) ...")
    session = new_session("isnet-general-use")
    classifier = MaterialClassifier(device)
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
