"""Single Gemini access point: structured generation + multimodal embeddings.

One client, used by perception (descriptor), retrieval (embeddings), and
prediction (per-gripper force). Every call is content-hash cached on disk and
wrapped with retries. This is the only module that imports google-genai, and it
is imported lazily so mock / dry-run runs need neither the package nor a key.

Offline behaviour lives in the callers: perception/prediction build deterministic
stubs when cfg.models.dry_run is set, and retrieval uses a mock embedder, so the
full pipeline runs with no network. This module is exercised only for live calls.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import REPO_ROOT, Config


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from a .env into the environment (existing env wins).

    No dependency; mirrors the material-segmenter pattern. Looks at the repo root
    by default so `GEMINI_API_KEY=...` in .env is picked up for live calls.
    """
    path = path or (REPO_ROOT / ".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class _DiskCache:
    """Tiny content-addressed JSON cache under paths.cache."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(*parts: Any) -> str:
        blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def get(self, key: str) -> Any | None:
        path = self.root / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def put(self, key: str, value: Any) -> None:
        (self.root / f"{key}.json").write_text(json.dumps(value))


def _encode_image(image_bgr: np.ndarray | None) -> str | None:
    if image_bgr is None:
        return None
    import cv2

    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise RuntimeError("failed to encode image")
    return base64.b64encode(buf.tobytes()).decode("ascii")


class GeminiClient:
    """Thin wrapper over google-genai with caching + retries."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.cache = _DiskCache(cfg.path("cache")) if cfg.models.cache else None
        self._client = None  # lazy

    def _sdk(self):  # noqa: ANN202 - external type
        if self._client is None:
            from google import genai

            load_dotenv()  # pick up GEMINI_API_KEY from a repo-root .env if present
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) for live Gemini calls.")
            self._client = genai.Client(api_key=api_key)
        return self._client

    # --- structured JSON generation ---------------------------------------- #
    def generate_json(
        self,
        *,
        system: str,
        instruction: str,
        schema: type[BaseModel],
        image_bgr: np.ndarray | None = None,
        extra: dict | None = None,
    ) -> dict:
        img_b64 = _encode_image(image_bgr)
        cache_key = None
        if self.cache is not None:
            cache_key = self.cache.key(
                "gen", self.cfg.models.vlm, system, instruction,
                schema.model_json_schema(), img_b64, extra,
            )
            hit = self.cache.get(cache_key)
            if hit is not None:
                return hit
        result = self._generate_json_live(system, instruction, schema, img_b64, extra)
        if self.cache is not None and cache_key is not None:
            self.cache.put(cache_key, result)
        return result

    @retry(stop=stop_after_attempt(8), wait=wait_exponential(min=2, max=60))
    def _generate_json_live(
        self, system: str, instruction: str, schema: type[BaseModel],
        img_b64: str | None, extra: dict | None,
    ) -> dict:
        from google.genai import types

        parts: list[Any] = [instruction]
        if extra:
            parts.append(json.dumps(extra))
        if img_b64 is not None:
            parts.append(types.Part.from_bytes(data=base64.b64decode(img_b64), mime_type="image/png"))
        resp = self._sdk().models.generate_content(
            model=self.cfg.models.vlm,
            contents=parts,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=self.cfg.models.temperature,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return json.loads(resp.text)

    # --- multimodal embedding ---------------------------------------------- #
    def embed(self, *, text: str, image_bgr: np.ndarray | None = None) -> np.ndarray:
        img_b64 = _encode_image(image_bgr)
        cache_key = None
        if self.cache is not None:
            cache_key = self.cache.key(
                "embed", self.cfg.retrieval.embedding.model,
                self.cfg.retrieval.embedding.dim, text, img_b64,
            )
            hit = self.cache.get(cache_key)
            if hit is not None:
                return np.asarray(hit, dtype=np.float32)
        vec = self._embed_live(text, img_b64)
        if self.cache is not None and cache_key is not None:
            self.cache.put(cache_key, vec.tolist())
        return vec

    @retry(stop=stop_after_attempt(8), wait=wait_exponential(min=2, max=60))
    def _embed_live(self, text: str, img_b64: str | None) -> np.ndarray:
        from google.genai import types

        contents: list[Any] = [text]
        if img_b64 is not None:
            contents.append(types.Part.from_bytes(data=base64.b64decode(img_b64), mime_type="image/png"))
        resp = self._sdk().models.embed_content(
            model=self.cfg.retrieval.embedding.model,
            contents=contents,
            config=types.EmbedContentConfig(output_dimensionality=self.cfg.retrieval.embedding.dim),
        )
        return np.asarray(resp.embeddings[0].values, dtype=np.float32)


_CLIENT_CACHE: dict[int, GeminiClient] = {}


def get_client(cfg: Config) -> GeminiClient:
    """Process-wide singleton keyed by config identity."""
    key = id(cfg)
    if key not in _CLIENT_CACHE:
        _CLIENT_CACHE[key] = GeminiClient(cfg)
    return _CLIENT_CACHE[key]
