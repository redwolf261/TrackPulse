"""Model loading and inference for track-condition classification.

Uses ONNX Runtime against the model trained by scripts/train.py. Falls back to a
heuristic (brightness/saturation-based) classifier if no trained model is present
yet, so the API and frontend can be developed/demoed before training finishes.
"""
from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

_MODEL_FILENAME = "trackpulse_classifier.onnx"


def _resolve_model_path() -> Path:
    """Find the ONNX model without assuming a specific deployment layout.

    Checked in order:
      1. TRACKPULSE_MODEL_PATH env var, if set (explicit override — use this
         if the model is deployed somewhere other than the layouts below).
      2. <repo_root>/models/<file> — the layout in this repo (backend/ and
         models/ are siblings).
      3. <backend_dir>/models/<file> — in case a deploy step copies the model
         inside the backend service's own directory instead.
    """
    env_override = os.environ.get("TRACKPULSE_MODEL_PATH")
    if env_override:
        return Path(env_override)

    backend_app_dir = Path(__file__).resolve().parent  # .../backend/app
    candidates = [
        backend_app_dir.parent.parent / "models" / _MODEL_FILENAME,  # repo_root/models
        backend_app_dir.parent / "models" / _MODEL_FILENAME,  # backend/models
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]  # default to the expected repo-root layout; _try_load() handles absence


MODEL_PATH = _resolve_model_path()
CLASSES = ["DRY", "DAMP", "WET"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Classifier:
    def __init__(self) -> None:
        self._session = None
        self._loaded_real_model = False
        self._try_load()

    def _try_load(self) -> None:
        if not MODEL_PATH.exists():
            return
        try:
            import onnxruntime as ort

            self._session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
            self._loaded_real_model = True
        except Exception as exc:  # noqa: BLE001 - degrade gracefully to heuristic fallback
            print(f"[inference] Failed to load ONNX model, using fallback heuristic: {exc}")
            self._session = None

    @property
    def using_trained_model(self) -> bool:
        return self._loaded_real_model

    def predict(self, image_bytes: bytes) -> dict:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()  # force decode now so truncated/corrupt files fail here, not mid-inference
            image = image.convert("RGB")
        except Exception as exc:
            raise UnidentifiedImageError(f"Could not decode image: {exc}") from exc
        if self._session is not None:
            return self._predict_onnx(image)
        return self._predict_heuristic(image)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        # BILINEAR to match torchvision.transforms.Resize's default, which is what
        # training/eval used (PIL's own .resize() default is BICUBIC and produced
        # measurably different predictions on borderline images — see project notes).
        image = image.resize((224, 224), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return arr[np.newaxis, ...].astype(np.float32)

    def _predict_onnx(self, image: Image.Image) -> dict:
        input_tensor = self._preprocess(image)
        input_name = self._session.get_inputs()[0].name
        logits = self._session.run(None, {input_name: input_tensor})[0][0]
        probs = _softmax(logits)
        return _to_result(probs)

    def _predict_heuristic(self, image: Image.Image) -> dict:
        """Crude placeholder: darker + more saturated/reflective -> wetter.
        Used only until the trained ONNX model is available."""
        small = image.resize((64, 64))
        hsv = np.asarray(small.convert("HSV"), dtype=np.float32) / 255.0
        v = hsv[..., 2].mean()  # brightness
        s = hsv[..., 1].mean()  # saturation
        wetness_score = np.clip((1 - v) * 0.6 + s * 0.4, 0, 1)
        p_wet = float(wetness_score)
        p_dry = float(1 - wetness_score)
        p_damp = float(1 - abs(p_dry - p_wet))
        probs = np.array([p_dry, p_damp, p_wet], dtype=np.float32)
        probs = probs / probs.sum()
        return _to_result(probs)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _to_result(probs: np.ndarray) -> dict:
    idx = int(np.argmax(probs))
    return {
        "label": CLASSES[idx],
        "p_dry": float(probs[0]),
        "p_damp": float(probs[1]),
        "p_wet": float(probs[2]),
        "confidence": float(probs[idx]),
    }


classifier = Classifier()
