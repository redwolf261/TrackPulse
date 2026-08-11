"""Video frame extraction — samples frames from an uploaded clip at a fixed
interval so a single video upload can populate a session's trend the same way
a sequence of photo uploads would.

Deliberately simple: OpenCV VideoCapture + interval-based sampling. No scene
detection, no motion analysis — the brief asks for "video frames" as an input
mode, not a video-understanding feature.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# Hard caps so a long/garbage video can't hang the request or flood a session
# with hundreds of near-identical observations.
MAX_FRAMES_EXTRACTED = 12
SAMPLE_INTERVAL_SECONDS = 2.0
MIN_SAMPLE_INTERVAL_SECONDS = 0.5  # floor, for short videos


class VideoDecodeError(Exception):
    pass


def extract_frames(video_bytes: bytes, suffix: str = ".mp4") -> list[bytes]:
    """Extract up to MAX_FRAMES_EXTRACTED JPEG-encoded frames, sampled at a
    fixed time interval across the video's duration. Returns a list of JPEG
    bytes, oldest first (matching upload/playback order)."""
    # OpenCV's VideoCapture needs a real file path on most backends (no in-memory
    # buffer support portable across platforms), so round-trip through a temp file.
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise VideoDecodeError("Could not open video file — unsupported format or corrupt file")

        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration_s = (frame_count / fps) if fps > 0 else 0

        if duration_s <= 0:
            # Metadata unavailable (some containers don't report it reliably) —
            # fall back to reading sequential frames at a fixed frame-count stride.
            frames = _extract_by_frame_stride(cap)
        else:
            interval = max(MIN_SAMPLE_INTERVAL_SECONDS, min(SAMPLE_INTERVAL_SECONDS, duration_s / MAX_FRAMES_EXTRACTED))
            frames = _extract_by_time_interval(cap, fps, duration_s, interval)

        cap.release()

        if not frames:
            raise VideoDecodeError("No readable frames found in video")

        return frames[:MAX_FRAMES_EXTRACTED]
    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            pass  # best-effort cleanup; Windows can hold a brief file lock


def _extract_by_time_interval(cap: cv2.VideoCapture, fps: float, duration_s: float, interval: float) -> list[bytes]:
    frames: list[bytes] = []
    t = 0.0
    while t < duration_s and len(frames) < MAX_FRAMES_EXTRACTED:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if ok:
            frames.append(_encode_jpeg(frame))
        t += interval
    return frames


def _extract_by_frame_stride(cap: cv2.VideoCapture) -> list[bytes]:
    """Fallback when duration metadata is missing: read sequentially, keep every
    Nth frame based on a rough total-frame estimate (or just the first N reads)."""
    frames: list[bytes] = []
    frame_idx = 0
    stride = 1
    while len(frames) < MAX_FRAMES_EXTRACTED:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            frames.append(_encode_jpeg(frame))
        frame_idx += 1
        if frame_idx > 5000:  # safety valve against pathological/corrupt streams
            break
    return frames


def _encode_jpeg(frame: np.ndarray) -> bytes:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return buf.getvalue()
