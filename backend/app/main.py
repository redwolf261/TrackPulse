"""TrackPulse backend — FastAPI service.

Endpoints:
  POST /predict            classify an uploaded image, persist it, return label+trend+suggestion
  POST /predict/video      sample frames from an uploaded video, classify each as if
                            uploaded in sequence, persist all, return the full batch
  GET  /history/{session}  full observation history for a session (for the trend chart)
  GET  /health             liveness + model-status check
"""
from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, File, Form, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from PIL import UnidentifiedImageError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from sqlalchemy import select

from .db import init_db, get_session, Observation
from .evidence import build_evidence_trail
from .inference import classifier
from .strategy import compute_trend, suggestion_for
from .video import extract_frames, VideoDecodeError

logger = logging.getLogger("trackpulse")

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_VIDEO_CONTENT_TYPES = {"video/mp4", "video/quicktime", "video/webm", "video/x-msvideo"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_VIDEO_UPLOAD_BYTES = 100 * 1024 * 1024

# Per-client-IP limits on the expensive endpoints (inference + video decode).
# Generous enough not to interfere with normal demo/judging use (a person
# clicking through samples, uploading a few photos) while bounding the worst
# case of one client hammering the free-tier deployment.
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="TrackPulse API", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hackathon scope: tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": classifier.using_trained_model,
        "model_source": "trained-onnx" if classifier.using_trained_model else "fallback-heuristic",
    }


def _classify_and_persist(
    image_bytes: bytes,
    session_id: str,
    filename: str,
    db: Session,
) -> dict:
    """Shared core of /predict and /predict/video: classify one frame, compute
    trend against this session's history so far, persist, return the API shape.
    Raises HTTPException on failure — callers decide whether that aborts the
    whole request (single image) or is recorded per-frame (video batch)."""
    try:
        result = classifier.predict(image_bytes)
    except UnidentifiedImageError:
        raise HTTPException(400, "File is not a valid/readable image")
    except Exception:
        logger.exception("Inference failed for session=%s filename=%s", session_id, filename)
        raise HTTPException(500, "Inference failed — please try a different image")

    try:
        past = db.execute(
            select(Observation.label)
            .where(Observation.session_id == session_id)
            .order_by(Observation.created_at.asc())
        ).scalars().all()

        # compute_trend expects the full window INCLUDING the current observation
        # (it compares the window's first vs last label) — `past` only has prior
        # rows since this one isn't inserted yet, so append it explicitly.
        past_labels = list(past)
        trend = compute_trend(past_labels + [result["label"]])
        suggestion = suggestion_for(result["label"], trend, result["confidence"])
        probabilities = {"DRY": result["p_dry"], "DAMP": result["p_damp"], "WET": result["p_wet"]}
        evidence = build_evidence_trail(
            result["label"], probabilities, trend, result["confidence"], past_labels
        )

        obs = Observation(
            session_id=session_id,
            label=result["label"],
            p_dry=result["p_dry"],
            p_damp=result["p_damp"],
            p_wet=result["p_wet"],
            confidence=result["confidence"],
            trend=trend,
            suggestion=suggestion,
            image_filename=filename[:255],
        )
        db.add(obs)
        db.commit()
        db.refresh(obs)
    except SQLAlchemyError:
        db.rollback()
        logger.exception("DB write failed for session=%s", session_id)
        raise HTTPException(500, "Failed to save observation — please retry")

    return {
        "session_id": session_id,
        "observation_id": obs.id,
        "label": result["label"],
        "probabilities": probabilities,
        "confidence": result["confidence"],
        "trend": trend,
        "suggestion": suggestion,
        "evidence": evidence,
        "created_at": obs.created_at.isoformat(),
        "model_source": "trained-onnx" if classifier.using_trained_model else "fallback-heuristic",
    }


@app.post("/predict")
@limiter.limit("30/minute")
async def predict(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
    db: Session = Depends(get_session),
):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(400, f"Unsupported content type: {file.content_type}")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "Uploaded file is empty")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(400, "Image too large (max 10MB)")

    session_id = session_id or str(uuid.uuid4())
    return _classify_and_persist(image_bytes, session_id, file.filename or "", db)


@app.post("/predict/video")
@limiter.limit("10/minute")
async def predict_video(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
    db: Session = Depends(get_session),
):
    if file.content_type not in ALLOWED_VIDEO_CONTENT_TYPES:
        raise HTTPException(
            400,
            f"Unsupported video type: {file.content_type} "
            f"(expected one of {', '.join(sorted(ALLOWED_VIDEO_CONTENT_TYPES))})",
        )

    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(400, "Uploaded file is empty")
    if len(video_bytes) > MAX_VIDEO_UPLOAD_BYTES:
        raise HTTPException(400, "Video too large (max 100MB)")

    suffix = {
        "video/mp4": ".mp4",
        "video/quicktime": ".mov",
        "video/webm": ".webm",
        "video/x-msvideo": ".avi",
    }.get(file.content_type, ".mp4")

    try:
        frames = extract_frames(video_bytes, suffix=suffix)
    except VideoDecodeError as exc:
        raise HTTPException(400, str(exc))
    except Exception:
        logger.exception("Video frame extraction failed for filename=%s", file.filename)
        raise HTTPException(500, "Could not process video — please try a different file")

    session_id = session_id or str(uuid.uuid4())
    base_name = (file.filename or "video").rsplit(".", 1)[0]

    results = []
    for i, frame_bytes in enumerate(frames):
        try:
            results.append(
                _classify_and_persist(frame_bytes, session_id, f"{base_name}_frame{i}.jpg", db)
            )
        except HTTPException as exc:
            # One bad frame shouldn't abort an otherwise-good video — record and continue.
            logger.warning("Skipping frame %d of video %s: %s", i, file.filename, exc.detail)
            continue

    if not results:
        raise HTTPException(500, "No frames could be classified from this video")

    return {
        "session_id": session_id,
        "frames_extracted": len(frames),
        "frames_classified": len(results),
        "observations": results,
    }


@app.get("/history/{session_id}")
def history(session_id: str, db: Session = Depends(get_session)):
    try:
        rows = db.execute(
            select(Observation)
            .where(Observation.session_id == session_id)
            .order_by(Observation.created_at.asc())
        ).scalars().all()
    except SQLAlchemyError:
        logger.exception("DB read failed for session=%s", session_id)
        raise HTTPException(500, "Failed to load history — please retry")

    return {
        "session_id": session_id,
        "count": len(rows),
        "observations": [
            {
                "id": r.id,
                "created_at": r.created_at.isoformat(),
                "label": r.label,
                "probabilities": {"DRY": r.p_dry, "DAMP": r.p_damp, "WET": r.p_wet},
                "confidence": r.confidence,
                "trend": r.trend,
                "suggestion": r.suggestion,
            }
            for r in rows
        ],
    }
