"""Deterministic trend + suggestion logic. No LLM/agent in this path by design —
see project notes: the model perceives, this module decides, both stay auditable."""
from __future__ import annotations

CONDITION_RANK = {"DRY": 0, "DAMP": 1, "WET": 2}


def compute_trend(history_labels: list[str]) -> str:
    """history_labels: this session's labels oldest-first, INCLUDING the current/newest
    observation as the last element — the trend describes the trajectory ending at
    that observation, not the trajectory before it.
    Returns WETTING | DRYING | STABLE based on short-window direction.

    Uses the sum of consecutive signed rank differences across the window rather
    than only the endpoint delta (first vs last), so transient mid-window spikes
    (e.g. DRY→WET→WET→DRY) don't produce a spurious STABLE result when the
    track was clearly cycling through a wet patch.
    """
    if len(history_labels) < 2:
        return "STABLE"

    window = history_labels[-4:]
    ranks = [CONDITION_RANK[label] for label in window]
    # Sum of all consecutive signed differences — positive means net wetting,
    # negative means net drying.  For monotone windows this equals ranks[-1]-ranks[0],
    # preserving existing behaviour.
    signed_sum = sum(ranks[i + 1] - ranks[i] for i in range(len(ranks) - 1))
    if signed_sum > 0:
        return "WETTING"
    if signed_sum < 0:
        return "DRYING"
    return "STABLE"


def suggestion_for(label: str, trend: str, confidence: float) -> str:
    if confidence < 0.45:
        return "Low confidence reading — awaiting clearer image before advising."

    if label == "WET":
        if trend == "DRYING":
            return "Track wet but drying: hold wets, watch for crossover window soon."
        return "Track wet: wet-tyre conditions likely, consider change if not already fitted."

    if label == "DAMP":
        if trend == "WETTING":
            return "Track dampening: intermediate tyre window approaching."
        if trend == "DRYING":
            return "Track drying: slick tyre window approaching."
        return "Track damp and stable: monitor closely, no immediate change needed."

    # DRY
    if trend == "WETTING":
        return "Conditions starting to change: monitor closely for rain."
    return "Track dry: no tyre change indicated."


def compute_grip_index(probabilities: dict[str, float]) -> int:
    """Calculates track grip percentage (0-100) from class probabilities.
    Weighted blend: DRY=100% grip, DAMP=65% grip, WET=32% grip."""
    raw = (
        probabilities.get("DRY", 0.0) * 1.00
        + probabilities.get("DAMP", 0.0) * 0.65
        + probabilities.get("WET", 0.0) * 0.32
    )
    return int(round(max(20.0, min(100.0, raw * 100.0))))


def compute_tyre_crossover(probabilities: dict[str, float], trend: str, confidence: float) -> dict:
    """Deterministic tyre strategy and crossover modeling from condition probabilities.
    Returns compound recommendation, grip index, lap delta, and crossover status."""
    p_dry = probabilities.get("DRY", 0.0)
    p_damp = probabilities.get("DAMP", 0.0)
    p_wet = probabilities.get("WET", 0.0)
    grip = compute_grip_index(probabilities)

    # Lap time penalty (seconds) vs a dry baseline for each compound
    slick_delta = round(p_damp * 4.5 + p_wet * 14.2, 1)
    inter_delta = round(p_dry * 2.8 + p_wet * 3.8, 1)
    wet_delta = round(p_dry * 8.5 + p_damp * 3.2, 1)

    # Compound decision tree based on grip index + trend
    if grip >= 80:
        compound = "SLICK (C3/C4)"
        status = "OPTIMAL"
        message = "Track in prime slick window — maximum mechanical grip available."
    elif grip >= 62:
        if trend == "DRYING":
            compound = "SLICK (C3)"
            status = "CROSSOVER_ACTIVE"
            message = "Dry line forming: slicks now faster than intermediates — pit window open."
        elif trend == "WETTING":
            compound = "INTERMEDIATE"
            status = "CROSSOVER_APPROACHING"
            message = "Moisture increasing: prepare to box for intermediates."
        else:
            compound = "INTERMEDIATE"
            status = "OPTIMAL"
            message = "Damp stable surface: intermediate compound optimal."
    elif grip >= 45:
        if trend == "DRYING":
            compound = "INTERMEDIATE"
            status = "CROSSOVER_ACTIVE"
            message = "Standing water clearing: intermediate crossover window active."
        else:
            compound = "INTERMEDIATE"
            status = "OPTIMAL"
            message = "Intermediate tyre window active across the circuit."
    else:
        if trend == "DRYING":
            compound = "FULL WET"
            status = "CROSSOVER_APPROACHING"
            message = "Heavy wet with drying trend: hold wets, monitor standing water."
        else:
            compound = "FULL WET"
            status = "OPTIMAL"
            message = "Standing water / aquaplaning risk: full wet compound required."

    # Best available lap delta = performance of optimal compound
    lap_delta = min(slick_delta, inter_delta, wet_delta)

    return {
        "grip_index": grip,
        "optimal_compound": compound,
        "crossover_status": status,
        "crossover_message": message,
        "lap_delta_seconds": lap_delta,
        "compound_deltas": {
            "SLICK": slick_delta,
            "INTERMEDIATE": inter_delta,
            "FULL_WET": wet_delta,
        },
    }

