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

