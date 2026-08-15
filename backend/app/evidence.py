"""Evidence Trail — explains WHY a prediction should (or shouldn't) be trusted,
using only data we already compute (class probabilities + trend history). No
new model, no LLM: a deterministic, auditable set of checks over numbers the
system already has, surfaced to the user instead of hidden behind a bare label.

This exists because our own measured results (see README) show the model is
sometimes confidently wrong — mean confidence on incorrect racing-domain
predictions was statistically indistinguishable from mean confidence on correct
ones in our exp02 evaluation. A raw confidence number alone doesn't let a user
tell those cases apart. These checks catch some of the situations where the
*label* looks confident but the *evidence* underneath it is thin or
contradictory, which is a materially different (and more honest) thing to show
than "WET, 91%" on its own.
"""
from __future__ import annotations

CONDITION_RANK = {"DRY": 0, "DAMP": 1, "WET": 2}

# How close the top two class probabilities have to be before we call the
# call "contested" rather than clear-cut.
CLOSE_CALL_MARGIN = 0.15

# Confidence below which we already suppress a strong suggestion (matches
# strategy.py's existing low-confidence gate) — reused here for consistency.
LOW_CONFIDENCE_THRESHOLD = 0.45

# Confidence band above the hard floor but still weak enough to flag — a 52%
# call isn't "low confidence" by the strategy.py gate, but it's not solid
# either; don't let it pass through as HIGH trust just because it cleared 45%.
MODERATE_CONFIDENCE_THRESHOLD = 0.60


def build_evidence_trail(
    label: str,
    probabilities: dict[str, float],
    trend: str,
    confidence: float,
    history_labels: list[str],
) -> dict:
    """Returns a small structured explanation: a trust level (HIGH / MODERATE /
    LOW) and a list of specific, human-readable reasons behind it. Every check
    here is a plain comparison over numbers already produced elsewhere in the
    pipeline — nothing here re-runs the model or invents new evidence."""
    reasons: list[str] = []
    concerns: list[str] = []

    sorted_probs = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_p = sorted_probs[0]
    second_label, second_p = sorted_probs[1]
    margin = top_p - second_p

    # Check 1: is the top call clear-cut, or nearly tied with the runner-up?
    if margin < CLOSE_CALL_MARGIN:
        concerns.append(
            f"Close call: {top_label} ({top_p:.0%}) vs {second_label} ({second_p:.0%}) "
            f"— only {margin:.0%} apart."
        )
    else:
        reasons.append(f"{top_label} is clearly ahead of {second_label} by {margin:.0%}.")

    # Check 2: does this prediction fit the recent trend, or contradict it?
    # history_labels is PRIOR readings only (current one not included) — we
    # compare the current reading directly against those prior ranks.
    if history_labels:
        recent_ranks = [CONDITION_RANK[l] for l in history_labels[-3:]]
        current_rank = CONDITION_RANK[label]
        if trend == "WETTING" and current_rank < max(recent_ranks):
            concerns.append("This reading is drier than the recent trend would suggest.")
        elif trend == "DRYING" and current_rank > min(recent_ranks):
            concerns.append("This reading is wetter than the recent trend would suggest.")
        else:
            reasons.append("Consistent with the recent trend, not an abrupt jump.")
    else:
        reasons.append("First reading this session — no trend history to compare against yet.")

    # Check 3: raw confidence floor.
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        concerns.append(f"Overall confidence is low ({confidence:.0%}).")
    elif confidence < MODERATE_CONFIDENCE_THRESHOLD:
        concerns.append(f"Confidence is only moderate ({confidence:.0%}).")
    elif confidence > 0.85 and margin < CLOSE_CALL_MARGIN:
        # The specific failure mode we measured: high confidence AND a close call
        # underneath it — the number alone would look trustworthy but isn't.
        concerns.append(
            "High confidence but the top two classes are close — treat this as "
            "less certain than the raw percentage implies."
        )

    if not concerns:
        trust = "HIGH"
    elif len(concerns) == 1 and confidence >= LOW_CONFIDENCE_THRESHOLD:
        trust = "MODERATE"
    else:
        trust = "LOW"

    return {"trust": trust, "reasons": reasons, "concerns": concerns}
