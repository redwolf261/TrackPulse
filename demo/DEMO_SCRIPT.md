# TrackPulse — Demo Script (~2 minutes)

Live app: **https://trackpulserace.netlify.app**
Backend: https://trackpulse-backend-7kod.onrender.com (free tier — cold-starts after
idling; hit `/health` ~1 min before presenting to warm it up)

## Before you start
- Open the live site in a fresh tab (not localhost — show the real deployed thing).
- Have `demo/test_sample_video.mp4` ready in case the video-upload moment needs a
  local file instead of a sample button.
- Know your two numbers cold, judges will probe these: **73% accuracy / 20% false-WET
  rate on racing imagery** (production model, exp02) and **43-point accuracy drop**
  we measured going from road-camera to racing-camera data before adapting to it.

## 0:00–0:15 — The problem, in one breath
> "Track condition changes faster than a weather report can keep up, but a
> race engineer needs to know *right now* if the surface is getting safer or
> riskier, so they can call a tyre change. We built TrackPulse: upload a photo
> or a video clip of the track, get a Dry/Damp/Wet call, a trend over time,
> and a plain suggestion — in under 2 milliseconds of model time."

## 0:15–0:40 — Show it working, three states
- Click **"Try a dry sample"** → point at the label, confidence %, probability bars.
- Click **"Try a damp sample"** → trend badge appears (this is the second upload,
  so the trend chart now has a real line).
- Click **"Try a wet sample"** → suggestion text changes to the wet-tyre message.
> "Nothing here is an LLM guessing — the trend and the suggestion are both
> deterministic rules over the model's output, so you can audit exactly why
> it said what it said."

## 0:40–1:00 — The video feature (the differentiator most teams won't have)
- Drop `test_sample_video.mp4` (or your own short clip) onto the upload zone.
> "This is a 6-second clip, dry to wet. One upload, and the whole trend chart
> builds itself — the backend samples frames, classifies each one, same
> pipeline as the single-image path."

## 1:00–1:30 — The part that shows real engineering, not just a working demo
> "We didn't just train a model and ship it. We trained on a public road-surface
> dataset from Hugging Face, then measured it against real racing photos — and
> it failed hard: 34% accuracy, misreading 97% of dry pit-lane images as wet.
> That's a real, measured domain gap, not something we're guessing at. We ran
> two rounds of targeted fine-tuning on racing-specific images, got the
> false-wet rate down to 20%, and when we tried pushing further — specifically
> targeting the DAMP class — we found and *documented* a real accuracy
> trade-off, and made the call not to ship that version. That whole
> investigation is in the README."

*(If asked "why not just keep tuning until it's perfect": because the trade-off
we found was real and instructive — chasing one metric degraded another, which
is a finding worth keeping, not a bug worth hiding.)*

## 1:30–1:50 — Reliability, briefly
> "It's hardened against bad input — corrupt files, spoofed content types,
> concurrent uploads — we found and fixed two real bugs doing that testing
> pass, both documented in the commit history. And it's honestly limited too:
> the caveat under every prediction says exactly that — this is one input
> signal, not a standalone call."

## 1:50–2:00 — Close
> "Working end to end, deployed, documented with real numbers throughout —
> not just claims. That's TrackPulse."

---

## Anticipated Q&A

**"Why MobileNetV3-Small and not a bigger model?"**
Fast enough to not be the bottleneck (p50 1.78ms/frame on CPU, no GPU needed),
small enough to fine-tune in minutes per round on consumer hardware, and this is
a classification task with 3 classes — capacity wasn't the limiting factor,
racing-domain training data was (see README's ablation section).

**"What's DAMP's real accuracy?"**
Honest answer: the production model (exp02) has *never had a direct DAMP
measurement on racing imagery* — our held-out eval set had zero DAMP examples
until a later experiment (exp03) specifically built one. That experiment measured
83% DAMP recall, but achieving it cost DRY accuracy, so it wasn't shipped. State
this plainly if asked — it's in the README, don't get caught contradicting your
own documentation.

**"Could this scale to a real F1 team?"**
Not as-is — SQLite, no auth, free-tier hosting are demo-grade, not production-grade,
and we say so directly in the README's Impact & Scalability section. The part that
*does* scale without a rearchitecture is the inference pipeline itself (stateless,
horizontally scalable). The part that's real, costed work is domain adaptation to
a new camera/track setup — which is exactly what this project spent most of its
effort proving out and measuring, not hand-waving.

**"What would you do with more time?"**
Grow the racing-domain eval set (still only 49-67 hand-labeled images), get real
DAMP training data at scale, and try class-weighted loss instead of oversampling
for the DAMP/DRY trade-off — that's the next experiment the README points to.
