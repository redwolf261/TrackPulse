# TrackPulse — Mentoring Round Script (~2–3 minutes + open Q&A)

Live app: **https://trackpulserace.netlify.app**
Backend: https://trackpulse-backend-7kod.onrender.com (free tier — cold-starts after
idling; hit `/health` ~1 min before presenting to warm it up)

**Format note**: this is a mentoring round, not final judging — expect a conversation,
not a scored pitch. Mentors will likely interrupt with questions; that's fine, let
them. Camera on throughout. Criteria to keep in mind: Tech Stack, Technical
Complexity, Social Impact, Presentation, Overall Product, Business Model.

## Before you start
- Open the live site in a fresh tab (not localhost — show the real deployed thing).
- Have `demo/test_sample_video.mp4` ready in case the video-upload moment needs a
  local file instead of a sample button.
- Know your two numbers cold: **73% accuracy / 20% false-WET rate on racing imagery**
  (production model, exp02) and **43-point accuracy drop** measured going from
  road-camera to racing-camera data before adapting to it.
- Note: some things mentioned below (Postgres, rate limiting, Evidence Trail UI) are
  built and tested locally but **not yet on the live deployed site** — say so plainly
  if you demo them, don't imply they're live if a mentor might check the URL.

## 0:00–0:20 — The problem, in one breath
> "Track condition changes faster than a weather report can keep up, but a race
> engineer needs to know *right now* if the surface is getting safer or riskier.
> We built TrackPulse: upload a photo or video of the track, get a Dry/Damp/Wet
> call, a trend over time, and a plain suggestion — with the model running in
> under 2 milliseconds, no GPU needed."

## 0:20–0:50 — Show it working, three states + Evidence Trail
- Click **"Try a dry sample"** → label, confidence %, probability bars.
- Point out the **Evidence Trail** panel: "This isn't just a number — it tells you
  *why* to trust it. If the top two classes are close, or the confidence is only
  moderate, it says so explicitly, instead of showing '91% WET' and leaving you to
  guess whether that's actually reliable."
- Click **"Try a damp sample"**, then **"Try a wet sample"** → trend chart builds.
> "The trend and suggestion are deterministic rules over the model's output — no
> LLM guessing — so every claim is auditable back to a number."

## 0:50–1:10 — Video upload
- Drop `test_sample_video.mp4` onto the upload zone.
> "One video upload samples frames automatically and populates the whole trend —
> same pipeline as single images, just batched."

## 1:10–1:40 — Technical complexity, the real story
> "We didn't just train a model and ship it. We trained on a public road-surface
> dataset from Hugging Face, then measured it against real racing photos — and it
> failed hard: 34% accuracy, misreading 97% of dry pit-lane images as wet. That's a
> measured domain gap. We ran two rounds of racing-specific fine-tuning, got false-wet
> down to 20%, and when we pushed further on the DAMP class specifically, we found and
> *documented* a real accuracy trade-off — and chose not to ship that version. All of
> that is in the README, with real numbers, not just claims."

## 1:40–2:00 — Social impact & business model (mentors will likely ask, get ahead of it)
> "The realistic buyer isn't Formula 1 — they already have people and better sensors
> for this. It's the level below: club racing, karting, track days, driving schools,
> where a human spotter watching a camera is a real cost and this is a genuine
> substitute. Business-model-wise, the most buildable path from what we already have
> is usage-based API access — we already have the inference endpoint, it's a
> packaging and auth question, not new technology. We say plainly in the README what's
> unvalidated: no real customer conversations yet, this is an honest sketch, not a
> claim of market fit."

## 2:00–2:20 — Close
> "Working end to end, deployed, and documented with real measured numbers
> throughout — including the parts that didn't work on the first or second try.
> That's TrackPulse. Happy to go deeper on any of this."

---

## Mapping to the mentoring criteria (know these cold, don't read them aloud)

- **Tech Stack**: FastAPI + SQLite/Postgres-ready + ONNX Runtime backend, React/TS/Vite
  frontend, MobileNetV3-Small fine-tuned via torchvision, deployed on Render + Netlify.
- **Technical Complexity**: real 3-round fine-tuning pipeline with leakage-safe splits,
  measured domain-gap experiments, ONNX export + equivalence verification, video frame
  extraction, rate limiting, Evidence Trail (deterministic uncertainty reasoning).
- **Social Impact**: cheap, fast surface-condition perception for the racing tiers
  below professional F1 where a human spotter is a real cost — see Business Model
  section in README for the honest, non-inflated version of this.
- **Presentation**: this script + the live site + the README's methodology narrative.
- **Overall Product**: works end-to-end, live, tested (two real bugs found and fixed
  during an explicit testing pass — mention this, it's a strength not a weakness).
- **Business Model**: README's new "Business model" section — usage-based API is the
  most concretely buildable near-term path from what already exists.

## Anticipated Q&A

**"Why MobileNetV3-Small and not a bigger model?"**
Fast enough to not be the bottleneck (p50 1.78ms/frame on CPU, no GPU needed), small
enough to fine-tune in minutes per round on consumer hardware. Capacity wasn't the
limiting factor — racing-domain training data was (see README's ablation section).

**"What's DAMP's real accuracy?"**
Honest answer: the production model (exp02) has never had a direct DAMP measurement on
racing imagery — our held-out eval set had zero DAMP examples until a later experiment
(exp03) built one specifically. That measured 83% DAMP recall, but achieving it cost
DRY accuracy, so it wasn't shipped. Say this plainly — it's in the README.

**"Could this scale to a real F1 team?"**
Not as-is — no auth, free-tier hosting are demo-grade. Rate limiting and a
Postgres-ready persistence layer are built and tested but not yet deployed live
(say this honestly if asked, don't imply they're live). The part that *does* scale
without a rearchitecture is the inference pipeline itself — stateless, horizontally
scalable. The part that's real, costed work is domain adaptation to a new
camera/track setup, which is what this project actually spent its effort measuring.

**"What's the business model, really?"**
Usage-based API for teams/apps wanting to embed this is the most buildable near-term
option — the inference endpoint already exists, it's a packaging/auth/billing
question. SaaS dashboard and model-licensing are also plausible but need more
groundwork (auth/persistence for the former, resolving the RSCD dataset's
non-commercial license terms for the latter). No revenue/market claims — genuinely
haven't validated a customer yet, and say so if asked.

**"What would you do with more time?"**
Grow the racing-domain eval set (still only 49-67 hand-labeled images), get real DAMP
training data at scale, try class-weighted loss instead of oversampling for the
DAMP/DRY trade-off, and deploy the rate-limiting/Postgres work that's currently
tested locally but not live.
