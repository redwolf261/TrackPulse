import { useState, useRef, useEffect, useCallback } from "react";
import { predict, type PredictResponse } from "./api";

const SCENARIOS = [
  {
    id: "dry",
    label: "☀️ DRY",
    frames: [
      { url: "/samples/sample-dry.jpg",  name: "sample-dry.jpg"  },
      { url: "/samples/sample-dry.jpg",  name: "sample-dry-2.jpg" },
      { url: "/samples/sample-damp.jpg", name: "sample-damp.jpg" },
    ],
    description: "Clear conditions → surface drying sequence",
  },
  {
    id: "mixed",
    label: "⛅ MIXED",
    frames: [
      { url: "/samples/sample-dry.jpg",  name: "lap1-s1.jpg" },
      { url: "/samples/sample-damp.jpg", name: "lap1-s2.jpg" },
      { url: "/samples/sample-wet.jpg",  name: "lap1-s3.jpg" },
    ],
    description: "Track surface transition: Dry → Damp → Wet",
  },
  {
    id: "wet",
    label: "🌧️ WET",
    frames: [
      { url: "/samples/sample-wet.jpg",  name: "wet-s1.jpg"  },
      { url: "/samples/sample-wet.jpg",  name: "wet-s2.jpg"  },
      { url: "/samples/sample-damp.jpg", name: "wet-dry.jpg" },
    ],
    description: "Full wet → drying tendency",
  },
];

interface RaceSimulatorProps {
  sessionId: string;
  onResult: (r: PredictResponse) => void;
}

export default function RaceSimulator({ sessionId, onResult }: RaceSimulatorProps) {
  const [running, setRunning]   = useState(false);
  const [scenario, setScenario] = useState(0);
  const [frame, setFrame]       = useState(0);
  const [status, setStatus]     = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const sessionRef = useRef(sessionId);
  useEffect(() => { sessionRef.current = sessionId; }, [sessionId]);

  const totalFrames = SCENARIOS[scenario].frames.length;

  const runFrame = useCallback(
    async (frameIdx: number, scenIdx: number) => {
      const sc = SCENARIOS[scenIdx];
      const f  = sc.frames[frameIdx];
      setStatus(`Analyzing frame ${frameIdx + 1}/${sc.frames.length} — ${f.name}`);
      setProgress(Math.round(((frameIdx + 1) / sc.frames.length) * 100));
      try {
        const blob = await fetch(f.url).then((r) => r.blob());
        const file = new File([blob], f.name, { type: blob.type || "image/jpeg" });
        const result = await predict(file, sessionRef.current);
        onResult(result);
      } catch {
        setStatus("⚠ Frame analysis failed — check backend connection");
      }
    },
    [onResult]
  );

  const start = useCallback(
    async (scenIdx: number) => {
      setRunning(true);
      setFrame(0);
      setProgress(0);
      const sc = SCENARIOS[scenIdx];
      for (let i = 0; i < sc.frames.length; i++) {
        setFrame(i);
        await runFrame(i, scenIdx);
        // Pause between frames so the user can see each result
        if (i < sc.frames.length - 1) {
          await new Promise<void>((res) => {
            timerRef.current = setTimeout(res, 2200);
          });
        }
      }
      setStatus("✓ Simulation complete");
      setRunning(false);
    },
    [runFrame]
  );

  const stop = () => {
    if (timerRef.current) clearTimeout(timerRef.current);
    setRunning(false);
    setStatus(null);
    setProgress(0);
    setFrame(0);
  };

  useEffect(() => () => { if (timerRef.current) clearTimeout(timerRef.current); }, []);

  return (
    <div className="sim-panel">
      <div className="sim-header">
        <span className="sim-title">🎙 RACE SIMULATOR</span>
        {running && <span className="sim-live-badge">LIVE</span>}
      </div>

      {/* Scenario selector */}
      <div className="sim-scenarios">
        {SCENARIOS.map((sc, i) => (
          <button
            key={sc.id}
            id={`sim-scenario-${sc.id}`}
            className={`sim-scenario-btn ${scenario === i ? "active" : ""}`}
            disabled={running}
            onClick={() => setScenario(i)}
          >
            {sc.label}
          </button>
        ))}
      </div>
      <p className="sim-desc">{SCENARIOS[scenario].description}</p>

      {/* Progress bar */}
      {running && (
        <div className="sim-progress-wrap">
          <div
            className="sim-progress-fill"
            style={{ width: `${progress}%`, transition: "width 0.4s ease" }}
          />
        </div>
      )}
      {status && <div className="sim-status">{status}</div>}

      {/* Frame indicator dots */}
      <div className="sim-dots">
        {Array.from({ length: totalFrames }).map((_, i) => (
          <span
            key={i}
            className={`sim-dot ${i === frame && running ? "active" : ""} ${i < frame || (!running && progress === 100) ? "done" : ""}`}
          />
        ))}
      </div>

      {/* Controls */}
      <div className="sim-controls">
        <button
          id="sim-start-btn"
          className="sim-start-btn"
          disabled={running}
          onClick={() => start(scenario)}
        >
          {running ? "Running…" : "▶ Run Simulation"}
        </button>
        {running && (
          <button id="sim-stop-btn" className="sim-stop-btn" onClick={stop}>
            ■ Stop
          </button>
        )}
      </div>
    </div>
  );
}
