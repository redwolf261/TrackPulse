import { useCallback, useEffect, useRef, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./App.css";
import {
  getHealth,
  getHistory,
  predict,
  predictVideo,
  type HistoryObservation,
  type PredictResponse,
} from "./api";
import CircuitMap from "./CircuitMap";
import TelemetryHUD from "./TelemetryHUD";
import RaceSimulator from "./RaceSimulator";
import SurfaceAnalyzer from "./SurfaceAnalyzer";

const CONDITION_RANK: Record<string, number> = { DRY: 0, DAMP: 1, WET: 2 };
const CONDITION_COLOR: Record<string, string> = {
  DRY:  "#10b981",
  DAMP: "#f59e0b",
  WET:  "#3b82f6",
};

function newSessionId() {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function App() {
  const [sessionId,      setSessionId]      = useState(newSessionId());
  const [latest,         setLatest]         = useState<PredictResponse | null>(null);
  const [previewUrl,     setPreviewUrl]     = useState<string | null>(null);
  const [previewIsVideo, setPreviewIsVideo] = useState(false);
  const [history,        setHistory]        = useState<HistoryObservation[]>([]);
  const [loading,        setLoading]        = useState(false);
  const [loadingMessage, setLoadingMessage] = useState<string | null>(null);
  const [error,          setError]          = useState<string | null>(null);
  const [modelStatus,    setModelStatus]    = useState<{ loaded: boolean; source: string } | null>(null);
  const [dragActive,     setDragActive]     = useState(false);
  const [selectedCorner, setSelectedCorner] = useState("ALL");

  const fileInputRef    = useRef<HTMLInputElement>(null);
  const previewUrlRef   = useRef<string | null>(null);
  const activeSessionRef = useRef(sessionId);

  const ACCEPTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];
  const ACCEPTED_VIDEO_TYPES = ["video/mp4", "video/quicktime", "video/webm", "video/x-msvideo"];
  const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
  const MAX_VIDEO_BYTES = 100 * 1024 * 1024;

  useEffect(() => { activeSessionRef.current = sessionId; }, [sessionId]);

  useEffect(() => {
    getHealth()
      .then((h) => setModelStatus({ loaded: h.model_loaded, source: h.model_source }))
      .catch(() => setModelStatus(null));
  }, []);

  const refreshHistory = useCallback(async (sid: string) => {
    try {
      const h = await getHistory(sid);
      setHistory(h.observations);
    } catch {
      // non-fatal
    }
  }, []);

  const setPreviewSafe = useCallback((url: string | null, isVideo: boolean) => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = url;
    setPreviewUrl(url);
    setPreviewIsVideo(isVideo);
  }, []);

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);
      const isVideo = ACCEPTED_VIDEO_TYPES.includes(file.type);
      const isImage = ACCEPTED_IMAGE_TYPES.includes(file.type);
      if (!isVideo && !isImage) {
        setError(`Unsupported file type "${file.type || "unknown"}" — please use JPEG, PNG, WebP, MP4, MOV, WebM, or AVI.`);
        return;
      }
      if (file.size === 0) { setError("That file is empty."); return; }

      const requestSession = sessionId;

      if (isVideo) {
        if (file.size > MAX_VIDEO_BYTES) {
          setError(`Video is too large (${(file.size / 1024 / 1024).toFixed(1)}MB) — max is 100MB.`);
          return;
        }
        setLoading(true);
        setLoadingMessage("Extracting and analyzing frames — this can take a moment…");
        setPreviewSafe(URL.createObjectURL(file), true);
        try {
          const result = await predictVideo(file, requestSession);
          if (activeSessionRef.current !== requestSession) return;
          if (result.observations.length > 0) {
            setLatest(result.observations[result.observations.length - 1]);
          }
          await refreshHistory(requestSession);
        } catch (e) {
          if (activeSessionRef.current !== requestSession) return;
          setError(e instanceof Error ? e.message : "Video processing failed");
        } finally {
          if (activeSessionRef.current === requestSession) {
            setLoading(false); setLoadingMessage(null);
          }
        }
        return;
      }

      if (file.size > MAX_IMAGE_BYTES) {
        setError(`Image is too large (${(file.size / 1024 / 1024).toFixed(1)}MB) — max is 10MB.`);
        return;
      }
      setLoading(true);
      setPreviewSafe(URL.createObjectURL(file), false);
      try {
        const result = await predict(file, requestSession);
        if (activeSessionRef.current !== requestSession) return;
        setLatest(result);
        await refreshHistory(requestSession);
      } catch (e) {
        if (activeSessionRef.current !== requestSession) return;
        setError(e instanceof Error ? e.message : "Prediction failed");
      } finally {
        if (activeSessionRef.current === requestSession) setLoading(false);
      }
    },
    [sessionId, refreshHistory, setPreviewSafe]
  );

  const loadSample = useCallback(
    async (url: string, filename: string) => {
      setError(null);
      setLoading(true);
      const requestSession = sessionId;
      try {
        const res = await fetch(url);
        const blob = await res.blob();
        const file = new File([blob], filename, { type: blob.type || "image/jpeg" });
        if (activeSessionRef.current !== requestSession) return;
        setPreviewSafe(URL.createObjectURL(file), false);
        const result = await predict(file, requestSession);
        if (activeSessionRef.current !== requestSession) return;
        setLatest(result);
        await refreshHistory(requestSession);
      } catch (e) {
        if (activeSessionRef.current !== requestSession) return;
        setError(e instanceof Error ? e.message : "Failed to load sample image");
      } finally {
        if (activeSessionRef.current === requestSession) setLoading(false);
      }
    },
    [sessionId, refreshHistory, setPreviewSafe]
  );

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault(); setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const resetSession = () => {
    const sid = newSessionId();
    activeSessionRef.current = sid;
    setSessionId(sid);
    setLatest(null);
    setPreviewSafe(null, false);
    setHistory([]);
    setError(null);
    setLoading(false);
    setLoadingMessage(null);
    setSelectedCorner("ALL");
  };

  // Chart data
  const chartData = history.map((obs, i) => ({
    index: i + 1,
    time: new Date(obs.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
    condition: CONDITION_RANK[obs.label],
    confidence: Math.round(obs.confidence * 100),
    label: obs.label,
  }));

  return (
    <div className="pitwall">
      {/* ═══ COMMAND HEADER ═══════════════════════════════════════════════════ */}
      <header className="cmd-header">
        <div className="cmd-logo">
          <span className="cmd-logo-flag">🏁</span>
          <span className="cmd-logo-name">TrackPulse <span className="cmd-logo-pro">PRO</span></span>
        </div>

        <div className="cmd-telemetry-strip">
          <div className="cmd-tel-item">
            <span className="cmd-tel-label">CIRCUIT</span>
            <span className="cmd-tel-value">SILVERSTONE GP</span>
          </div>
          <div className="cmd-tel-sep" />
          {latest && (
            <>
              <div className="cmd-tel-item">
                <span className="cmd-tel-label">CONDITION</span>
                <span className="cmd-tel-value" style={{ color: CONDITION_COLOR[latest.label] }}>
                  {latest.label} · {(latest.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div className="cmd-tel-sep" />
              <div className="cmd-tel-item">
                <span className="cmd-tel-label">TREND</span>
                <span className="cmd-tel-value" style={{ color: CONDITION_COLOR[latest.label] }}>
                  {latest.trend}
                </span>
              </div>
              <div className="cmd-tel-sep" />
            </>
          )}
          <div className="cmd-tel-item">
            <span className="cmd-tel-label">MODEL</span>
            <span className={`cmd-tel-value ${modelStatus?.loaded ? "text-green" : "text-amber"}`}>
              {modelStatus?.loaded ? "ONNX 1.78ms" : "—"}
            </span>
          </div>
        </div>

        <div className="cmd-actions">
          {error && <span className="cmd-error-dot" title={error}>⚠</span>}
          <button id="reset-session-btn" className="cmd-reset-btn" onClick={resetSession}>
            ↺ New Session
          </button>
        </div>
      </header>

      {/* ═══ ERROR BANNER ═════════════════════════════════════════════════════ */}
      {error && (
        <div className="pw-error-banner">
          <span>⚠ {error}</span>
          <button onClick={() => setError(null)}>×</button>
        </div>
      )}

      {/* ═══ MAIN 3-COLUMN GRID ══════════════════════════════════════════════ */}
      <div className="pw-grid">

        {/* ─── LEFT: Camera / Upload / Analyzer / Simulator ─────────────── */}
        <aside className="pw-col pw-col-left">

          {/* Upload / Camera Panel */}
          <div className="pw-card">
            <div className="pw-card-title">📷 CAMERA FEED</div>

            {/* Drop Zone */}
            <div
              id="dropzone"
              className={`dropzone ${dragActive ? "active" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={onDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime,video/webm,video/x-msvideo"
                onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); e.target.value = ""; }}
              />
              {loading ? (
                <span className="dz-loading">
                  <span className="spinner" />
                  <span>{loadingMessage ?? "Analyzing…"}</span>
                </span>
              ) : (
                <span className="dz-hint">
                  <span className="dz-icon">↑</span>
                  Drop image or video, or click to browse
                </span>
              )}
            </div>

            {/* Preview */}
            {previewUrl && (
              previewIsVideo
                ? <video src={previewUrl} className="preview-media" controls muted playsInline />
                : <img    src={previewUrl} className="preview-media" alt="Track preview" />
            )}

            {/* Sample triggers */}
            <div className="sample-row">
              <span>Samples:</span>
              {[
                { label: "☀ Dry",  url: "/samples/sample-dry.jpg",  name: "sample-dry.jpg"  },
                { label: "⛅ Damp", url: "/samples/sample-damp.jpg", name: "sample-damp.jpg" },
                { label: "🌧 Wet", url: "/samples/sample-wet.jpg",  name: "sample-wet.jpg"  },
              ].map((s) => (
                <button
                  key={s.label}
                  id={`sample-${s.name}`}
                  disabled={loading}
                  onClick={() => loadSample(s.url, s.name)}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          {/* Surface Analyzer */}
          <SurfaceAnalyzer previewUrl={previewUrl} isVideo={previewIsVideo} />

          {/* Race Simulator */}
          <RaceSimulator sessionId={sessionId} onResult={(r) => {
            setLatest(r);
            refreshHistory(sessionId);
          }} />
        </aside>

        {/* ─── CENTER: Circuit Map ──────────────────────────────────────── */}
        <section className="pw-col pw-col-center">
          <div className="pw-card pw-card-circuit">
            <CircuitMap
              latest={latest}
              selectedCorner={selectedCorner}
              onCornerSelect={setSelectedCorner}
            />
          </div>

          {/* Session telemetry trend chart */}
          {chartData.length > 0 && (
            <div className="pw-card pw-card-chart">
              <div className="pw-card-title">📈 SESSION TREND — {history.length} OBSERVATION{history.length !== 1 ? "S" : ""}</div>
              <ResponsiveContainer width="100%" height={130}>
                <LineChart data={chartData} margin={{ top: 8, right: 12, left: -20, bottom: 0 }}>
                  <CartesianGrid stroke="#1a2535" strokeDasharray="4 4" />
                  <XAxis dataKey="index" tick={{ fill: "#4b5563", fontSize: 10 }} />
                  <YAxis
                    domain={[-0.2, 2.2]}
                    tickFormatter={(v: number) => ["DRY", "DAMP", "WET"][Math.round(v)] ?? ""}
                    tick={{ fill: "#4b5563", fontSize: 9 }}
                  />
                  <Tooltip
                    contentStyle={{ background: "#0d1a26", border: "1px solid #1e2d3d", borderRadius: 8, fontSize: 12 }}
                    // eslint-disable-next-line @typescript-eslint/no-explicit-any
                    formatter={(_v: any, _n: any, entry: any) =>
                      [`${entry.payload?.label ?? "?"} (${entry.payload?.confidence ?? "?"}%)`, "Condition"] as [string, string]}
                    labelFormatter={(i: unknown) => `Frame ${i}`}
                  />
                  <Line
                    dataKey="condition"
                    stroke="#3b82f6"
                    strokeWidth={2}
                    dot={{ r: 3, fill: "#3b82f6" }}
                    activeDot={{ r: 5, stroke: "#93c5fd" }}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>

        {/* ─── RIGHT: Telemetry HUD + Prediction Result ────────────────── */}
        <aside className="pw-col pw-col-right">

          {/* Prediction result */}
          {latest ? (
            <div className="pw-card pw-card-result">
              <div className="pw-card-title">🔬 AI CONDITION ANALYSIS</div>

              {/* Big label + confidence */}
              <div className="result-label-row">
                <span className={`condition-label ${latest.label}`}>{latest.label}</span>
                <div className="result-meta">
                  <span className="result-confidence">{(latest.confidence * 100).toFixed(1)}%</span>
                  <span className={`trend-badge ${latest.trend}`}>{latest.trend}</span>
                </div>
              </div>

              {/* Probability bars */}
              <div className="prob-bars">
                {(["DRY", "DAMP", "WET"] as const).map((cls) => (
                  <div key={cls} className="prob-bar-row">
                    <span className="prob-bar-label">{cls}</span>
                    <div className="prob-bar-track">
                      <div
                        className={`prob-bar-fill ${cls}`}
                        style={{ width: `${latest.probabilities[cls] * 100}%` }}
                      />
                    </div>
                    <span className="prob-bar-pct">{(latest.probabilities[cls] * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>

              {/* Suggestion */}
              <div className={`suggestion ${latest.confidence < 0.45 ? "low-confidence" : ""}`}>
                {latest.suggestion}
              </div>

              {/* Evidence Trail */}
              <div className={`evidence-trail trust-${latest.evidence.trust}`}>
                <div className="evidence-header">
                  <span className="evidence-trust-badge">{latest.evidence.trust} TRUST</span>
                  <span className="evidence-label">Evidence Trail</span>
                </div>
                {latest.evidence.concerns.length > 0 && (
                  <ul className="evidence-list evidence-concerns">
                    {latest.evidence.concerns.map((c, i) => <li key={i}>⚠ {c}</li>)}
                  </ul>
                )}
                {latest.evidence.reasons.length > 0 && (
                  <ul className="evidence-list evidence-reasons">
                    {latest.evidence.reasons.map((r, i) => <li key={i}>· {r}</li>)}
                  </ul>
                )}
              </div>

              {/* Model source */}
              <div className="model-caveat">
                {latest.model_source === "fallback-heuristic"
                  ? "⚠ Fallback heuristic — no trained model loaded."
                  : "MobileNetV3-Small (exp02) · ONNX Runtime · CPUExecutionProvider"}
              </div>
            </div>
          ) : (
            <div className="pw-card pw-card-result pw-empty-result">
              <div className="empty-state">
                <div className="empty-state-icon">📷</div>
                <div>Upload a track image or run the Race Simulator to begin analysis</div>
              </div>
            </div>
          )}

          {/* Telemetry HUD — all values below are the model's own softmax
              output and deterministic trend/evidence logic, not measurements */}
          <div className="pw-card">
            <div className="pw-card-title">⚡ CLASSIFIER TELEMETRY</div>
            <TelemetryHUD latest={latest} />
          </div>

        </aside>
      </div>

      {/* ═══ FOOTER ═══════════════════════════════════════════════════════════ */}
      <footer className="pw-footer">
        <span>Session: <code>{sessionId.slice(-12)}</code></span>
        <span>·</span>
        <span>TrackPulse Pro · <a href="https://github.com/redwolf261/TrackPulse" target="_blank" rel="noreferrer">GitHub</a></span>
        <span>·</span>
        <span>MobileNetV3-Small · exp02 · {history.length} observation{history.length !== 1 ? "s" : ""}</span>
      </footer>
    </div>
  );
}
