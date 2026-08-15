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

const CONDITION_RANK: Record<string, number> = { DRY: 0, DAMP: 1, WET: 2 };

function newSessionId() {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function App() {
  const [sessionId, setSessionId] = useState(newSessionId());
  const [latest, setLatest] = useState<PredictResponse | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [previewIsVideo, setPreviewIsVideo] = useState(false);
  const [history, setHistory] = useState<HistoryObservation[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingMessage, setLoadingMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [modelStatus, setModelStatus] = useState<{ loaded: boolean; source: string } | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  // Tracks the currently live object URL so we can revoke the previous one
  // before creating a new one — prevents memory accumulation across uploads.
  const previewUrlRef = useRef<string | null>(null);
  // Tracks the session that was active when each async request started so
  // we can drop stale responses if the user resets the session mid-flight.
  const activeSessionRef = useRef(sessionId);

  const ACCEPTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];
  const ACCEPTED_VIDEO_TYPES = ["video/mp4", "video/quicktime", "video/webm", "video/x-msvideo"];
  const MAX_IMAGE_BYTES = 10 * 1024 * 1024;
  const MAX_VIDEO_BYTES = 100 * 1024 * 1024;

  // Keep the session ref in sync so async handlers can detect staleness.
  useEffect(() => {
    activeSessionRef.current = sessionId;
  }, [sessionId]);

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
      // non-fatal: chart just stays empty
    }
  }, []);

  const setPreviewSafe = useCallback((url: string | null, isVideo: boolean) => {
    if (previewUrlRef.current) {
      URL.revokeObjectURL(previewUrlRef.current);
    }
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
        setError(
          `Unsupported file type "${file.type || "unknown"}" — please use JPEG, PNG, WebP, MP4, MOV, WebM, or AVI.`
        );
        return;
      }
      if (file.size === 0) {
        setError("That file is empty.");
        return;
      }

      // Capture the session at the moment this request starts; if the user
      // resets the session while we await the server, we discard the result.
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
          if (activeSessionRef.current !== requestSession) return; // stale — session was reset
          if (result.observations.length > 0) {
            setLatest(result.observations[result.observations.length - 1]);
          }
          await refreshHistory(requestSession);
        } catch (e) {
          if (activeSessionRef.current !== requestSession) return;
          setError(e instanceof Error ? e.message : "Video processing failed");
        } finally {
          if (activeSessionRef.current === requestSession) {
            setLoading(false);
            setLoadingMessage(null);
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
        if (activeSessionRef.current !== requestSession) return; // stale
        setLatest(result);
        await refreshHistory(requestSession);
      } catch (e) {
        if (activeSessionRef.current !== requestSession) return;
        setError(e instanceof Error ? e.message : "Prediction failed");
      } finally {
        if (activeSessionRef.current === requestSession) {
          setLoading(false);
        }
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
        if (activeSessionRef.current === requestSession) {
          setLoading(false);
        }
      }
    },
    [sessionId, refreshHistory, setPreviewSafe]
  );

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const resetSession = () => {
    const sid = newSessionId();
    // Update the ref first so any in-flight async handlers see the new session
    // immediately and drop their stale results.
    activeSessionRef.current = sid;
    setSessionId(sid);
    setLatest(null);
    setPreviewSafe(null, false);
    setHistory([]);
    setError(null);
    setLoading(false);
    setLoadingMessage(null);
  };

  const chartData = history.map((obs, i) => ({
    index: i + 1,
    time: new Date(obs.created_at).toLocaleTimeString(),
    condition: CONDITION_RANK[obs.label],
    label: obs.label,
    pWet: Math.round(obs.probabilities.WET * 100),
  }));

  return (
    <div className="app">
      <header className="app-header">
        <h1>🏁 TrackPulse — Live Track Condition</h1>
        {modelStatus && (
          <span className={`model-badge ${modelStatus.loaded ? "trained" : "fallback"}`}>
            {modelStatus.loaded ? "trained model active" : "fallback heuristic (model not loaded)"}
          </span>
        )}
      </header>

      <div className="session-row">
        <span>Session: {sessionId}</span>
        <button onClick={resetSession}>New session</button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <div className="grid">
        <div className="panel">
          <div
            className={`dropzone ${dragActive ? "active" : ""}`}
            onDrop={onDrop}
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,video/mp4,video/quicktime,video/webm,video/x-msvideo"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFile(file);
              }}
            />
            {loading && <span className="spinner" aria-hidden="true" />}
            {loading
              ? loadingMessage ?? "Analyzing…"
              : "Drop a track photo or short video here, or click to upload"}
          </div>

          <div className="sample-row">
            <span>No image handy?</span>
            <button onClick={() => loadSample("/samples/sample-dry.jpg", "sample-dry.jpg")} disabled={loading}>
              Try a dry sample
            </button>
            <button onClick={() => loadSample("/samples/sample-damp.jpg", "sample-damp.jpg")} disabled={loading}>
              Try a damp sample
            </button>
            <button onClick={() => loadSample("/samples/sample-wet.jpg", "sample-wet.jpg")} disabled={loading}>
              Try a wet sample
            </button>
          </div>

          {previewUrl && previewIsVideo && (
            <video src={previewUrl} className="preview-img" controls muted />
          )}
          {previewUrl && !previewIsVideo && (
            <img src={previewUrl} alt="uploaded track frame" className="preview-img" />
          )}
          <p className="sample-credit">
            Sample photos: Wikimedia Commons, CC BY 2.0 / CC BY-SA 2.0 / CC BY-SA 4.0
            (2010 Chinese GP; 2021 Russian GP starting grid; Safety Car in Heavy Rain).
          </p>
        </div>

        <div className="panel">
          {!latest ? (
            <div className="empty-state">Upload an image to see the predicted condition.</div>
          ) : (
            <>
              <div className={`condition-label ${latest.label}`}>{latest.label}</div>
              <div className="confidence-row">
                {(latest.confidence * 100).toFixed(0)}% confidence
              </div>
              <span className={`trend-badge ${latest.trend}`}>
                {latest.trend === "WETTING" && "↑ Getting wetter"}
                {latest.trend === "DRYING" && "↓ Drying out"}
                {latest.trend === "STABLE" && "→ Stable"}
              </span>

              <div className="prob-bars">
                {(["DRY", "DAMP", "WET"] as const).map((k) => (
                  <div className="prob-bar-row" key={k}>
                    <span className="prob-bar-label">{k}</span>
                    <div className="prob-bar-track">
                      <div
                        className={`prob-bar-fill ${k}`}
                        style={{ width: `${latest.probabilities[k] * 100}%` }}
                      />
                    </div>
                    <span className="prob-bar-pct">{(latest.probabilities[k] * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>

              <div className={`suggestion ${latest.confidence < 0.45 ? "low-confidence" : ""}`}>
                💡 {latest.suggestion}
              </div>

              {latest.evidence && (
                <div className={`evidence-trail trust-${latest.evidence.trust}`}>
                  <div className="evidence-header">
                    <span className="evidence-trust-badge">{latest.evidence.trust} TRUST</span>
                    <span className="evidence-label">Why this reading?</span>
                  </div>
                  {latest.evidence.concerns.length > 0 && (
                    <ul className="evidence-list evidence-concerns">
                      {latest.evidence.concerns.map((c, i) => (
                        <li key={`concern-${i}`}>⚠ {c}</li>
                      ))}
                    </ul>
                  )}
                  {latest.evidence.reasons.length > 0 && (
                    <ul className="evidence-list evidence-reasons">
                      {latest.evidence.reasons.map((r, i) => (
                        <li key={`reason-${i}`}>✓ {r}</li>
                      ))}
                    </ul>
                  )}
                </div>
              )}

              <p className="model-caveat">
                This model is a single-frame visual estimate, not a physical wetness
                measurement — treat it as one input alongside weather and driver reports,
                not a standalone call.
              </p>
            </>
          )}
        </div>

        <div className="panel chart-panel">
          <h3 style={{ marginTop: 0 }}>Condition trend</h3>
          {chartData.length < 2 ? (
            <div className="empty-state">Upload at least two images to see a trend.</div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#223040" />
                <XAxis dataKey="index" stroke="#8b98a5" fontSize={12} />
                <YAxis
                  domain={[0, 2]}
                  ticks={[0, 1, 2]}
                  tickFormatter={(v) => ["DRY", "DAMP", "WET"][v as number]}
                  stroke="#8b98a5"
                  fontSize={12}
                  width={50}
                />
                <Tooltip
                  contentStyle={{ background: "#131a22", border: "1px solid #223040" }}
                  formatter={(_value, _name, props) => [props.payload.label, "Condition"]}
                  labelFormatter={(i) => `Frame #${i}`}
                />
                <Line
                  type="stepAfter"
                  dataKey="condition"
                  stroke="#2563eb"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}
