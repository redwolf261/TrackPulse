import { type PredictResponse } from "./api";

// Every value rendered here comes directly from the classifier's own softmax
// output (probabilities/confidence) or the deterministic trend/evidence logic
// in the backend — nothing here is a physical measurement or an invented
// unit. See README's "Impact & scalability" and Evidence Trail sections for
// why that distinction matters for this project specifically.

const LABEL_COLOR: Record<string, string> = {
  DRY: "#10b981",
  DAMP: "#f59e0b",
  WET: "#3b82f6",
};

const TREND_META: Record<string, { color: string; icon: string; text: string }> = {
  WETTING: { color: "#3b82f6", icon: "↑", text: "Trending wetter" },
  DRYING: { color: "#f59e0b", icon: "↓", text: "Trending drier" },
  STABLE: { color: "#6b7a92", icon: "→", text: "Stable" },
};

interface GaugeProps {
  value: number; // 0-100, the winning class's confidence
  color: string;
}

function ConfidenceGauge({ value, color }: GaugeProps) {
  const R = 66;
  const CIRC = 2 * Math.PI * R;
  const SWEEP = CIRC * 0.75;
  const GAP = CIRC - SWEEP;
  const offset = SWEEP * (1 - value / 100);

  return (
    <svg viewBox="-90 -90 180 180" className="grip-gauge-svg" aria-label={`Model confidence ${value}%`}>
      <circle
        cx="0" cy="0" r={R} fill="none" stroke="#1a2535" strokeWidth="14"
        strokeDasharray={`${SWEEP} ${GAP}`} strokeDashoffset={-GAP / 2}
        strokeLinecap="round" transform="rotate(135)"
      />
      <circle
        cx="0" cy="0" r={R} fill="none" stroke={color} strokeWidth="14"
        strokeDasharray={`${SWEEP - offset} ${CIRC - (SWEEP - offset)}`}
        strokeDashoffset={-GAP / 2} strokeLinecap="round" transform="rotate(135)"
        style={{ transition: "stroke-dasharray 0.7s cubic-bezier(.4,0,.2,1), stroke 0.5s ease" }}
        filter={`drop-shadow(0 0 6px ${color}88)`}
      />
      <text x="0" y="4" textAnchor="middle" dominantBaseline="middle"
        fill={color} fontSize="26" fontWeight="700" fontFamily="JetBrains Mono, monospace">
        {value}
      </text>
      <text x="0" y="26" textAnchor="middle" dominantBaseline="middle"
        fill="#6b7a92" fontSize="10" fontFamily="Inter, sans-serif">
        CONFIDENCE %
      </text>
    </svg>
  );
}

interface TelemetryHUDProps {
  latest: PredictResponse | null;
}

export default function TelemetryHUD({ latest }: TelemetryHUDProps) {
  const label = latest?.label ?? null;
  const confidencePct = latest ? Math.round(latest.confidence * 100) : 0;
  const color = label ? LABEL_COLOR[label] : "#6b7a92";
  const trend = latest?.trend ?? "STABLE";
  const trendMeta = TREND_META[trend];

  return (
    <div className="telemetry-hud">
      <div className="grip-gauge-wrap">
        <ConfidenceGauge value={confidencePct} color={color} />
        <div className="grip-compound">
          <span className="grip-compound-label">CURRENT READING</span>
          <span className="grip-compound-val" style={{ color }}>
            {label ?? "—"}
          </span>
        </div>
      </div>

      {latest && (
        <div
          className="pit-callout"
          style={{ borderColor: trendMeta.color, color: trendMeta.color }}
        >
          <span className="pit-callout-icon">{trendMeta.icon}</span>
          <span className="pit-callout-text">{trendMeta.text} — {latest.suggestion}</span>
        </div>
      )}

      {latest && (
        <div className="compound-matrix">
          <div className="compound-matrix-title">CLASS PROBABILITIES (MODEL OUTPUT)</div>
          {(["DRY", "DAMP", "WET"] as const).map((cls) => {
            const p = Math.round(latest.probabilities[cls] * 100);
            const isTop = cls === label;
            const col = LABEL_COLOR[cls];
            return (
              <div key={cls} className={`compound-row ${isTop ? "compound-row--optimal" : ""}`}
                style={{ borderColor: isTop ? col : "transparent" }}>
                <span className="compound-name" style={{ color: isTop ? col : "#6b7a92" }}>
                  {isTop ? "▶ " : "  "}{cls}
                </span>
                <div className="compound-bar-track">
                  <div
                    className="compound-bar-fill"
                    style={{
                      width: `${p}%`,
                      background: `linear-gradient(90deg, ${col}44, ${col})`,
                      transition: "width 0.7s cubic-bezier(.4,0,.2,1)",
                    }}
                  />
                </div>
                <span className="compound-delta" style={{ color: isTop ? col : "#6b7a92" }}>
                  {p}%
                </span>
              </div>
            );
          })}
        </div>
      )}

      {latest && (
        <div className="lap-delta-row">
          <span className="lap-delta-label">TRUST LEVEL</span>
          <span className="lap-delta-value" style={{ color }}>
            {latest.evidence.trust}
          </span>
        </div>
      )}
    </div>
  );
}
