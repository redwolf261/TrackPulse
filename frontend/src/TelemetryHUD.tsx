import { type TelemetryData } from "./api";

const COMPOUND_COLORS: Record<string, string> = {
  SLICK:        "#10b981",
  INTERMEDIATE: "#f59e0b",
  FULL_WET:     "#3b82f6",
};

const STATUS_META: Record<string, { color: string; icon: string }> = {
  OPTIMAL:              { color: "#10b981", icon: "✓" },
  CROSSOVER_APPROACHING:{ color: "#f59e0b", icon: "⚑" },
  CROSSOVER_ACTIVE:     { color: "#ef4444", icon: "⚡" },
};

interface GaugeProps { value: number; color: string }

function GripGauge({ value, color }: GaugeProps) {
  // SVG arc gauge: 270° sweep, starts at 7 o'clock, ends at 5 o'clock
  const R = 66;
  const CIRC = 2 * Math.PI * R;
  const SWEEP = CIRC * 0.75;          // 270° = 75% of full circle
  const GAP   = CIRC - SWEEP;
  const offset = SWEEP * (1 - value / 100); // how much arc to "empty"

  return (
    <svg
      viewBox="-90 -90 180 180"
      className="grip-gauge-svg"
      aria-label={`Track grip ${value}%`}
    >
      {/* Background track */}
      <circle
        cx="0" cy="0" r={R}
        fill="none"
        stroke="#1a2535"
        strokeWidth="14"
        strokeDasharray={`${SWEEP} ${GAP}`}
        strokeDashoffset={-GAP / 2}
        strokeLinecap="round"
        transform="rotate(135)"
      />
      {/* Coloured fill arc */}
      <circle
        cx="0" cy="0" r={R}
        fill="none"
        stroke={color}
        strokeWidth="14"
        strokeDasharray={`${SWEEP - offset} ${CIRC - (SWEEP - offset)}`}
        strokeDashoffset={-GAP / 2}
        strokeLinecap="round"
        transform="rotate(135)"
        style={{ transition: "stroke-dasharray 0.7s cubic-bezier(.4,0,.2,1), stroke 0.5s ease" }}
        filter={`drop-shadow(0 0 6px ${color}88)`}
      />
      {/* Grip % text */}
      <text x="0" y="4" textAnchor="middle" dominantBaseline="middle"
        fill={color} fontSize="26" fontWeight="700"
        fontFamily="JetBrains Mono, monospace"
        style={{ transition: "fill 0.5s ease" }}
      >
        {value}
      </text>
      <text x="0" y="26" textAnchor="middle" dominantBaseline="middle"
        fill="#6b7a92" fontSize="10" fontFamily="Inter, sans-serif"
      >
        GRIP %
      </text>
    </svg>
  );
}

interface TelemetryHUDProps {
  telemetry: TelemetryData | null | undefined;
  label: "DRY" | "DAMP" | "WET" | null;
}

const LABEL_GRIP_COLOR: Record<string, string> = {
  DRY:  "#10b981",
  DAMP: "#f59e0b",
  WET:  "#3b82f6",
};

export default function TelemetryHUD({ telemetry, label }: TelemetryHUDProps) {
  const grip = telemetry?.grip_index ?? 82;
  const gripColor = label ? LABEL_GRIP_COLOR[label] : "#10b981";
  const status = telemetry?.crossover_status ?? "OPTIMAL";
  const meta = STATUS_META[status] ?? STATUS_META["OPTIMAL"];
  const deltas = telemetry?.compound_deltas;

  return (
    <div className="telemetry-hud">
      {/* Grip Gauge */}
      <div className="grip-gauge-wrap">
        <GripGauge value={grip} color={gripColor} />
        <div className="grip-compound">
          <span className="grip-compound-label">OPTIMAL TYRE</span>
          <span className="grip-compound-val" style={{ color: gripColor }}>
            {telemetry?.optimal_compound ?? "—"}
          </span>
        </div>
      </div>

      {/* Pit Strategy Callout */}
      {telemetry && (
        <div
          className={`pit-callout pit-callout-${status.toLowerCase()}`}
          style={{ borderColor: meta.color, color: meta.color }}
        >
          <span className="pit-callout-icon">{meta.icon}</span>
          <span className="pit-callout-text">{telemetry.crossover_message}</span>
          {status !== "OPTIMAL" && (
            <span className="pit-callout-pulse" style={{ background: meta.color }} />
          )}
        </div>
      )}

      {/* Compound Delta Matrix */}
      {deltas && (
        <div className="compound-matrix">
          <div className="compound-matrix-title">LAP Δ vs DRY BENCHMARK</div>
          {(["SLICK", "INTERMEDIATE", "FULL_WET"] as const).map((c) => {
            const delta = deltas[c];
            const isOptimal = telemetry?.optimal_compound?.toUpperCase().startsWith(c.split(" ")[0]);
            const col = COMPOUND_COLORS[c];
            const barWidth = Math.min(100, (delta / 15) * 100);
            return (
              <div key={c} className={`compound-row ${isOptimal ? "compound-row--optimal" : ""}`}
                style={{ borderColor: isOptimal ? col : "transparent" }}>
                <span className="compound-name" style={{ color: isOptimal ? col : "#6b7a92" }}>
                  {isOptimal ? "▶ " : "  "}{c.replace("_", " ")}
                </span>
                <div className="compound-bar-track">
                  <div
                    className="compound-bar-fill"
                    style={{
                      width: `${barWidth}%`,
                      background: `linear-gradient(90deg, ${col}44, ${col})`,
                      transition: "width 0.7s cubic-bezier(.4,0,.2,1)",
                    }}
                  />
                </div>
                <span className="compound-delta" style={{ color: isOptimal ? col : "#6b7a92" }}>
                  +{delta.toFixed(1)}s
                </span>
              </div>
            );
          })}
        </div>
      )}

      {/* Lap delta summary */}
      {telemetry && (
        <div className="lap-delta-row">
          <span className="lap-delta-label">BEST LAP Δ</span>
          <span className="lap-delta-value" style={{ color: gripColor }}>
            +{telemetry.lap_delta_seconds.toFixed(1)}s
          </span>
        </div>
      )}
    </div>
  );
}
