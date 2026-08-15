import { type PredictResponse } from "./api";

// Corner definitions: id, label, position on 580×380 viewBox, sector (1/2/3)
export const CORNERS = [
  { id: "SF",          label: "S/F",         x: 472, y: 178, sector: 1 },
  { id: "copse",       label: "Copse",        x: 510, y: 138, sector: 1 },
  { id: "maggotts",   label: "Maggotts",     x: 482, y: 102, sector: 1 },
  { id: "becketts",   label: "Becketts",     x: 412, y:  82, sector: 1 },
  { id: "chapel",     label: "Chapel",       x: 348, y:  92, sector: 1 },
  { id: "stowe",      label: "Stowe",        x: 122, y: 128, sector: 2 },
  { id: "vale",       label: "Vale",         x: 105, y: 188, sector: 2 },
  { id: "club",       label: "Club",         x: 130, y: 252, sector: 2 },
  { id: "abbey",      label: "Abbey",        x: 202, y: 294, sector: 2 },
  { id: "village",    label: "Village",      x: 362, y: 278, sector: 2 },
  { id: "loop",       label: "Loop",         x: 388, y: 248, sector: 2 },
  { id: "aintree",    label: "Aintree",      x: 420, y: 230, sector: 2 },
  { id: "brooklands", label: "Brooklands",   x: 450, y: 262, sector: 3 },
  { id: "luffield",   label: "Luffield",     x: 455, y: 305, sector: 3 },
  { id: "woodcote",   label: "Woodcote",     x: 472, y: 262, sector: 3 },
] as const;

// Approximate SVG paths for each sector (split of the full circuit)
// Sector 1: S/F → Copse → Maggotts → Becketts → Chapel → Hangar start
const S1 = `M 472 178 C 482 162, 498 148, 510 138
            C 522 128, 528 112, 518 100
            C 508 88, 494 90, 482 102
            C 470 114, 450 96, 430 88
            C 414 80, 398 78, 380 84
            C 362 90, 350 88, 338 92`;

// Sector 2: Hangar → Stowe → Vale → Club → Abbey → Farm → Village → Loop → Aintree
const S2 = `M 338 92 C 280 88, 210 86, 172 94
            C 144 100, 124 112, 118 128
            C 112 144, 105 162, 105 188
            C 105 212, 112 232, 130 252
            C 148 272, 174 286, 202 294
            C 230 302, 272 306, 308 300
            C 336 294, 352 284, 362 278
            C 374 268, 382 256, 388 248
            C 396 238, 410 226, 420 230`;

// Sector 3: Aintree → Brooklands → Luffield → Woodcote → S/F
const S3 = `M 420 230 C 434 234, 444 246, 450 262
            C 454 278, 456 294, 455 305
            C 454 318, 460 325, 470 318
            C 480 310, 484 295, 484 278
            C 484 258, 482 236, 478 214
            C 476 198, 474 190, 472 178`;

const CONDITION_COLORS: Record<string, string> = {
  DRY:  "#10b981",
  DAMP: "#f59e0b",
  WET:  "#3b82f6",
};


interface CircuitMapProps {
  latest: PredictResponse | null;
  selectedCorner: string;
  onCornerSelect: (id: string) => void;
}

export default function CircuitMap({ latest, selectedCorner, onCornerSelect }: CircuitMapProps) {
  const label = latest?.label ?? null;
  const confidencePct = latest ? Math.round(latest.confidence * 100) : null;

  // Color each sector independently — use the same label color for now;
  // when multi-sector data is available each sector can diverge.
  const sectorColor = (s: number) => {
    if (!label) return s === 1 ? "#10b981" : s === 2 ? "#f59e0b" : "#3b82f6";
    return CONDITION_COLORS[label] ?? "#4b5563";
  };

  return (
    <div className="circuit-map-wrap">
      <div className="circuit-map-label">
        <span className="circuit-name">🏁 SILVERSTONE GP</span>
        {confidencePct !== null && label && (
          <span className="circuit-grip-badge" style={{ color: sectorColor(1) }}>
            {label} · {confidencePct}%
          </span>
        )}
      </div>

      <svg
        viewBox="0 0 580 380"
        className="circuit-svg"
        aria-label="Silverstone Grand Prix circuit map"
      >
        {/* --- Base tarmac (full circuit grey) --- */}
        <path
          d={`${S1} ${S2.replace("M 338 92", "L 338 92")} ${S3.replace("M 420 230", "L 420 230")}`}
          fill="none"
          stroke="#1e2d3d"
          strokeWidth="34"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* --- Sector paths coloured by condition --- */}
        {[
          { path: S1, s: 1 },
          { path: S2, s: 2 },
          { path: S3, s: 3 },
        ].map(({ path, s }) => (
          <path
            key={s}
            d={path}
            fill="none"
            stroke={sectorColor(s)}
            strokeWidth="20"
            strokeLinecap="round"
            strokeLinejoin="round"
            opacity={label ? 0.85 : 0.35}
            style={{ transition: "stroke 0.6s ease, opacity 0.4s ease" }}
          />
        ))}

        {/* --- White center-line dashes --- */}
        <path
          d={`${S1} ${S2.replace("M 338 92", "L 338 92")} ${S3.replace("M 420 230", "L 420 230")}`}
          fill="none"
          stroke="rgba(255,255,255,0.15)"
          strokeWidth="1.5"
          strokeDasharray="8 10"
          strokeLinecap="round"
        />

        {/* --- S/F line --- */}
        <line x1="461" y1="168" x2="483" y2="188" stroke="#ffffff" strokeWidth="3" strokeLinecap="round" />
        <text x="488" y="175" fill="#ffffff" fontSize="9" fontFamily="JetBrains Mono, monospace" opacity="0.7">S/F</text>

        {/* --- Sector dividers --- */}
        {/* S1/S2 divider at Chapel / Hangar entry */}
        <line x1="330" y1="86" x2="346" y2="100" stroke="rgba(255,255,255,0.35)" strokeWidth="1.5" strokeDasharray="4 3" />
        {/* S2/S3 divider at Aintree */}
        <line x1="412" y1="222" x2="430" y2="238" stroke="rgba(255,255,255,0.35)" strokeWidth="1.5" strokeDasharray="4 3" />

        {/* --- Sector labels --- */}
        <text x="425" y="112" fill={sectorColor(1)} fontSize="10" fontFamily="JetBrains Mono, monospace" fontWeight="bold" opacity="0.8">S1</text>
        <text x="155" y="228" fill={sectorColor(2)} fontSize="10" fontFamily="JetBrains Mono, monospace" fontWeight="bold" opacity="0.8">S2</text>
        <text x="462" y="340" fill={sectorColor(3)} fontSize="10" fontFamily="JetBrains Mono, monospace" fontWeight="bold" opacity="0.8">S3</text>

        {/* --- Corner nodes --- */}
        {CORNERS.filter(c => c.id !== "SF").map((c) => {
          const isSelected = c.id === selectedCorner;
          const col = sectorColor(c.sector);
          return (
            <g
              key={c.id}
              onClick={() => onCornerSelect(c.id === selectedCorner ? "ALL" : c.id)}
              style={{ cursor: "pointer" }}
            >
              {/* Outer glow ring */}
              {isSelected && (
                <circle cx={c.x} cy={c.y} r={11} fill="none" stroke={col} strokeWidth="2" opacity="0.6">
                  <animate attributeName="r" values="9;13;9" dur="1.5s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values="0.6;0.1;0.6" dur="1.5s" repeatCount="indefinite" />
                </circle>
              )}
              <circle
                cx={c.x}
                cy={c.y}
                r={isSelected ? 6 : 4}
                fill={isSelected ? col : "#1e2d3d"}
                stroke={col}
                strokeWidth="1.8"
                style={{ transition: "r 0.2s, fill 0.2s" }}
              />
              {/* Corner label — only show for important ones or selected */}
              {(isSelected || ["copse","becketts","stowe","village","brooklands"].includes(c.id)) && (
                <text
                  x={c.x + (c.x > 300 ? 8 : -8)}
                  y={c.y + 4}
                  fill={col}
                  fontSize="8"
                  fontFamily="JetBrains Mono, monospace"
                  textAnchor={c.x > 300 ? "start" : "end"}
                  opacity="0.9"
                >
                  {c.label}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {/* Sector status strip */}
      <div className="sector-strip">
        {[1, 2, 3].map((s) => (
          <div key={s} className="sector-chip" style={{ borderColor: sectorColor(s) }}>
            <span className="sector-chip-label" style={{ color: sectorColor(s) }}>S{s}</span>
            <span className="sector-chip-cond">
              {label ?? "—"}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
