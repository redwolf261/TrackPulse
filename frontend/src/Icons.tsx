// Minimal, single-color SVG icon set replacing pictorial emoji throughout the
// UI, per the enterprise design critique: emoji read as informal/AI-generated,
// crisp vector glyphs read as professional telemetry tooling. Every icon is a
// simple stroked outline (no fill, no color baked in) so it inherits
// currentColor and sits correctly in both light-accent and muted contexts.

interface IconProps {
  size?: number;
  className?: string;
}

const base = (size: number) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.75,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
});

export function IconFlag({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M5 3v18" />
      <path d="M5 4h6l-1.5 3.5L11 11H5" />
      <path d="M11 4h6l-1.5 3.5L17 11h-6" />
    </svg>
  );
}

export function IconCamera({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M4 8h3l1.5-2h7L17 8h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1Z" />
      <circle cx="12" cy="13" r="3.5" />
    </svg>
  );
}

export function IconChart({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M4 20V10" />
      <path d="M10 20V4" />
      <path d="M16 20v-7" />
      <path d="M22 20V13" />
      <path d="M2 20h20" />
    </svg>
  );
}

export function IconScan({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M4 8V5a1 1 0 0 1 1-1h3" />
      <path d="M16 4h3a1 1 0 0 1 1 1v3" />
      <path d="M20 16v3a1 1 0 0 1-1 1h-3" />
      <path d="M8 20H5a1 1 0 0 1-1-1v-3" />
      <path d="M12 8v8" />
      <path d="M8.5 10.5 12 8l3.5 2.5" />
      <path d="M8.5 13.5 12 16l3.5-2.5" />
    </svg>
  );
}

export function IconBolt({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M13 2 4 14h6l-1 8 9-12h-6l1-8Z" />
    </svg>
  );
}

export function IconSun({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6l1.4 1.4M17 17l1.4 1.4M18.4 5.6 17 7M7 17l-1.4 1.4" />
    </svg>
  );
}

export function IconCloudDrizzle({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M7 15a4 4 0 1 1 1.2-7.8A5 5 0 0 1 18 9a3.5 3.5 0 0 1-.5 6H7Z" />
      <path d="M9 19v1M12 19v2M15 19v1" />
    </svg>
  );
}

export function IconRain({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M7 14a4 4 0 1 1 1.2-7.8A5 5 0 0 1 18 8a3.5 3.5 0 0 1-.5 6H7Z" />
      <path d="M8 18v2M12 18v3M16 18v2" />
    </svg>
  );
}

export function IconAlert({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M12 3 2 20h20L12 3Z" />
      <path d="M12 10v4M12 17h.01" />
    </svg>
  );
}

export function IconCheck({ size = 16, className }: IconProps) {
  return (
    <svg {...base(size)} className={className}>
      <path d="M4 12l5 5L20 6" />
    </svg>
  );
}
