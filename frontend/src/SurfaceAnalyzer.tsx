import { useRef, useEffect, useState } from "react";
import { IconScan } from "./Icons";

type Mode = "raw" | "sheen" | "texture";

interface SurfaceAnalyzerProps {
  previewUrl: string | null;
  isVideo: boolean;
}

export default function SurfaceAnalyzer({ previewUrl, isVideo }: SurfaceAnalyzerProps) {
  const [mode, setMode] = useState<Mode>("raw");
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!previewUrl || isVideo) return;
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => applyFilter(img, mode);
    img.src = previewUrl;
  }, [previewUrl, mode, isVideo]);

  const applyFilter = (img: HTMLImageElement, m: Mode) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    canvas.width  = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);

    if (m === "raw") return; // nothing to process

    const id = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const d  = id.data;

    for (let i = 0; i < d.length; i += 4) {
      const r = d[i], g = d[i + 1], b = d[i + 2];

      if (m === "sheen") {
        // Highlight specular moisture: pixels where blue+green >> red (water sheen)
        const moisture = (b * 0.55 + g * 0.35 - r * 0.30) / 255;
        const lum      = (r * 0.299 + g * 0.587 + b * 0.114) / 255;
        if (moisture > 0.18 && lum > 0.28) {
          // Tint towards electric cyan
          d[i]     = Math.min(255, r * 0.3);
          d[i + 1] = Math.min(255, g * 0.5 + 80);
          d[i + 2] = Math.min(255, b * 0.7 + 140);
        } else {
          // Desaturate everything else for contrast
          const grey = r * 0.299 + g * 0.587 + b * 0.114;
          d[i] = d[i + 1] = d[i + 2] = grey;
        }
      }

      if (m === "texture") {
        // Highlight tarmac micro-texture: high-frequency luminance variance → bright
        // Simplified: sharpen edges by boosting mid-tone contrast
        const lum = (r * 0.299 + g * 0.587 + b * 0.114) / 255;
        // Dry grooves tend to be mid-dark with high texture; wet asphalt is smoother
        const grooveScore = lum > 0.15 && lum < 0.65 ? 1.0 : 0.25;
        const grey = Math.min(255, (r * 0.299 + g * 0.587 + b * 0.114) * grooveScore * 2.2);
        d[i]     = Math.min(255, grey + (lum > 0.5 ? 20 : 0));   // slight warm highlight
        d[i + 1] = Math.min(255, grey * 0.88);
        d[i + 2] = Math.min(255, grey * 0.72);
      }
    }
    ctx.putImageData(id, 0, 0);
  };

  if (!previewUrl) return null;

  const modes: { id: Mode; label: string }[] = [
    { id: "raw",     label: "RGB" },
    { id: "sheen",   label: "Sheen Map" },
    { id: "texture", label: "Groove Texture" },
  ];

  return (
    <div className="surface-analyzer">
      <div className="surface-header">
        <span className="surface-title"><IconScan size={13} /> SURFACE ANALYZER</span>
        <div className="surface-modes">
          {modes.map((m) => (
            <button
              key={m.id}
              id={`surface-mode-${m.id}`}
              className={`surface-mode-btn ${mode === m.id ? "active" : ""}`}
              onClick={() => setMode(m.id)}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {isVideo ? (
        <div className="surface-video-note">
          Surface analyzer available for image uploads.
        </div>
      ) : (
        <div className="surface-canvas-wrap">
          <canvas ref={canvasRef} className="surface-canvas" />
          <div className="surface-mode-label">
            {mode === "raw"     && "Raw RGB — unprocessed image"}
            {mode === "sheen"   && "Moisture / Sheen Map — cyan highlights show water reflectance"}
            {mode === "texture" && "Groove Texture — bright areas indicate dry tarmac micro-texture"}
          </div>
        </div>
      )}
    </div>
  );
}
