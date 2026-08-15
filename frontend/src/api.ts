const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export type Probabilities = { DRY: number; DAMP: number; WET: number };

export interface EvidenceTrail {
  trust: "HIGH" | "MODERATE" | "LOW";
  reasons: string[];
  concerns: string[];
}

export interface PredictResponse {
  session_id: string;
  observation_id: number;
  label: "DRY" | "DAMP" | "WET";
  probabilities: Probabilities;
  confidence: number;
  trend: "WETTING" | "DRYING" | "STABLE";
  suggestion: string;
  evidence: EvidenceTrail;
  sector_id?: string;
  created_at: string;
  model_source: "trained-onnx" | "fallback-heuristic";
}

export interface HistoryObservation {
  id: number;
  created_at: string;
  label: PredictResponse["label"];
  probabilities: Probabilities;
  confidence: number;
  trend: PredictResponse["trend"];
  suggestion: string;
}

export interface HistoryResponse {
  session_id: string;
  count: number;
  observations: HistoryObservation[];
}

export async function predict(
  file: File,
  sessionId: string,
  sectorId: string = "ALL"
): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("session_id", sessionId);
  form.append("sector_id", sectorId);
  const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Prediction failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export interface PredictVideoResponse {
  session_id: string;
  frames_extracted: number;
  frames_classified: number;
  observations: PredictResponse[];
}

export async function predictVideo(
  file: File,
  sessionId: string,
  sectorId: string = "ALL"
): Promise<PredictVideoResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("session_id", sessionId);
  form.append("sector_id", sectorId);
  const res = await fetch(`${API_BASE}/predict/video`, { method: "POST", body: form });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`Video processing failed (${res.status}): ${detail}`);
  }
  return res.json();
}

export async function getHistory(sessionId: string): Promise<HistoryResponse> {
  const res = await fetch(`${API_BASE}/history/${sessionId}`);
  if (!res.ok) throw new Error(`History fetch failed (${res.status})`);
  return res.json();
}

export async function getHealth(): Promise<{ status: string; model_loaded: boolean; model_source: string }> {
  const res = await fetch(`${API_BASE}/health`);
  return res.json();
}
