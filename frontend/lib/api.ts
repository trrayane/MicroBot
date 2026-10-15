const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface Source { doc_id: string; title: string; snippet: string }
export interface ChatResponse {
  answer: string; grounded: boolean; sources: Source[];
  conversation_id: string; llm_calls: number;
  prompt_tokens: number; completion_tokens: number; latency_ms: number;
}
export interface MeResponse { role: "admin" | "member" | "anonymous"; display_name: string | null; avatar: string | null; user_id: string | null }
export interface DocumentOut { doc_id: string; title: string; source: string; visibility: string; chunks: number; uploaded_by: string | null; created_at: number | null }
export interface DocumentStatus { doc_id: string; title: string; status: string; chunks: number; error: string | null; elapsed_ms: number }
export interface FeedbackEntry { feedback_id: string; conversation_id: string | null; vote: string; created_at: number }

function authHeaders(): Record<string, string> {
  if (typeof window === "undefined") return {};
  const token = sessionStorage.getItem("mc_token");
  if (token) return { Authorization: `Bearer ${token}` };
  const key = sessionStorage.getItem("mc_admin_key");
  if (key) return { "X-Admin-Key": key };
  return {};
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}/api/v1${path}`, { ...init, headers: { ...authHeaders(), ...init?.headers } });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body?.error?.message || `HTTP ${res.status}`);
  }
  return res.json();
}

export const chat = (message: string, conversationId?: string) =>
  api<ChatResponse>("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message, conversation_id: conversationId || null }) });

export const me = () => api<MeResponse>("/auth/me");
export const loginWithKey = (key: string) =>
  api<{ ok: boolean }>("/auth/login", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ key }) });
export const listDocuments = () => api<{ documents: DocumentOut[] }>("/documents");
export const documentStatus = (docId: string) => api<DocumentStatus>(`/documents/${docId}/status`);
export const deleteDocument = (docId: string) => api<{ doc_id: string; chunks_deleted: number }>(`/documents/${docId}`, { method: "DELETE" });
export const uploadDocument = (formData: FormData) => api<{ doc_id: string; title: string; status: string }>("/documents", { method: "POST", body: formData });
export const feedback = (conversationId: string | null, vote: "up" | "down") =>
  api("/feedback", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ conversation_id: conversationId, vote }) });
export const listFeedback = (limit = 100) => api<{ feedback: FeedbackEntry[] }>(`/feedback?limit=${limit}`);
export const stats = () => api<{ collection: string; documents: number; chunks: number }>("/stats");
export const discordLoginUrl = () => `${API}/api/v1/auth/discord/login`;
