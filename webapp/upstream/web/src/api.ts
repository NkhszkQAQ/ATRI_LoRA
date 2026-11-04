const API_BASE = import.meta.env.VITE_API_BASE || '/api';

export function getToken() {
  return localStorage.getItem('atri_token') || '';
}

export function setToken(t: string) {
  localStorage.setItem('atri_token', t);
}

export function clearToken() {
  localStorage.removeItem('atri_token');
}

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers || {});
  if (!headers.has('Content-Type') && init.body) headers.set('Content-Type', 'application/json');
  const token = getToken();
  if (token) headers.set('Authorization', `Bearer ${token}`);
  const res = await fetch(`${API_BASE}${path.replace(/^\/api/, '')}`, { ...init, headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail || JSON.stringify(j);
    } catch {}
    throw new Error(typeof detail === 'string' ? detail : '请求失败');
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export const api = {
  health: () => req<{ ok: boolean }>('/health'),
  sendCode: (email: string) => req('/auth/send-code', { method: 'POST', body: JSON.stringify({ email }) }),
  register: (body: object) => req<{ access_token: string }>('/auth/register', { method: 'POST', body: JSON.stringify(body) }),
  login: (body: object) => req<{ access_token: string }>('/auth/login', { method: 'POST', body: JSON.stringify(body) }),
  me: () => req<any>('/auth/me'),
  getMemory: () => req<{ memory_note: string }>('/chat/memory'),
  setMemory: (memory_note: string) =>
    req<{ memory_note: string }>('/chat/memory', { method: 'PUT', body: JSON.stringify({ memory_note }) }),
  conversations: () => req<any[]>('/chat/conversations'),
  createConversation: () => req<any>('/chat/conversations', { method: 'POST' }),
  messages: (id: number) => req<any[]>(`/chat/conversations/${id}/messages`),
  adminUsers: (q?: string) => req<any[]>(`/admin/users${q ? `?q=${encodeURIComponent(q)}` : ''}`),
  adminPatchUser: (id: number, body: object) => req(`/admin/users/${id}`, { method: 'PATCH', body: JSON.stringify(body) }),
  adminConversations: (params: URLSearchParams) => req<any[]>(`/admin/conversations?${params}`),
  adminMessages: (id: number) => req<any[]>(`/admin/conversations/${id}/messages`),
  adminSafety: (params: URLSearchParams) => req<any[]>(`/admin/safety-events?${params}`),
  adminSafetyDetail: (id: number) => req<any>(`/admin/safety-events/${id}`),
  adminAudit: () => req<any[]>('/admin/audit-logs'),
  adminModels: () => req<any>('/admin/models'),
  adminSetModel: (model: string) => req('/admin/models', { method: 'PUT', body: JSON.stringify({ model }) }),
  adminSmtp: () => req<any>('/admin/smtp'),
  adminSetSmtp: (body: object) => req('/admin/smtp', { method: 'PUT', body: JSON.stringify(body) }),
};

export async function streamChat(
  content: string,
  conversationId: number | null,
  enableThinking: boolean,
  onChunk: (data: any) => void,
) {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  const token = getToken();
  if (token) headers.Authorization = `Bearer ${token}`;
  const res = await fetch(`${API_BASE}/chat/send`.replace(/\/api\/api/, '/api'), {
    method: 'POST',
    headers,
    body: JSON.stringify({
      content,
      conversation_id: conversationId,
      enable_thinking: enableThinking,
    }),
  });
  if (!res.ok || !res.body) throw new Error('发送失败');
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const parts = buf.split('\n\n');
    buf = parts.pop() || '';
    for (const part of parts) {
      const line = part.trim();
      if (!line.startsWith('data:')) continue;
      const json = line.slice(5).trim();
      onChunk(JSON.parse(json));
    }
  }
}
