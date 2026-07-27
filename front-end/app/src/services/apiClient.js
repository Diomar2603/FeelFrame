import { API_BASE_URL } from '../config/api';

const TOKEN_KEY = 'feelframe_token';
export const AUTH_EXPIRED_EVENT = 'feelframe:auth-expired';

export function authHeader() {
  const token = localStorage.getItem(TOKEN_KEY);
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/**
 * fetch autenticado central: injeta o token e, se o backend responder 401
 * (token ausente/expirado/inválido), dispara um evento global para que o
 * AuthContext encerre a sessão e redirecione para a tela de login.
 */
export async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers: { ...authHeader(), ...(options.headers || {}) },
  });

  if (res.status === 401) {
    window.dispatchEvent(new Event(AUTH_EXPIRED_EVENT));
    throw new Error('Sessão expirada. Faça login novamente.');
  }

  return res;
}
