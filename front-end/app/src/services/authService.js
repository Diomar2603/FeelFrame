import { API_BASE_URL } from '../config/api';

async function _post(path, body) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || 'Erro desconhecido');
  return data;
}

export const authService = {
  register: (name, email, password) =>
    _post('/auth/register', { name, email, password }),

  login: (email, password) =>
    _post('/auth/login', { email, password }),

  googleAuth: (credential) =>
    _post('/auth/google', { credential }),

  getMe: (token) =>
    fetch(`${API_BASE_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${token}` },
    }).then((r) => r.json()),
};
