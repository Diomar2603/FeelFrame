import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { GoogleLogin } from '@react-oauth/google';
import { useAuth } from '../../context/AuthContext';
import { authService } from '../../services/authService';
import './LoginPage.css';

export default function LoginPage() {
  const { login } = useAuth();
  const navigate   = useNavigate();

  const [mode,     setMode]     = useState('login'); // 'login' | 'register'
  const [name,     setName]     = useState('');
  const [email,    setEmail]    = useState('');
  const [password, setPassword] = useState('');
  const [error,    setError]    = useState(null);
  const [loading,  setLoading]  = useState(false);

  const handleSuccess = ({ token, user }) => {
    login(token, user);
    navigate('/', { replace: true });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      let result;
      if (mode === 'register') {
        result = await authService.register(name, email, password);
      } else {
        result = await authService.login(email, password);
      }
      handleSuccess(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSuccess = async (credentialResponse) => {
    setError(null);
    setLoading(true);
    try {
      const result = await authService.googleAuth(credentialResponse.credential);
      handleSuccess(result);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-card">

        <div className="login-brand">
          <span className="login-logo">FF</span>
          <h1>FeelFrame</h1>
          <p>Análise emocional de vídeos com IA</p>
        </div>

        <div className="login-tabs">
          <button
            className={mode === 'login' ? 'tab active' : 'tab'}
            onClick={() => { setMode('login'); setError(null); }}
          >
            Entrar
          </button>
          <button
            className={mode === 'register' ? 'tab active' : 'tab'}
            onClick={() => { setMode('register'); setError(null); }}
          >
            Criar conta
          </button>
        </div>

        <form onSubmit={handleSubmit} className="login-form">
          {mode === 'register' && (
            <div className="field-group">
              <label>Nome</label>
              <input
                type="text"
                placeholder="Seu nome completo"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                autoComplete="name"
              />
            </div>
          )}

          <div className="field-group">
            <label>Email</label>
            <input
              type="email"
              placeholder="seu@email.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
            />
          </div>

          <div className="field-group">
            <label>Senha</label>
            <input
              type="password"
              placeholder={mode === 'register' ? 'Mínimo 6 caracteres' : '••••••••'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete={mode === 'register' ? 'new-password' : 'current-password'}
            />
          </div>

          {error && <p className="login-error">{error}</p>}

          <button type="submit" className="btn-submit" disabled={loading}>
            {loading ? 'Aguarde...' : mode === 'login' ? 'Entrar' : 'Criar conta'}
          </button>
        </form>

        <div className="login-divider"><span>ou</span></div>

        <div className="google-btn-wrapper">
          <GoogleLogin
            onSuccess={handleGoogleSuccess}
            onError={() => setError('Falha no login com Google.')}
            useOneTap={false}
            text={mode === 'register' ? 'signup_with' : 'signin_with'}
            shape="rectangular"
            width="100%"
          />
        </div>

      </div>
    </div>
  );
}
