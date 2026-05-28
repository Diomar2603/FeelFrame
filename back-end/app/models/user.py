from pydantic import BaseModel
from typing import Optional


# ── Requests ─────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class GoogleAuthRequest(BaseModel):
    """Payload enviado pelo frontend após o Google One-Tap / OAuth flow."""
    credential: str  # ID Token JWT retornado pelo @react-oauth/google


# ── Responses ─────────────────────────────────────────────────────────────────

class UserResponse(BaseModel):
    user_id: str
    name: str
    email: str


class AuthResponse(BaseModel):
    token: str
    user: UserResponse
