import os

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException, status
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

from app.models.user import AuthResponse, GoogleAuthRequest, LoginRequest, RegisterRequest
from app.services.authService import AuthService
from app.services.userService import UserService
from app.utils.DatabaseConfig import DatabaseConfig
from app.utils.auth_deps import get_current_user

load_dotenv()

router = APIRouter(prefix="/autenticacao", tags=["Autenticacao"])

_GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

# Singleton de serviço (conexão MongoDB reutilizada)
_user_service = UserService(DatabaseConfig())


# ---------------------------------------------------------------------------
# Registro
# ---------------------------------------------------------------------------

@router.post("/registrar", response_model=AuthResponse, status_code=201)
def register(req: RegisterRequest):
    """Cria nova conta com email e senha."""
    if not req.name or not req.email or not req.password:
        raise HTTPException(status_code=400, detail="Nome, email e senha são obrigatórios.")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="A senha deve ter pelo menos 6 caracteres.")
    try:
        user = _user_service.create_user(req.name, req.email, req.password)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    token = AuthService.create_token(user["user_id"], user["email"])
    return {"token": token, "user": user}


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

@router.post("/entrar", response_model=AuthResponse)
def login(req: LoginRequest):
    """Autentica com email e senha, retorna JWT."""
    user = _user_service.authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou senha incorretos.",
        )
    token = AuthService.create_token(user["user_id"], user["email"])
    return {"token": token, "user": user}


# ---------------------------------------------------------------------------
# Google OAuth
# ---------------------------------------------------------------------------

@router.post("/google", response_model=AuthResponse)
def google_auth(req: GoogleAuthRequest):
    """
    Recebe o ID Token do Google (gerado pelo @react-oauth/google no frontend),
    verifica a assinatura e cria/recupera a conta local.
    """
    if not _GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="GOOGLE_CLIENT_ID não configurado no servidor.",
        )
    try:
        info = id_token.verify_oauth2_token(
            req.credential,
            google_requests.Request(),
            _GOOGLE_CLIENT_ID,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token Google inválido: {exc}",
        )

    google_id = info["sub"]
    email     = info.get("email", "")
    name      = info.get("name") or info.get("given_name") or email

    user  = _user_service.get_or_create_google_user(google_id, email, name)
    token = AuthService.create_token(user["user_id"], user["email"])
    return {"token": token, "user": user}


# ---------------------------------------------------------------------------
# Perfil do usuário logado
# ---------------------------------------------------------------------------

@router.get("/perfil")
def get_me(current_user: dict = Depends(get_current_user)):
    """Retorna os dados do usuário autenticado (requer Bearer token)."""
    user = _user_service.get_user_by_id(current_user["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="Usuário não encontrado.")
    return user
