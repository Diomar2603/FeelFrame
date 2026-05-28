from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.services.authService import AuthService

_security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> dict:
    """
    Dependência FastAPI que extrai e valida o JWT do header Authorization.
    Injete com `Depends(get_current_user)` em qualquer endpoint protegido.

    Retorna: {"user_id": str, "email": str}
    """
    payload = AuthService.decode_token(credentials.credentials)
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido ou expirado.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return payload
