import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from jwt.exceptions import PyJWTError

_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production-use-a-long-random-string")
_ALGORITHM = "HS256"
_EXPIRE_HOURS = int(os.getenv("JWT_EXPIRE_HOURS", "168"))  # 7 dias


class AuthService:
    @staticmethod
    def create_token(user_id: str, email: str) -> str:
        expire = datetime.now(timezone.utc) + timedelta(hours=_EXPIRE_HOURS)
        payload = {"sub": user_id, "email": email, "exp": expire}
        return jwt.encode(payload, _SECRET_KEY, algorithm=_ALGORITHM)

    @staticmethod
    def decode_token(token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, _SECRET_KEY, algorithms=[_ALGORITHM])
            return {"user_id": payload["sub"], "email": payload["email"]}
        except PyJWTError:
            return None
