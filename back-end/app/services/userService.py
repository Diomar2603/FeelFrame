import uuid
from datetime import datetime, timezone
from typing import Optional

import bcrypt
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from app.utils.DatabaseConfig import DatabaseConfig


class UserService:
    """
    Serviço de usuários com persistência em MongoDB.

    Suporta dois fluxos de autenticação:
      1. Email + senha  →  create_user / authenticate_user
      2. Google OAuth   →  get_or_create_google_user

    Um usuário pode ter ambos os métodos vinculados ao mesmo email.
    """

    def __init__(self, db_config: DatabaseConfig):
        db: Collection = db_config.client[db_config.db_name]
        self.users: Collection = db["users"]
        # Garante índice único no email (idempotente)
        self.users.create_index("email", unique=True, sparse=True)

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(password: str) -> str:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    @staticmethod
    def _verify(plain: str, hashed: str) -> bool:
        try:
            return bcrypt.checkpw(plain.encode(), hashed.encode())
        except Exception:
            return False

    @staticmethod
    def _to_response(doc: dict) -> dict:
        return {"user_id": doc["_id"], "name": doc["name"], "email": doc["email"]}

    def _new_doc(self, name: str, email: str,
                 password_hash: Optional[str], google_id: Optional[str]) -> dict:
        return {
            "_id":           str(uuid.uuid4()),
            "name":          name,
            "email":         email,
            "password_hash": password_hash,
            "google_id":     google_id,
            "is_active":     True,
            "created_at":    datetime.now(timezone.utc).isoformat(),
            "updated_at":    datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Registro por email/senha
    # ------------------------------------------------------------------

    def create_user(self, name: str, email: str, password: str) -> dict:
        """
        Cria novo usuário com senha hasheada.
        Lança ValueError se o email já existir.
        """
        doc = self._new_doc(name, email.lower(), self._hash(password), None)
        try:
            self.users.insert_one(doc)
        except DuplicateKeyError:
            raise ValueError("Este email já está cadastrado.")
        return self._to_response(doc)

    # ------------------------------------------------------------------
    # Login por email/senha
    # ------------------------------------------------------------------

    def authenticate_user(self, email: str, password: str) -> Optional[dict]:
        """
        Valida credenciais. Retorna dados do usuário ou None se inválido.
        """
        user = self.users.find_one({"email": email.lower()})
        if not user or not user.get("password_hash"):
            return None
        if not self._verify(password, user["password_hash"]):
            return None
        return self._to_response(user)

    # ------------------------------------------------------------------
    # Login / Cadastro via Google OAuth
    # ------------------------------------------------------------------

    def get_or_create_google_user(self, google_id: str,
                                   email: str, name: str) -> dict:
        """
        Encontra usuário pelo google_id, ou pelo email (vincula o google_id
        a uma conta já existente), ou cria um novo registro.
        """
        # 1. Já existe conta com este google_id
        user = self.users.find_one({"google_id": google_id})
        if user:
            return self._to_response(user)

        # 2. Existe conta com o mesmo email → vincula o google_id
        existing = self.users.find_one({"email": email.lower()})
        if existing:
            self.users.update_one(
                {"_id": existing["_id"]},
                {"$set": {"google_id": google_id,
                           "updated_at": datetime.now(timezone.utc).isoformat()}},
            )
            return self._to_response(existing)

        # 3. Novo usuário via Google
        doc = self._new_doc(name, email.lower(), None, google_id)
        self.users.insert_one(doc)
        return self._to_response(doc)

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        user = self.users.find_one({"_id": user_id})
        return self._to_response(user) if user else None
