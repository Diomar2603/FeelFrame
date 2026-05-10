import os

from app.services.interfaces.IStorageService import IStorageService


def create_storage_service() -> IStorageService:
    """
    Instancia a implementação de armazenamento conforme STORAGE_BACKEND no .env.

    Valores suportados:
        firebase  — FirebaseStorageService (padrão, produção)
        local     — LocalStorageService (desenvolvimento / testes offline)

    Exemplo de .env para trocar para armazenamento local:
        STORAGE_BACKEND=local
        LOCAL_STORAGE_PATH=/caminho/opcional/para/pasta
    """
    backend = os.getenv("STORAGE_BACKEND", "firebase").lower()

    if backend == "local":
        from app.services.localStorageService import LocalStorageService
        return LocalStorageService()

    from app.services.firebaseStorageService import FirebaseStorageService
    return FirebaseStorageService()
