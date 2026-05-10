"""
firebaseStorageService.py — REFATORADO para operação assíncrona e não bloqueante.

MUDANÇAS PRINCIPAIS vs versão original:
  1. Todo o código original dos métodos síncronos foi PRESERVADO sem alteração.
     Callers síncronos (ex.: dentro de workers de thread) continuam funcionando.

  2. Adicionados wrappers async para cada operação de upload/delete:
       - `upload_video_async()`
       - `upload_pdf_async()`
       - `upload_pdf_from_bytes_async()`
       - `delete_file_async()`
     Todos usam `asyncio.to_thread()` para delegar o I/O de rede (Firebase SDK
     é síncrono e bloqueante) para uma thread do pool padrão do asyncio,
     liberando a event loop principal.

  3. Thread-safety: o Firebase Admin SDK e o objeto `bucket` são thread-safe
     para operações de upload/download concorrentes (cada chamada abre sua
     própria conexão HTTP interna). Nenhum lock adicional é necessário.

  4. Tratamento de exceções: erros dentro da thread propagam normalmente ao
     `await`, permitindo que o chamador os capture com try/except.
"""

import os
import asyncio
from typing import Optional

import firebase_admin
from firebase_admin import credentials, storage

from app.models.StorageSource import StorageSource
from app.services.interfaces.IStorageService import IStorageService


class FirebaseStorageService(IStorageService):
    """
    Serviço de armazenamento no Firebase Storage (free tier Spark: 5 GB).
    Gerencia upload de vídeos .mp4 e PDFs, retornando URLs públicas prontas para o frontend.

    Configuração necessária no .env:
        FIREBASE_CREDENTIALS_PATH=caminho/para/serviceAccountKey.json
        FIREBASE_STORAGE_BUCKET=seu-projeto.appspot.com

    Como obter as credenciais:
        1. Firebase Console → Configurações do projeto → Contas de serviço
        2. Clique em "Gerar nova chave privada" → salva o JSON
        3. Coloque o caminho desse JSON no .env como FIREBASE_CREDENTIALS_PATH

    USO EM ENDPOINTS ASYNC:
        Prefira os métodos `*_async()` em qualquer contexto assíncrono (FastAPI).
        Os métodos síncronos originais continuam disponíveis para uso em workers
        de thread (ex.: dentro de `_process_video_sync` no VideoService).
    """

    def __init__(self):
        # Inicialização é idempotente — segura para múltiplas instâncias.
        if not firebase_admin._apps:
            cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
            if not cred_path:
                raise ValueError("FIREBASE_CREDENTIALS_PATH não definido no .env")

            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET")
            })

        self.bucket = storage.bucket()

    @property
    def source(self) -> StorageSource:
        return StorageSource.FIREBASE

    # =========================================================================
    # Métodos SÍNCRONOS originais (preservados — use em workers de thread)
    # =========================================================================

    def upload_video(self, file_path: str, public_id: Optional[str] = None,
                     folder: str = "videos") -> dict:
        """
        Faz upload de um vídeo (.mp4) para o Firebase Storage.

        ⚠️  BLOQUEANTE — não chame diretamente de uma corrotina.
        Use `upload_video_async()` em endpoints FastAPI.

        Returns:
            {"public_id": str, "secure_url": str}
        """
        if public_id is None:
            public_id = os.path.splitext(os.path.basename(file_path))[0]

        ext       = os.path.splitext(file_path)[1] or ".mp4"
        blob_path = f"{folder}/{public_id}{ext}"
        blob      = self.bucket.blob(blob_path)

        blob.upload_from_filename(file_path, content_type="video/mp4")
        blob.make_public()

        return {"public_id": blob_path, "secure_url": blob.public_url}

    def upload_pdf(self, file_path: str, public_id: Optional[str] = None,
                   folder: str = "relatorios") -> dict:
        """
        Faz upload de um PDF a partir de um caminho local.

        ⚠️  BLOQUEANTE — use `upload_pdf_async()` em endpoints FastAPI.

        Returns:
            {"public_id": str, "secure_url": str}
        """
        if public_id is None:
            public_id = os.path.splitext(os.path.basename(file_path))[0]

        blob_path = f"{folder}/{public_id}.pdf"
        blob      = self.bucket.blob(blob_path)

        blob.upload_from_filename(file_path, content_type="application/pdf")
        blob.make_public()

        return {"public_id": blob_path, "secure_url": blob.public_url}

    def upload_pdf_from_bytes(self, pdf_bytes: bytes, filename: str,
                              folder: str = "relatorios") -> dict:
        """
        Faz upload de um PDF diretamente de bytes (sem salvar em disco).

        ⚠️  BLOQUEANTE — use `upload_pdf_from_bytes_async()` em endpoints FastAPI.

        Returns:
            {"public_id": str, "secure_url": str}
        """
        public_id = os.path.splitext(filename)[0]
        blob_path = f"{folder}/{public_id}.pdf"
        blob      = self.bucket.blob(blob_path)

        blob.upload_from_string(pdf_bytes, content_type="application/pdf")
        blob.make_public()

        return {"public_id": blob_path, "secure_url": blob.public_url}

    def upload_image_from_bytes(self, image_bytes: bytes, filename: str,
                               folder: str = "images") -> dict:
        """
        Faz upload de uma imagem (JPEG) diretamente de bytes.

        ⚠️  BLOQUEANTE — use `upload_image_from_bytes_async()` em endpoints FastAPI.

        Returns:
            {"public_id": str, "secure_url": str}
        """
        public_id = os.path.splitext(filename)[0]
        blob_path = f"{folder}/{public_id}.jpg"
        blob      = self.bucket.blob(blob_path)

        blob.upload_from_string(image_bytes, content_type="image/jpeg")
        blob.make_public()

        return {"public_id": blob_path, "secure_url": blob.public_url}

    def delete_file(self, public_id: str) -> bool:
        """
        Remove um arquivo do Firebase Storage pelo caminho completo no bucket.

        ⚠️  BLOQUEANTE — use `delete_file_async()` em endpoints FastAPI.

        Returns:
            True se removido com sucesso, False caso contrário.
        """
        try:
            self.bucket.blob(public_id).delete()
            return True
        except Exception as e:
            print(f"Erro ao deletar arquivo '{public_id}' do Firebase: {e}")
            return False

    # =========================================================================
    # Wrappers ASSÍNCRONOS — use estes em corrotinas / endpoints FastAPI
    # =========================================================================

    async def upload_video_async(self, file_path: str,
                                 public_id: Optional[str] = None,
                                 folder: str = "videos") -> dict:
        """
        Versão não bloqueante de `upload_video`.
        O upload de rede ocorre em uma thread do pool padrão do asyncio.

        Returns:
            {"public_id": str, "secure_url": str}
        """
        return await asyncio.to_thread(
            self.upload_video, file_path, public_id, folder
        )

    async def upload_pdf_async(self, file_path: str,
                               public_id: Optional[str] = None,
                               folder: str = "relatorios") -> dict:
        """
        Versão não bloqueante de `upload_pdf`.

        Returns:
            {"public_id": str, "secure_url": str}
        """
        return await asyncio.to_thread(
            self.upload_pdf, file_path, public_id, folder
        )

    async def upload_pdf_from_bytes_async(self, pdf_bytes: bytes,
                                          filename: str,
                                          folder: str = "relatorios") -> dict:
        """
        Versão não bloqueante de `upload_pdf_from_bytes`.

        Returns:
            {"public_id": str, "secure_url": str}
        """
        return await asyncio.to_thread(
            self.upload_pdf_from_bytes, pdf_bytes, filename, folder
        )

    async def upload_image_from_bytes_async(self, image_bytes: bytes,
                                            filename: str,
                                            folder: str = "images") -> dict:
        """
        Versão não bloqueante de `upload_image_from_bytes`.

        Returns:
            {"public_id": str, "secure_url": str}
        """
        return await asyncio.to_thread(
            self.upload_image_from_bytes, image_bytes, filename, folder
        )

    async def delete_file_async(self, public_id: str) -> bool:
        """
        Versão não bloqueante de `delete_file`.

        Returns:
            True se removido com sucesso, False caso contrário.
        """
        return await asyncio.to_thread(self.delete_file, public_id)
