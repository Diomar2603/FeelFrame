import os
import asyncio
from typing import Optional

from app.models.StorageSource import StorageSource
from app.services.interfaces.IStorageService import IStorageService


class LocalStorageService(IStorageService):
    """
    Implementação de armazenamento em disco local.

    Ideal para desenvolvimento e testes — não requer credenciais externas.
    Os arquivos são salvos em LOCAL_STORAGE_PATH (padrão: ~/FeelFrameLocalStorage).
    O campo "secure_url" retornado é o caminho absoluto no sistema de arquivos.

    Para ativar: defina STORAGE_BACKEND=local no arquivo .env.
    """

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.environ.get(
            "LOCAL_STORAGE_PATH",
            os.path.join(os.path.expanduser("~"), "FeelFrameLocalStorage"),
        )
        os.makedirs(self.base_dir, exist_ok=True)

    @property
    def source(self) -> StorageSource:
        return StorageSource.LOCAL

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    def _save_bytes(self, data: bytes, relative_path: str) -> dict:
        full_path = os.path.join(self.base_dir, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "wb") as f:
            f.write(data)
        return {"public_id": relative_path, "secure_url": full_path}

    def _read_and_save(self, source_path: str, relative_path: str) -> dict:
        with open(source_path, "rb") as f:
            return self._save_bytes(f.read(), relative_path)

    # ------------------------------------------------------------------
    # Métodos SÍNCRONOS
    # ------------------------------------------------------------------

    def upload_video(self, file_path: str, public_id: Optional[str] = None,
                     folder: str = "videos") -> dict:
        if public_id is None:
            public_id = os.path.splitext(os.path.basename(file_path))[0]
        ext = os.path.splitext(file_path)[1] or ".mp4"
        return self._read_and_save(file_path, f"{folder}/{public_id}{ext}")

    def upload_pdf(self, file_path: str, public_id: Optional[str] = None,
                   folder: str = "relatorios") -> dict:
        if public_id is None:
            public_id = os.path.splitext(os.path.basename(file_path))[0]
        return self._read_and_save(file_path, f"{folder}/{public_id}.pdf")

    def upload_pdf_from_bytes(self, pdf_bytes: bytes, filename: str,
                              folder: str = "relatorios") -> dict:
        public_id = os.path.splitext(filename)[0]
        return self._save_bytes(pdf_bytes, f"{folder}/{public_id}.pdf")

    def upload_image_from_bytes(self, image_bytes: bytes, filename: str,
                                folder: str = "images") -> dict:
        public_id = os.path.splitext(filename)[0]
        return self._save_bytes(image_bytes, f"{folder}/{public_id}.jpg")

    def delete_file(self, public_id: str) -> bool:
        full_path = os.path.join(self.base_dir, public_id)
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
            return True
        except Exception as e:
            print(f"Erro ao deletar arquivo local '{full_path}': {e}")
            return False

    # ------------------------------------------------------------------
    # Wrappers ASSÍNCRONOS
    # ------------------------------------------------------------------

    async def upload_video_async(self, file_path: str,
                                 public_id: Optional[str] = None,
                                 folder: str = "videos") -> dict:
        return await asyncio.to_thread(self.upload_video, file_path, public_id, folder)

    async def upload_pdf_async(self, file_path: str,
                               public_id: Optional[str] = None,
                               folder: str = "relatorios") -> dict:
        return await asyncio.to_thread(self.upload_pdf, file_path, public_id, folder)

    async def upload_pdf_from_bytes_async(self, pdf_bytes: bytes,
                                          filename: str,
                                          folder: str = "relatorios") -> dict:
        return await asyncio.to_thread(
            self.upload_pdf_from_bytes, pdf_bytes, filename, folder
        )

    async def upload_image_from_bytes_async(self, image_bytes: bytes,
                                            filename: str,
                                            folder: str = "images") -> dict:
        return await asyncio.to_thread(
            self.upload_image_from_bytes, image_bytes, filename, folder
        )

    async def delete_file_async(self, public_id: str) -> bool:
        return await asyncio.to_thread(self.delete_file, public_id)
