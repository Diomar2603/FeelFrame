from abc import ABC, abstractmethod
from typing import Optional

from app.models.StorageSource import StorageSource


class IStorageService(ABC):
    """
    Contrato de armazenamento de arquivos do FeelFrame.

    Implementações disponíveis:
        FirebaseStorageService — armazenamento em nuvem (produção)
        LocalStorageService    — disco local (desenvolvimento / testes)

    Para trocar de implementação, basta alterar a variável de ambiente
    STORAGE_BACKEND (firebase | local) e reiniciar o servidor.
    """

    # ------------------------------------------------------------------
    # Identificação da origem
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def source(self) -> StorageSource:
        """Identifica qual backend de armazenamento esta instância representa."""

    # ------------------------------------------------------------------
    # Métodos SÍNCRONOS (use em workers de thread, nunca em corrotinas)
    # ------------------------------------------------------------------

    @abstractmethod
    def upload_video(self, file_path: str, public_id: Optional[str] = None,
                     folder: str = "videos") -> dict:
        """
        Faz upload de um arquivo de vídeo.
        Returns: {"public_id": str, "secure_url": str}
        """

    @abstractmethod
    def upload_pdf(self, file_path: str, public_id: Optional[str] = None,
                   folder: str = "relatorios") -> dict:
        """
        Faz upload de um PDF a partir de um caminho local.
        Returns: {"public_id": str, "secure_url": str}
        """

    @abstractmethod
    def upload_pdf_from_bytes(self, pdf_bytes: bytes, filename: str,
                              folder: str = "relatorios") -> dict:
        """
        Faz upload de um PDF diretamente de bytes (sem salvar em disco).
        Returns: {"public_id": str, "secure_url": str}
        """

    @abstractmethod
    def upload_image_from_bytes(self, image_bytes: bytes, filename: str,
                                folder: str = "images") -> dict:
        """
        Faz upload de uma imagem (JPEG/PNG) diretamente de bytes.
        Returns: {"public_id": str, "secure_url": str}
        """

    @abstractmethod
    def delete_file(self, public_id: str) -> bool:
        """
        Remove um arquivo pelo caminho/identificador.
        Returns: True se removido com sucesso, False caso contrário.
        """

    # ------------------------------------------------------------------
    # Wrappers ASSÍNCRONOS (use em corrotinas / endpoints FastAPI)
    # ------------------------------------------------------------------

    @abstractmethod
    async def upload_video_async(self, file_path: str,
                                 public_id: Optional[str] = None,
                                 folder: str = "videos") -> dict:
        """Versão não bloqueante de upload_video."""

    @abstractmethod
    async def upload_pdf_async(self, file_path: str,
                               public_id: Optional[str] = None,
                               folder: str = "relatorios") -> dict:
        """Versão não bloqueante de upload_pdf."""

    @abstractmethod
    async def upload_pdf_from_bytes_async(self, pdf_bytes: bytes,
                                          filename: str,
                                          folder: str = "relatorios") -> dict:
        """Versão não bloqueante de upload_pdf_from_bytes."""

    @abstractmethod
    async def upload_image_from_bytes_async(self, image_bytes: bytes,
                                            filename: str,
                                            folder: str = "images") -> dict:
        """Versão não bloqueante de upload_image_from_bytes."""

    @abstractmethod
    async def delete_file_async(self, public_id: str) -> bool:
        """Versão não bloqueante de delete_file."""
