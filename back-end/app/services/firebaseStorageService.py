import os
import uuid
from typing import Optional

import firebase_admin
from firebase_admin import credentials, storage

class FirebaseStorageService:
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
    """

    def __init__(self):
        if not firebase_admin._apps:
            cred_path = os.environ.get("FIREBASE_CREDENTIALS_PATH")
            if not cred_path:
                raise ValueError("FIREBASE_CREDENTIALS_PATH não definido no .env")

            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred, {
                "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET")
            })

        self.bucket = storage.bucket()

    # -------------------------------------------------------------------------
    # Vídeos
    # -------------------------------------------------------------------------

    def upload_video(self, file_path: str, public_id: Optional[str] = None, folder: str = "videos") -> dict:
        """
        Faz upload de um vídeo (.mp4) para o Firebase Storage.

        Args:
            file_path:  Caminho local do arquivo.
            public_id:  Nome base do arquivo no Storage (sem extensão). Se omitido, usa o nome do arquivo.
            folder:     Pasta de destino no bucket.

        Returns:
            {
                "public_id": str,   # Caminho completo no bucket
                "secure_url": str,  # URL pública HTTPS para uso no frontend
            }
        """
        if public_id is None:
            public_id = os.path.splitext(os.path.basename(file_path))[0]

        ext = os.path.splitext(file_path)[1] or ".mp4"
        blob_path = f"{folder}/{public_id}{ext}"
        blob = self.bucket.blob(blob_path)

        blob.upload_from_filename(file_path, content_type="video/mp4")
        blob.make_public()

        return {
            "public_id": blob_path,
            "secure_url": blob.public_url,
        }

    # -------------------------------------------------------------------------
    # PDFs — upload por caminho
    # -------------------------------------------------------------------------

    def upload_pdf(self, file_path: str, public_id: Optional[str] = None, folder: str = "relatorios") -> dict:
        """
        Faz upload de um PDF para o Firebase Storage a partir de um caminho local.

        Returns:
            {
                "public_id": str,
                "secure_url": str,
            }
        """
        if public_id is None:
            public_id = os.path.splitext(os.path.basename(file_path))[0]

        blob_path = f"{folder}/{public_id}.pdf"
        blob = self.bucket.blob(blob_path)

        blob.upload_from_filename(file_path, content_type="application/pdf")
        blob.make_public()

        return {
            "public_id": blob_path,
            "secure_url": blob.public_url,
        }

    # -------------------------------------------------------------------------
    # PDFs — upload direto de bytes (sem salvar em disco)
    # -------------------------------------------------------------------------

    def upload_pdf_from_bytes(self, pdf_bytes: bytes, filename: str, folder: str = "relatorios") -> dict:
        """
        Faz upload de um PDF diretamente de bytes — ideal para PDFs gerados em memória.

        Args:
            pdf_bytes:  Conteúdo do PDF em bytes.
            filename:   Nome base do arquivo (com ou sem .pdf).
            folder:     Pasta de destino no bucket.

        Returns:
            {
                "public_id": str,
                "secure_url": str,
            }
        """
        public_id = os.path.splitext(filename)[0]
        blob_path = f"{folder}/{public_id}.pdf"
        blob = self.bucket.blob(blob_path)

        blob.upload_from_string(pdf_bytes, content_type="application/pdf")
        blob.make_public()

        return {
            "public_id": blob_path,
            "secure_url": blob.public_url,
        }

    # -------------------------------------------------------------------------
    # Deleção
    # -------------------------------------------------------------------------

    def delete_file(self, public_id: str) -> bool:
        """
        Remove um arquivo do Firebase Storage pelo caminho completo no bucket.

        Args:
            public_id: Caminho do arquivo no bucket (ex: "videos/abc123_original.mp4").

        Returns:
            True se removido com sucesso, False caso contrário.
        """
        try:
            blob = self.bucket.blob(public_id)
            blob.delete()
            return True
        except Exception as e:
            print(f"Erro ao deletar arquivo '{public_id}' do Firebase: {e}")
            return False