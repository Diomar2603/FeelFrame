import os
import shutil
from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException
from dotenv import load_dotenv

# --- Importações de Configuração e Serviço ---
# Importa as classes de que precisamos
from app.services.videoService import VideoService
from app.utils.DatabaseConfig import DatabaseConfig

# --- Configuração do Router ---
router = APIRouter(
    prefix="/files",
    tags=["Files"]
)

# --- Carregamento de Configurações e Instâncias ---

# 1. Carrega as variáveis do arquivo .env para o ambiente (os.environ)
load_dotenv()

video_service_instance: VideoService | None = None
db_config_instance: DatabaseConfig | None = None

try:
    # 3. Lê o caminho de upload do .env (com um padrão)
    upload_subfolder = os.environ.get("VIDEO_UPLOAD_SUBFOLDER", "VideosFeelFrame")
    user_home_directory = os.path.expanduser("~")
    VIDEO_UPLOAD_DIRECTORY = os.path.join(user_home_directory, upload_subfolder)

    # Cria o diretório se não existir
    if not os.path.exists(VIDEO_UPLOAD_DIRECTORY):
        os.makedirs(VIDEO_UPLOAD_DIRECTORY)
        print(f"Diretório de upload criado: {VIDEO_UPLOAD_DIRECTORY}")

    # 4. Inicializa a configuração do DB PRIMEIRO
    db_config_instance = DatabaseConfig()
    
    # 5. (CORREÇÃO) Passa AMBOS os argumentos para o VideoService
    #    Adicionado 'filePath=VIDEO_UPLOAD_DIRECTORY'
    video_service_instance = VideoService(
        filePath=VIDEO_UPLOAD_DIRECTORY, 
        db_config=db_config_instance
    )
    print(f"VideoService instanciado com sucesso. DB: '{db_config_instance.db_name}', Caminho: '{VIDEO_UPLOAD_DIRECTORY}'")

except Exception as e:
    # Se qualquer coisa acima falhar (ex: DB offline), o serviço ficará como 'None'
    print(f"ERRO FATAL: Falha ao instanciar serviços: {e}")
    # (video_service_instance permanece None)


# --- Rotas ---

@router.post("/upload/")
async def upload_single_file(file: UploadFile = File(...)):
    """Faz upload e processa um único arquivo de vídeo."""
    
    # 6. (CORREÇÃO DE ERRO) Verifica se o serviço está funcional
    if video_service_instance is None:
        raise HTTPException(
            status_code=503, # 503 Service Unavailable
            detail="Erro fatal: O serviço de processamento de vídeo não foi inicializado."
        )

    if not file:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    try:
        # A instância do serviço é válida aqui
        processed_files_info = await video_service_instance.processFile(file)
        
        if processed_files_info.get("status") == "falha":
             raise HTTPException(status_code=500, detail=processed_files_info.get("error"))

    except Exception as e:
        # Captura outros erros inesperados
        raise HTTPException(status_code=500, detail=f"Erro ao processar o arquivo: {str(e)}")
    
    return {"message": "Processamento concluído.", "files": processed_files_info}



@router.post("/upload-multiple/")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """Faz upload e processa múltiplos arquivos de vídeo."""

    if video_service_instance is None:
        raise HTTPException(
            status_code=503, 
            detail="Erro fatal: O serviço de processamento de vídeo não foi inicializado."
        )

    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    processed_files_info = []

    for file in files:
        try:
            result = await video_service_instance.processFile(file)
            processed_files_info.append(result)

        except Exception as e:
            processed_files_info.append({
                "filename": file.filename,
                "status": "falha na chamada",
                "error": str(e)
            })
        finally:
            await file.close() 

    return {"message": "Processamento em lote concluído.", "files": processed_files_info}