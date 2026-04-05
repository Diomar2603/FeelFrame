import os
import shutil
import asyncio
from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from dotenv import load_dotenv

# --- Importações de Configuração e Serviço ---
from app.services.videoService import VideoService
from app.utils.DatabaseConfig import DatabaseConfig

# --- Configuração do Router ---
router = APIRouter(
    prefix="/files",
    tags=["Files"]
)

# --- Carregamento de Configurações e Instâncias ---

load_dotenv()

video_service_instance: VideoService | None = None
db_config_instance: DatabaseConfig | None = None

try:
    # Lê o caminho de upload do .env (com um padrão)
    upload_subfolder = os.environ.get("VIDEO_UPLOAD_SUBFOLDER", "VideosFeelFrame")
    user_home_directory = os.path.expanduser("~")
    VIDEO_UPLOAD_DIRECTORY = os.path.join(user_home_directory, upload_subfolder)

    # Cria o diretório se não existir
    if not os.path.exists(VIDEO_UPLOAD_DIRECTORY):
        os.makedirs(VIDEO_UPLOAD_DIRECTORY)
        print(f"Diretório de upload criado: {VIDEO_UPLOAD_DIRECTORY}")

    # Inicializa a configuração do DB PRIMEIRO
    db_config_instance = DatabaseConfig()
    
    # Inicializa o VideoService
    video_service_instance = VideoService(
        filePath=VIDEO_UPLOAD_DIRECTORY, 
        db_config=db_config_instance
    )
    print(f"VideoService instanciado com sucesso. DB: '{db_config_instance.db_name}', Caminho: '{VIDEO_UPLOAD_DIRECTORY}'")

except Exception as e:
    # Se qualquer coisa acima falhar (ex: DB offline), o serviço ficará como 'None'
    print(f"ERRO FATAL: Falha ao instanciar serviços: {e}")

# --- Rotas ---

@router.post("/upload/")
async def upload_single_file(file: UploadFile = File(...)):
    """Faz upload e processa um único arquivo de vídeo."""
    
    if video_service_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Erro fatal: O serviço de processamento de vídeo não foi inicializado."
        )

    if not file:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    try:
        processed_files_info = await video_service_instance.processFile(file)
        
        if processed_files_info.get("status") == "falha":
             raise HTTPException(status_code=500, detail=processed_files_info.get("error"))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar o arquivo: {str(e)}")
    
    return {"message": "Processamento concluído.", "files": processed_files_info}

@router.post("/upload-multiple/")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    """Faz upload e processa múltiplos arquivos de vídeo em paralelo."""

    if video_service_instance is None:
        raise HTTPException(
            status_code=503, 
            detail="Erro fatal: O serviço de processamento de vídeo não foi inicializado."
        )

    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    # Limita o número máximo de arquivos processados simultaneamente
    MAX_CONCURRENT_PROCESSING = 4
    if len(files) > MAX_CONCURRENT_PROCESSING:
        raise HTTPException(
            status_code=400, 
            detail=f"Máximo de {MAX_CONCURRENT_PROCESSING} arquivos permitidos simultaneamente. Enviados: {len(files)}"
        )

    try:
        # Usa o processamento paralelo do VideoService
        processed_files_info = await video_service_instance.process_multiple_files(files)
        
        # Conta sucessos e falhas
        success_count = len([f for f in processed_files_info if f.get("status") != "falha"])
        failure_count = len([f for f in processed_files_info if f.get("status") == "falha"])
        
        return {
            "message": f"Processamento em lote concluído. Sucessos: {success_count}, Falhas: {failure_count}",
            "total_files": len(files),
            "successful_processing": success_count,
            "failed_processing": failure_count,
            "files": processed_files_info
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erro durante o processamento em lote: {str(e)}"
        )

@router.post("/upload-multiple-batch/")
async def upload_multiple_files_batch(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    Faz upload de múltiplos arquivos e processa em lotes de 4 em 4 em background.
    Ideal para grandes quantidades de arquivos.
    """
    
    if video_service_instance is None:
        raise HTTPException(
            status_code=503, 
            detail="Erro fatal: O serviço de processamento de vídeo não foi inicializado."
        )

    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    # Salva os arquivos temporariamente
    saved_files = []
    try:
        for file in files:
            safe_filename = os.path.basename(file.filename)
            temp_location = os.path.join(video_service_instance.original_videos_dir, f"temp_{safe_filename}")
            
            with open(temp_location, "wb") as f:
                content = await file.read()
                f.write(content)
            
            saved_files.append({
                "filename": safe_filename,
                "temp_path": temp_location,
                "content_type": file.content_type
            })
            await file.close()

        # Agenda o processamento em background
        background_tasks.add_task(process_files_in_background, saved_files)
        
        return {
            "message": f"{len(files)} arquivos recebidos. Processamento em background iniciado.",
            "files_received": len(files),
            "batch_size": 4,
            "status": "processing_in_background"
        }

    except Exception as e:
        # Limpeza em caso de erro
        for saved_file in saved_files:
            try:
                if os.path.exists(saved_file["temp_path"]):
                    os.remove(saved_file["temp_path"])
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Erro ao salvar arquivos: {str(e)}")

async def process_files_in_background(saved_files: List[dict]):
    """
    Processa arquivos em lotes de 4 em background.
    """
    print(f"Iniciando processamento em background de {len(saved_files)} arquivos...")
    
    batch_size = 4
    for i in range(0, len(saved_files), batch_size):
        batch = saved_files[i:i + batch_size]
        print(f"Processando lote {i//batch_size + 1}: {len(batch)} arquivos")
        
        # Aqui você pode implementar o processamento do lote
        # Por simplicidade, vamos apenas simular
        for file_info in batch:
            try:
                print(f"Processando {file_info['filename']} em background...")
                # Simula processamento
                await asyncio.sleep(1)
            except Exception as e:
                print(f"Erro ao processar {file_info['filename']} em background: {e}")
        
        # Limpa arquivos temporários após processamento
        for file_info in batch:
            try:
                if os.path.exists(file_info["temp_path"]):
                    os.remove(file_info["temp_path"])
            except Exception as e:
                print(f"Erro ao limpar arquivo temporário {file_info['temp_path']}: {e}")
    
    print("Processamento em background concluído.")

@router.get("/processing-status/")
async def get_processing_status():
    """Retorna o status atual do serviço de processamento."""
    if video_service_instance is None:
        return {
            "status": "service_unavailable",
            "message": "Serviço de processamento não inicializado"
        }
    
    return {
        "status": "service_available",
        "message": "Serviço de processamento funcionando normalmente",
        "max_concurrent_processing": 4,
        "supported_formats": [".mp4", ".avi", ".mov", ".mkv"]
    }