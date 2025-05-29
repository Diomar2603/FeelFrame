from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
from app.services.videoService import VideoService
import shutil 
import os 

# --- Configuração do Router ---
router = APIRouter(
    prefix="/files",
    tags=["Files"]
)

# --- Configuração e Instância do Serviço ---
user_home_directory = os.path.expanduser("~")
pasta_de_uploads_no_usuario = "VideosFeelFrame"
VIDEO_UPLOAD_DIRECTORY = os.path.join(user_home_directory, pasta_de_uploads_no_usuario)

if not os.path.exists(VIDEO_UPLOAD_DIRECTORY):
    try:
        os.makedirs(VIDEO_UPLOAD_DIRECTORY)
        print(f"Diretório de upload criado: {VIDEO_UPLOAD_DIRECTORY}")
    except OSError as e:
        print(f"Falha ao criar o diretório de upload {VIDEO_UPLOAD_DIRECTORY}: {e}")
        
try:
    video_service_instance = VideoService(VIDEO_UPLOAD_DIRECTORY)
    print(f"VideoService instanciada com diretório: {VIDEO_UPLOAD_DIRECTORY}")
except Exception as e:
    print(f"Falha ao instanciar VideoService: {e}")
    video_service_instance = None 


# --- Rotas ---

@router.post("/upload/")
async def upload_single_file(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    try:
        processed_files_info = await video_service_instance.processFile(file)    

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar o arquivo: {str(e)}")
    
    return {"message": "Processamento concluído.", "files": processed_files_info}



@router.post("/upload-multiple/")
async def upload_multiple_files(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    processed_files_info = []

    for file in files:
        try:
            await video_service_instance.processFile(file)  

        except Exception as e:
            processed_files_info.append({
                "filename": file.filename,
                "status": "falha no processamento",
                "error": str(e)
            })
            await file.close() 

    return {"message": "Processamento concluído.", "files": processed_files_info}