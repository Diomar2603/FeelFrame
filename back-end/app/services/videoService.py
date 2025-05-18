from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import shutil 
import os 

class VideoService:
    
    def __init__(self, filePath: str):
        self.filePath = filePath

    async def processFile(self, file: UploadFile):
        try:
            file_location = os.path.join(self.filePath, file.filename)

            contents = await file.read()

            processed_data = {
                "message": "Arquivo processado com sucesso (simulação)",
                "filename": file.filename,
                "content_type": file.content_type,
                "size_bytes": len(contents), 
            }

            with open(file_location, "wb+") as file_object:
                file_object.write(contents)

            await file.close()
        except Exception as e:
            processed_data = {
                "filename": file.filename,
                "status": "falha no processamento",
                "error": str(e)
            }
            await file.close()

        return processed_data
