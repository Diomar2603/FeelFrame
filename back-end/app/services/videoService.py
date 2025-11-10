import os
import cv2 as cv
import numpy as np
from pymongo.collection import Collection
from fastapi import UploadFile
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from app.models.VideoMetadata import VideoMetadata
from app.services.faceAnalyzer import FaceAnalyzer 
from app.models.FrameAnalysis import FrameAnalysis
from app.utils.DatabaseConfig import DatabaseConfig

class VideoService:
    
    def __init__(self, filePath: str, db_config: DatabaseConfig):
        
        self.db = db_config.client[db_config.db_name]
        self.video_collection: Collection = self.db["videos"]
        self.frame_analysis_collection: Collection = self.db["frame_analysis"]
        
        self.original_videos_dir = filePath + "\\videos_originais"
        
        self.fixed_frame_videos_dir = filePath + "\\" +os.getenv("FIXED_FRAME_VIDEOS_DIR", "videos_quadro_fixo")

        self.OUTPUT_WIDTH = int(os.getenv("OUTPUT_WIDTH", "480"))
        self.OUTPUT_HEIGHT = int(os.getenv("OUTPUT_HEIGHT", "480"))
        
        try:
            self.frame_analysis_skip_rate = int(os.getenv("FRAME_ANALYSIS_SKIP_RATE", "4"))
            if self.frame_analysis_skip_rate <= 0:
                print("Aviso: FRAME_ANALYSIS_SKIP_RATE inválido, usando padrão 1 (todos os frames)")
                self.frame_analysis_skip_rate = 1
        except ValueError:
            print("Aviso: FRAME_ANALYSIS_SKIP_RATE inválido, usando padrão 5")
            self.frame_analysis_skip_rate = 5

        os.makedirs(self.fixed_frame_videos_dir, exist_ok=True)
        os.makedirs(self.original_videos_dir, exist_ok=True) 

        self.face_analyzer = FaceAnalyzer()
        
    async def saveFile(self, file: UploadFile) -> str:
        """Salva o arquivo de upload no diretório 'original'."""
        safe_filename = os.path.basename(file.filename) 
        location = os.path.join(self.original_videos_dir, safe_filename)
        with open(location, "wb") as f:
            f.write(await file.read())
        return location

    def frameResize(self, frame, width, height):
        """Redimensiona um frame para as dimensões de saída."""
        return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

    async def processFile(self, file: UploadFile) -> Dict[str, Any]:
        """
        Orquestra o processo completo:
        1. Salva o arquivo.
        2. Extrai metadados.
        3. Insere registro "processing" no DB de vídeos.
        4. Processa o vídeo (quadro a quadro), gerando o vídeo de saída
           E analisando os frames (a cada X quadros).
        5. Insere as análises de frame no DB de análises.
        6. Atualiza o DB de vídeos com "success" ou "failed".
        """
        output_file_location: str = ""
        original_file_location: str = ""
        fixed_crop_rect = None
        cap = out = None
        
        video_meta: Optional[VideoMetadata] = None
        video_id: Optional[str] = None

        black_frame = np.zeros((self.OUTPUT_HEIGHT, self.OUTPUT_WIDTH, 3), dtype=np.uint8)

        analysis_results_list = []

        try:
            original_file_location = await self.saveFile(file)
            
            cap = cv.VideoCapture(original_file_location)
            if not cap.isOpened():
                raise Exception(f"Erro ao abrir vídeo: {original_file_location}")

            fps = cap.get(cv.CAP_PROP_FPS) or 25.0
            original_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            original_filesize_bytes = os.path.getsize(original_file_location)
            safe_original_filename = os.path.basename(file.filename)

            output_filename = f"quadro_fixo_{safe_original_filename}"
            if not output_filename.lower().endswith((".mp4", ".avi", ".mov")):
                output_filename += ".mp4"
            output_file_location = os.path.join(self.fixed_frame_videos_dir, output_filename)

            video_meta = VideoMetadata(
                original_filename=safe_original_filename,
                original_filepath=original_file_location,
                original_filesize_bytes=original_filesize_bytes,
                original_width=original_width,
                original_height=original_height,
                fps=fps,
                processed_width=self.OUTPUT_WIDTH,
                processed_height=self.OUTPUT_HEIGHT
            )

            self.video_collection.insert_one(video_meta.to_dict())
            video_id = video_meta._id 
            print(f"Iniciando processamento para video_id: {video_id}")

            fourcc = cv.VideoWriter_fourcc(*'avc1') 
            out = cv.VideoWriter(output_file_location, fourcc, fps, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT))
            if not out.isOpened():
                print(f"Aviso: Falha ao abrir VideoWriter com 'avc1'. Tentando com 'mp4v' para {output_filename}.")
                fourcc = cv.VideoWriter_fourcc(*'mp4v') 
                out = cv.VideoWriter(output_file_location, fourcc, fps, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT))
                if not out.isOpened():
                    raise Exception(f"Erro ao criar VideoWriter para {output_filename} com codecs 'avc1' e 'mp4v'.")

            input_frame_count = output_frame_count = 0
            
            frame_skip_rate = self.frame_analysis_skip_rate
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break 
                input_frame_count += 1
                
                frame_resized = self.frameResize(frame, self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT)

                if fixed_crop_rect is None:
                    faces = self.face_analyzer.detect_faces(frame_resized)
                    if len(faces) > 0 and faces[0][4] > 0.8: # Confiança > 80%
                        x, y, w, h = faces[0][:4].astype(int)
                        cx = max(0, x + w // 2 - self.OUTPUT_WIDTH // 6)
                        cy = max(0, y + h // 2 - self.OUTPUT_HEIGHT // 6)
                        cw = min(self.OUTPUT_WIDTH, frame_resized.shape[1] - cx)
                        ch = min(self.OUTPUT_HEIGHT, frame_resized.shape[0] - cy)
                        fixed_crop_rect = (cx, cy, cw, ch)
                
                if fixed_crop_rect:
                    cx, cy, cw, ch = fixed_crop_rect
                    cropped = frame_resized[cy:cy+ch, cx:cx+cw]
                    if cropped.size > 0:
                        frame_out = cv.resize(cropped, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT), interpolation=cv.INTER_CUBIC)
                    else:
                        frame_out = black_frame 
                else:
                    frame_out = black_frame 

                if (input_frame_count - 1) % frame_skip_rate == 0:
                    try:
                        timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
                        
                        analysis_result_obj = self.face_analyzer.analyze_frame(
                            frame=frame_out, 
                            timestamp_ms=timestamp_ms,
                            video_id=video_id,
                            frame_number=input_frame_count
                        )
                        
                        analysis_results_list.append(analysis_result_obj.to_dict())
                        
                    except Exception as e_analyze:
                        print(f"Aviso: Falha ao analisar o frame {input_frame_count} para video_id {video_id}. Erro: {e_analyze}")

                out.write(frame_out)
                output_frame_count += 1
            
            if analysis_results_list:
                print(f"Processamento de frames concluído. Inserindo {len(analysis_results_list)} análises no DB...")
                self.frame_analysis_collection.insert_many(analysis_results_list)
                print("Análises inseridas com sucesso.")
            
            final_message = "Vídeo processado com sucesso."
            if input_frame_count == 0:
                if os.path.exists(output_file_location): os.remove(output_file_location)
                raise Exception("Vídeo de entrada está vazio ou corrompido.")
            if fixed_crop_rect is None:
                final_message = "Nenhum rosto detectado com confiança. Quadros pretos foram usados."
            
            duration_seconds = (input_frame_count / fps) if fps > 0 else 0.0

            update_data = {
                "status": "success",
                "processing_message": final_message,
                "processed_filepath": output_file_location,
                "frame_count": input_frame_count,
                "duration_seconds": duration_seconds,
                "fixed_crop_rect_in_source": str(fixed_crop_rect) if fixed_crop_rect else "N/A",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            self.video_collection.update_one(
                {"_id": video_id},
                {"$set": update_data}
            )
            print(f"Sucesso no processamento do video_id: {video_id}")

            return {
                "message": final_message,
                "video_id": video_id, 
                "original_uploaded_filename": safe_original_filename,
                "fixed_frame_video_location": output_file_location,
                "input_frames_read": input_frame_count,
                "frames_analyzed": len(analysis_results_list)
            }

        except Exception as e:
            error_message_str = str(e)
            print(f"Falha no processamento do video_id {video_id}: {error_message_str}")
            
            # 10. ATUALIZA no MongoDB (Falha)
            if video_id: # Só atualiza se o registro já foi criado
                self.video_collection.update_one(
                    {"_id": video_id},
                    {"$set": {
                        "status": "failed",
                        "error_message": error_message_str,
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    }}
                )

            # Limpa o arquivo de saída parcialmente criado (se houver)
            if out and out.isOpened(): out.release()
            out = None 
            if output_file_location and os.path.exists(output_file_location):
                try:
                    os.remove(output_file_location)
                except Exception as remove_error:
                    print(f"Erro ao tentar remover o arquivo de saída após falha: {remove_error}")
            
            # Retorna o erro para a API
            return {
                "filename": file.filename,
                "status": "falha",
                "error": error_message_str,
                "video_id": video_id 
            }
        finally:
            if cap and cap.isOpened(): 
                cap.release()
            if out and out.isOpened(): 
                out.release()