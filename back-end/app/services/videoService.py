import os
import cv2 as cv
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymongo.collection import Collection
from fastapi import UploadFile
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from app.models.VideoMetadata import VideoMetadata
from app.services.faceAnalyzer import FaceAnalyzer 
from app.models.FrameAnalysis import FrameAnalysis, EstimativaEngajamentoEnum
from app.utils.DatabaseConfig import DatabaseConfig

class VideoService:
    
    def __init__(self, filePath: str, db_config: DatabaseConfig):
        self.db = db_config.client[db_config.db_name]
        self.video_collection: Collection = self.db["videos"]
        self.frame_analysis_collection: Collection = self.db["frame_analysis"]
        
        self.original_videos_dir = filePath + "\\videos_originais"
        self.fixed_frame_videos_dir = filePath + "\\" + os.getenv("FIXED_FRAME_VIDEOS_DIR", "videos_quadro_fixo")
        self.fluxo_frames_dir = filePath + "\\cenas_fluxo"
        self.desengajamento_frames_dir = filePath + "\\cenas_desengajamento"  # NOVO: Pasta para alto desengajamento

        self.OUTPUT_WIDTH = int(os.getenv("OUTPUT_WIDTH", "480"))
        self.OUTPUT_HEIGHT = int(os.getenv("OUTPUT_HEIGHT", "480"))
        
        try:
            self.frame_analysis_skip_rate = int(os.getenv("FRAME_ANALYSIS_SKIP_RATE", "6"))
            if self.frame_analysis_skip_rate <= 0:
                self.frame_analysis_skip_rate = 1
        except ValueError:
            self.frame_analysis_skip_rate = 6

        # Cria diretórios necessários
        os.makedirs(self.fixed_frame_videos_dir, exist_ok=True)
        os.makedirs(self.original_videos_dir, exist_ok=True)
        os.makedirs(self.fluxo_frames_dir, exist_ok=True)
        os.makedirs(self.desengajamento_frames_dir, exist_ok=True)  # NOVO

        self.face_analyzer = FaceAnalyzer()
        self.thread_pool = ThreadPoolExecutor(max_workers=4)  # NOVO: Pool de threads para processamento paralelo

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

    def _salvar_frame_destaque(self, frame_original, frame_number, timestamp_ms, video_id, analysis, tipo: str):
        """Salva frame original quando detectado estado de fluxo ou alto desengajamento."""
        try:
            # Cria nome do arquivo baseado no tipo
            timestamp_str = f"{timestamp_ms//1000}s_{timestamp_ms%1000:03d}ms"
            
            if tipo == "fluxo":
                filename = f"fluxo_{video_id}_{frame_number:06d}_{timestamp_str}.jpg"
                filepath = os.path.join(self.fluxo_frames_dir, filename)
            elif tipo == "desengajamento":
                filename = f"desengajamento_{video_id}_{frame_number:06d}_{timestamp_str}.jpg"
                filepath = os.path.join(self.desengajamento_frames_dir, filename)
            else:
                return None
            
            # Salva frame com qualidade boa
            cv.imwrite(filepath, frame_original, [cv.IMWRITE_JPEG_QUALITY, 90])
            
            # Retorna metadados
            return {
                "frame_number": frame_number,
                "timestamp_ms": timestamp_ms,
                "filepath": filepath,
                "tipo": tipo,
                "emocao": analysis.emocao,
                "engajamento": analysis.estimativa_engajamento.value,
                "dimensao": analysis.dimensao_comportamental.value,
                "confidence": getattr(analysis, 'emotion_confidence', 0.0)
            }
            
        except Exception as e:
            print(f"Erro ao salvar frame de {tipo}: {e}")
            return None

    def _salvar_metadados_destaque(self, video_id, frames_destaque_info):
        """Salva metadados dos frames de destaque no MongoDB."""
        try:
            if frames_destaque_info:
                destaque_collection = self.db["frames_destaque"]
                for frame_info in frames_destaque_info:
                    frame_info["video_id"] = video_id
                    frame_info["created_at"] = datetime.now(timezone.utc).isoformat()
                
                destaque_collection.insert_many(frames_destaque_info)
                print(f"Metadados de {len(frames_destaque_info)} frames de destaque salvos no banco.")
        except Exception as e:
            print(f"Erro ao salvar metadados de destaque: {e}")

    def _get_final_message(self, fixed_crop_rect, frames_destaque_info):
        """Gera mensagem final baseada nos resultados."""
        base_message = "Vídeo processado com sucesso."
        if fixed_crop_rect is None:
            base_message = "Nenhum rosto detectado com confiança. Quadros pretos foram usados."
        
        if frames_destaque_info:
            fluxo_count = len([f for f in frames_destaque_info if f['tipo'] == 'fluxo'])
            desengajamento_count = len([f for f in frames_destaque_info if f['tipo'] == 'desengajamento'])
            base_message += f" {fluxo_count} cenas de fluxo e {desengajamento_count} cenas de desengajamento detectadas."
        
        return base_message

    async def processFile(self, file: UploadFile) -> Dict[str, Any]:
        """
        Processa o vídeo e exporta frames de estado de fluxo e alto desengajamento.
        """
        output_file_location: str = ""
        original_file_location: str = ""
        fixed_crop_rect = None
        cap = out = None
        
        video_meta: Optional[VideoMetadata] = None
        video_id: Optional[str] = None

        black_frame = np.zeros((self.OUTPUT_HEIGHT, self.OUTPUT_WIDTH, 3), dtype=np.uint8)
        analysis_results_list = []

        # Lista para armazenar frames de destaque (fluxo e desengajamento)
        frames_destaque_info = []

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

            # Metadados do vídeo
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

            # Prepara video de saída
            fourcc = cv.VideoWriter_fourcc(*'avc1') 
            output_filename = f"quadro_fixo_{safe_original_filename}"
            if not output_filename.lower().endswith((".mp4", ".avi", ".mov")):
                output_filename += ".mp4"
            output_file_location = os.path.join(self.fixed_frame_videos_dir, output_filename)
            
            out = cv.VideoWriter(output_file_location, fourcc, fps, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT))
            if not out.isOpened():
                fourcc = cv.VideoWriter_fourcc(*'mp4v') 
                out = cv.VideoWriter(output_file_location, fourcc, fps, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT))
                if not out.isOpened():
                    raise Exception(f"Erro ao criar VideoWriter")

            input_frame_count = output_frame_count = 0
            frame_skip_rate = self.frame_analysis_skip_rate
            
            # Cache para detecção de faces
            face_detection_cache = None
            cache_valid_frames = 10

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break 
                
                input_frame_count += 1
                frame_resized = self.frameResize(frame, self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT)

                # Detecção de faces
                if (fixed_crop_rect is None or face_detection_cache is None or 
                    input_frame_count % cache_valid_frames == 0):
                    
                    faces = self.face_analyzer.detect_faces(frame_resized)
                    if len(faces) > 0 and faces[0][4] > 0.8:
                        x, y, w, h = faces[0][:4].astype(int)
                        cx = max(0, x + w // 2 - self.OUTPUT_WIDTH // 6)
                        cy = max(0, y + h // 2 - self.OUTPUT_HEIGHT // 6)
                        cw = min(self.OUTPUT_WIDTH, frame_resized.shape[1] - cx)
                        ch = min(self.OUTPUT_HEIGHT, frame_resized.shape[0] - cy)
                        fixed_crop_rect = (cx, cy, cw, ch)
                        face_detection_cache = fixed_crop_rect
                else:
                    fixed_crop_rect = face_detection_cache
                
                # Prepara frame de saída
                if fixed_crop_rect:
                    cx, cy, cw, ch = fixed_crop_rect
                    cropped = frame_resized[cy:cy+ch, cx:cx+cw]
                    frame_out = cv.resize(cropped, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT), 
                                        interpolation=cv.INTER_CUBIC) if cropped.size > 0 else black_frame
                else:
                    frame_out = black_frame

                # Análise de frames
                if (input_frame_count - 1) % frame_skip_rate == 0:
                    try:
                        timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))
                        
                        analysis_result_obj = self.face_analyzer.analyze_frame(
                            frame=frame_out, 
                            timestamp_ms=timestamp_ms,
                            video_id=video_id,
                            frame_number=input_frame_count
                        )
                        
                        # Verifica se é estado de fluxo e salva frame original
                        if hasattr(analysis_result_obj, 'estado_fluxo') and analysis_result_obj.estado_fluxo:
                            frame_destaque_info = self._salvar_frame_destaque(
                                frame, input_frame_count, timestamp_ms, video_id, analysis_result_obj, "fluxo"
                            )
                            if frame_destaque_info:
                                frames_destaque_info.append(frame_destaque_info)
                                print(f"Frame de fluxo detectado: {input_frame_count}")
                        
                        # NOVO: Verifica se é alto desengajamento e salva frame
                        elif analysis_result_obj.estimativa_engajamento == EstimativaEngajamentoEnum.ALTAMENTE_DESENGAJADO:
                            frame_destaque_info = self._salvar_frame_destaque(
                                frame, input_frame_count, timestamp_ms, video_id, analysis_result_obj, "desengajamento"
                            )
                            if frame_destaque_info:
                                frames_destaque_info.append(frame_destaque_info)
                                print(f"Frame de alto desengajamento detectado: {input_frame_count}")
                        
                        analysis_results_list.append(analysis_result_obj.to_dict())
                        
                        # Batch processing
                        if len(analysis_results_list) >= 50:
                            self.frame_analysis_collection.insert_many(analysis_results_list)
                            analysis_results_list = []
                            print(f"Batch de 50 análises inserido para video_id {video_id}")
                            
                    except Exception as e_analyze:
                        print(f"Aviso: Falha ao analisar frame {input_frame_count}: {e_analyze}")

                out.write(frame_out)
                output_frame_count += 1
            
            # Insere análises restantes
            if analysis_results_list:
                self.frame_analysis_collection.insert_many(analysis_results_list)
                print(f"Último batch de {len(analysis_results_list)} análises inserido.")

            # Salva metadados dos frames de destaque
            if frames_destaque_info:
                self._salvar_metadados_destaque(video_id, frames_destaque_info)
            
            # Atualiza banco com resultados
            duration_seconds = (input_frame_count / fps) if fps > 0 else 0.0
            final_message = self._get_final_message(fixed_crop_rect, frames_destaque_info)

            fluxo_count = len([f for f in frames_destaque_info if f['tipo'] == 'fluxo'])
            desengajamento_count = len([f for f in frames_destaque_info if f['tipo'] == 'desengajamento'])

            update_data = {
                "status": "success",
                "processing_message": final_message,
                "processed_filepath": output_file_location,
                "frame_count": input_frame_count,
                "duration_seconds": duration_seconds,
                "fluxo_frames_count": fluxo_count,
                "desengajamento_frames_count": desengajamento_count,
                "fixed_crop_rect_in_source": str(fixed_crop_rect) if fixed_crop_rect else "N/A",
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            self.video_collection.update_one({"_id": video_id}, {"$set": update_data})

            print(f"Sucesso no processamento do video_id: {video_id}")

            return {
                "message": final_message,
                "video_id": video_id, 
                "original_uploaded_filename": safe_original_filename,
                "fixed_frame_video_location": output_file_location,
                "input_frames_read": input_frame_count,
                "frames_analyzed": len(analysis_results_list),
                "fluxo_frames_captured": fluxo_count,
                "desengajamento_frames_captured": desengajamento_count
            }

        except Exception as e:
            error_message_str = str(e)
            print(f"Falha no processamento do video_id {video_id}: {error_message_str}")
            
            if video_id:
                self.video_collection.update_one(
                    {"_id": video_id},
                    {"$set": {
                        "status": "failed",
                        "error_message": error_message_str,
                        "updated_at": datetime.now(timezone.utc).isoformat()
                    }}
                )

            if out and out.isOpened(): 
                out.release()
            if output_file_location and os.path.exists(output_file_location):
                try: 
                    os.remove(output_file_location)
                except Exception as remove_error:
                    print(f"Erro ao tentar remover o arquivo de saída após falha: {remove_error}")
            
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

    # NOVO: Método para processamento em lote com threads
    async def process_multiple_files(self, files: List[UploadFile]) -> List[Dict[str, Any]]:
        """
        Processa múltiplos arquivos em paralelo usando thread pool.
        """
        print(f"Iniciando processamento paralelo de {len(files)} arquivos...")
        
        # Prepara as tasks para execução paralela
        tasks = []
        for file in files:
            # Cria uma task para cada arquivo
            task = asyncio.get_event_loop().run_in_executor(
                self.thread_pool, 
                self._process_single_file_sync, 
                file
            )
            tasks.append(task)
        
        # Executa todas as tasks em paralelo (máximo 4 simultaneamente)
        results = []
        for completed_task in asyncio.as_completed(tasks):
            try:
                result = await completed_task
                results.append(result)
            except Exception as e:
                results.append({
                    "filename": "unknown",
                    "status": "falha",
                    "error": str(e)
                })
        
        print(f"Processamento paralelo concluído. {len(results)} arquivos processados.")
        return results

    def _process_single_file_sync(self, file: UploadFile) -> Dict[str, Any]:
        """
        Método auxiliar para processar um arquivo de forma síncrona (para uso com thread pool).
        """
        # Cria um novo event loop para a thread
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Executa o processamento de forma síncrona
            result = loop.run_until_complete(self.processFile(file))
            return result
        except Exception as e:
            return {
                "filename": file.filename,
                "status": "falha",
                "error": str(e)
            }
        finally:
            loop.close()

    def __del__(self):
        """Garante que o thread pool seja fechado adequadamente."""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)