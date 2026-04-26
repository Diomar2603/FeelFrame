import os
import cv2 as cv
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pymongo.collection import Collection
from fastapi import UploadFile
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from app.models.VideoMetadata import VideoMetadata
from app.services.faceAnalyzer import FaceAnalyzer
from app.models.FrameAnalysis import FrameAnalysis, EstimativaEngajamentoEnum
from app.utils.DatabaseConfig import DatabaseConfig
from app.services.firebaseStorageService import FirebaseStorageService


class VideoService:

    def __init__(self, filePath: str, db_config: DatabaseConfig):
        self.db = db_config.client[db_config.db_name]
        self.video_collection: Collection = self.db["videos"]
        self.frame_analysis_collection: Collection = self.db["frame_analysis"]

        self.original_videos_dir = os.path.join(filePath, "videos_originais")
        self.fixed_frame_videos_dir = os.path.join(filePath, os.getenv("FIXED_FRAME_VIDEOS_DIR", "videos_quadro_fixo"))
        self.fluxo_frames_dir = os.path.join(filePath, "cenas_fluxo")
        self.desengajamento_frames_dir = os.path.join(filePath, "cenas_desengajamento")

        self.OUTPUT_WIDTH = int(os.getenv("OUTPUT_WIDTH", "480"))
        self.OUTPUT_HEIGHT = int(os.getenv("OUTPUT_HEIGHT", "480"))

        # MUDANÇA: Forçando a análise de TODOS os frames para máxima precisão (skip_rate = 1)
        self.frame_analysis_skip_rate = 1 

        os.makedirs(self.fixed_frame_videos_dir, exist_ok=True)
        os.makedirs(self.original_videos_dir, exist_ok=True)
        os.makedirs(self.fluxo_frames_dir, exist_ok=True)
        os.makedirs(self.desengajamento_frames_dir, exist_ok=True)

        # MUDANÇA: O FaceAnalyzer global foi removido daqui para evitar poluição de memória entre requisições
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.storage = FirebaseStorageService()

    async def saveFile(self, file: UploadFile) -> str:
        safe_filename = os.path.basename(file.filename)
        location = os.path.join(self.original_videos_dir, safe_filename)
        with open(location, "wb") as f:
            f.write(await file.read())
        return location

    async def saveFileFromBytes(self, file_bytes: bytes, filename: str) -> str:
        safe_filename = os.path.basename(filename)
        location = os.path.join(self.original_videos_dir, safe_filename)
        with open(location, "wb") as f:
            f.write(file_bytes)
        return location

    def frameResize(self, frame, width, height):
        return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

    def _crop_head_padded(self, frame, face_rect: tuple) -> np.ndarray:
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = face_rect

        pad_top    = int(h * 0.80)
        pad_bottom = int(h * 0.30)
        pad_side   = int(w * 0.40)

        x1 = x - pad_side
        y1 = y - pad_top
        x2 = x + w + pad_side
        y2 = y + h + pad_bottom

        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_h > crop_w:
            diff = crop_h - crop_w
            x1 -= diff // 2
            x2 += diff - diff // 2
        elif crop_w > crop_h:
            diff = crop_w - crop_h
            y1 -= diff // 2
            y2 += diff - diff // 2

        pad_left_fill   = max(0, -x1)
        pad_top_fill    = max(0, -y1)
        pad_right_fill  = max(0, x2 - frame_w)
        pad_bottom_fill = max(0, y2 - frame_h)

        x1c = max(0, x1);   y1c = max(0, y1)
        x2c = min(frame_w, x2);  y2c = min(frame_h, y2)

        cropped = frame[y1c:y2c, x1c:x2c]

        if pad_left_fill or pad_top_fill or pad_right_fill or pad_bottom_fill:
            cropped = cv.copyMakeBorder(
                cropped,
                pad_top_fill, pad_bottom_fill,
                pad_left_fill, pad_right_fill,
                cv.BORDER_CONSTANT, value=(0, 0, 0)
            )

        return cv.resize(cropped, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT), interpolation=cv.INTER_LANCZOS4)

    def _update_progress(self, video_id: str, percent: int, message: Optional[str] = None):
        update: Dict[str, Any] = {
            "progress_percent": percent,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if message: update["processing_message"] = message
        self.video_collection.update_one({"_id": video_id}, {"$set": update})

    def _salvar_frame_destaque(self, frame_original, frame_number, timestamp_ms, video_id, analysis, tipo: str):
        try:
            timestamp_str = f"{timestamp_ms // 1000}s_{timestamp_ms % 1000:03d}ms"
            if tipo == "fluxo":
                filename = f"fluxo_{video_id}_{frame_number:06d}_{timestamp_str}.jpg"
                folder = "feelframe/frames_destaque/fluxo"
            elif tipo == "desengajamento":
                filename = f"desengajamento_{video_id}_{frame_number:06d}_{timestamp_str}.jpg"
                folder = "feelframe/frames_destaque/desengajamento"
            else: return None

            success, buffer = cv.imencode(".jpg", frame_original, [cv.IMWRITE_JPEG_QUALITY, 90])
            if not success: raise RuntimeError(f"Falha ao codificar frame {frame_number} como JPEG.")

            image_bytes = buffer.tobytes()
            public_id = os.path.splitext(filename)[0] 
            blob_path = f"{folder}/{public_id}.jpg"
            blob = self.storage.bucket.blob(blob_path)
            blob.upload_from_string(image_bytes, content_type="image/jpeg")
            blob.make_public()
            firebase_url = blob.public_url

            return {
                "frame_number": frame_number,
                "timestamp_ms": timestamp_ms,
                "firebase_url": firebase_url,
                "tipo": tipo,
                "emocao": analysis.emocao,
                "engajamento": analysis.estimativa_engajamento.value,
                "dimensao": analysis.dimensao_comportamental.value,
                "confidence": getattr(analysis, "emotion_confidence", 0.0),
            }
        except Exception as e:
            print(f"Erro ao fazer upload do frame de destaque ({tipo}): {e}")
            return None

    def _salvar_metadados_destaque(self, video_id, frames_destaque_info):
        try:
            if frames_destaque_info:
                destaque_collection = self.db["frames_destaque"]
                for frame_info in frames_destaque_info:
                    frame_info["video_id"] = video_id
                    frame_info["created_at"] = datetime.now(timezone.utc).isoformat()
                destaque_collection.insert_many(frames_destaque_info)
        except Exception as e:
            print(f"Erro ao salvar metadados de destaque: {e}")

    def _get_final_message(self, fixed_crop_rect, frames_destaque_info):
        base_message = "Vídeo processado com sucesso."
        if fixed_crop_rect is None:
            base_message = "Nenhum rosto detectado com confiança. Quadros pretos foram usados."
        if frames_destaque_info:
            fluxo_count = len([f for f in frames_destaque_info if f["tipo"] == "fluxo"])
            desengajamento_count = len([f for f in frames_destaque_info if f["tipo"] == "desengajamento"])
            base_message += f" {fluxo_count} cenas de fluxo e {desengajamento_count} de desengajamento detectadas."
        return base_message

    async def processFile(self, file: UploadFile) -> Dict[str, Any]:
        output_file_location: str = ""
        original_file_location: str = ""
        fixed_crop_rect = None
        cap = out = None

        video_meta: Optional[VideoMetadata] = None
        video_id: Optional[str] = None

        black_frame = np.zeros((self.OUTPUT_HEIGHT, self.OUTPUT_WIDTH, 3), dtype=np.uint8)
        analysis_results_list = []
        frames_destaque_info = []
        
        # MUDANÇA CRÍTICA: Instancia um Analyzer LIMPO e EXCLUSIVO para este vídeo
        local_analyzer = FaceAnalyzer()

        try:
            self._update_progress(video_id or "", 0, "Salvando arquivo...")
            original_file_location = await self.saveFile(file)

            cap = cv.VideoCapture(original_file_location)
            if not cap.isOpened(): raise Exception(f"Erro ao abrir vídeo: {original_file_location}")

            fps = cap.get(cv.CAP_PROP_FPS) or 25.0
            original_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or 1
            original_filesize_bytes = os.path.getsize(original_file_location)
            safe_original_filename = os.path.basename(file.filename)

            video_meta = VideoMetadata(
                original_filename=safe_original_filename, original_filepath=original_file_location,
                original_filesize_bytes=original_filesize_bytes, original_width=original_width,
                original_height=original_height, fps=fps,
                processed_width=self.OUTPUT_WIDTH, processed_height=self.OUTPUT_HEIGHT,
            )
            self.video_collection.insert_one(video_meta.to_dict())
            video_id = video_meta._id
            
            self._update_progress(video_id, 5, "Enviando vídeo original para a nuvem...")
            original_cloud = self.storage.upload_video(
                file_path=original_file_location, public_id=f"{video_id}_original", folder="feelframe/videos_originais",
            )
            self.video_collection.update_one(
                {"_id": video_id},
                {"$set": {"original_url": original_cloud["secure_url"], "original_filepath": original_cloud["secure_url"]}},
            )

            fourcc = cv.VideoWriter_fourcc(*"avc1")
            output_filename = f"quadro_fixo_{safe_original_filename}"
            if not output_filename.lower().endswith((".mp4", ".avi", ".mov")): output_filename += ".mp4"
            output_file_location = os.path.join(self.fixed_frame_videos_dir, output_filename)

            out = cv.VideoWriter(output_file_location, fourcc, fps, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT))

            input_frame_count = 0
            output_frame_count = 0

            self._update_progress(video_id, 10, "Processando frames com análise máxima...")

            while True:
                ret, frame = cap.read()
                if not ret: break

                input_frame_count += 1
                timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))

                raw_progress = int((input_frame_count / total_frames) * 75) 
                progress_value = 10 + raw_progress                           
                if input_frame_count % max(1, total_frames // 20) == 0:
                    self._update_progress(video_id, progress_value, f"Analisando quadros em alta precisão... {progress_value}%")

                frame_out = black_frame.copy()

                if input_frame_count == 1 or fixed_crop_rect is None:
                    # Usa a instância local para detecção
                    detected_rect = local_analyzer.detect_face_rect(frame)
                    if detected_rect: fixed_crop_rect = detected_rect

                if fixed_crop_rect: frame_out = self._crop_head_padded(frame, fixed_crop_rect)
                else: frame_out = self.frameResize(frame, self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT)

                if input_frame_count % self.frame_analysis_skip_rate == 0:
                    try:
                        # Usa a instância local para análise
                        analysis_result_obj: FrameAnalysis = local_analyzer.analyze(
                            frame=frame_out, video_id=video_id, timestamp_ms=timestamp_ms, frame_number=input_frame_count,
                        )

                        if hasattr(analysis_result_obj, "estado_fluxo") and analysis_result_obj.estado_fluxo:
                            info = self._salvar_frame_destaque(frame, input_frame_count, timestamp_ms, video_id, analysis_result_obj, "fluxo")
                            if info: frames_destaque_info.append(info)
                        elif analysis_result_obj.estimativa_engajamento == EstimativaEngajamentoEnum.ALTAMENTE_DESENGAJADO:
                            info = self._salvar_frame_destaque(frame, input_frame_count, timestamp_ms, video_id, analysis_result_obj, "desengajamento")
                            if info: frames_destaque_info.append(info)

                        analysis_results_list.append(analysis_result_obj.to_dict())

                        if len(analysis_results_list) >= 150: # Batch de DB maior já que analisamos mais frames
                            self.frame_analysis_collection.insert_many(analysis_results_list)
                            analysis_results_list = []

                    except Exception as e_analyze:
                        print(f"Aviso: Falha ao analisar frame {input_frame_count}: {e_analyze}")

                out.write(frame_out)
                output_frame_count += 1

            if analysis_results_list: self.frame_analysis_collection.insert_many(analysis_results_list)
            if frames_destaque_info: self._salvar_metadados_destaque(video_id, frames_destaque_info)

            cap.release()
            out.release()
            cap = out = None

            self._update_progress(video_id, 85, "Enviando vídeo processado para a nuvem...")
            processed_cloud = self.storage.upload_video(
                file_path=output_file_location, public_id=f"{video_id}_processado", folder="feelframe/videos_processados",
            )

            duration_seconds = (input_frame_count / fps) if fps > 0 else 0.0
            final_message = self._get_final_message(fixed_crop_rect, frames_destaque_info)
            fluxo_count = len([f for f in frames_destaque_info if f["tipo"] == "fluxo"])
            desengajamento_count = len([f for f in frames_destaque_info if f["tipo"] == "desengajamento"])

            update_data = {
                "status": "success", "progress_percent": 100, "processing_message": final_message,
                "processed_url": processed_cloud["secure_url"], "processed_filepath": processed_cloud["secure_url"],
                "frame_count": input_frame_count, "duration_seconds": duration_seconds,
                "fluxo_frames_count": fluxo_count, "desengajamento_frames_count": desengajamento_count,
                "fixed_crop_rect_in_source": str(fixed_crop_rect) if fixed_crop_rect else "N/A",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.video_collection.update_one({"_id": video_id}, {"$set": update_data})

            self._cleanup_local_files(original_file_location, output_file_location)

            return {
                "message": final_message, "video_id": video_id, "original_filename": safe_original_filename,
                "original_url": original_cloud["secure_url"], "processed_url": processed_cloud["secure_url"],
                "input_frames_read": input_frame_count, "fluxo_frames_captured": fluxo_count,
                "desengajamento_frames_captured": desengajamento_count,
            }

        except Exception as e:
            error_message_str = str(e)
            if video_id:
                self.video_collection.update_one({"_id": video_id}, {"$set": {"status": "failed", "progress_percent": 0, "error_message": error_message_str, "updated_at": datetime.now(timezone.utc).isoformat()}})
            return {"filename": file.filename, "status": "falha", "error": error_message_str, "video_id": video_id}
        finally:
            if cap and cap.isOpened(): cap.release()
            if out and out.isOpened(): out.release()

    def _cleanup_local_files(self, *paths: str):
        for path in paths:
            if path and os.path.exists(path):
                try: os.remove(path)
                except Exception as e: print(f"Aviso: não foi possível remover {path}: {e}")

    async def processFile_by_id(self, file_bytes: bytes, filename: str, video_id: str) -> Dict[str, Any]:
        output_file_location: str = ""
        original_file_location: str = ""
        fixed_crop_rect = None
        cap = out = None

        black_frame = np.zeros((self.OUTPUT_HEIGHT, self.OUTPUT_WIDTH, 3), dtype=np.uint8)
        analysis_results_list = []
        frames_destaque_info = []

        # MUDANÇA CRÍTICA: Instancia um Analyzer LIMPO e EXCLUSIVO
        local_analyzer = FaceAnalyzer()

        try:
            self._update_progress(video_id, 0, "Salvando arquivo...")
            original_file_location = await self.saveFileFromBytes(file_bytes, filename)

            cap = cv.VideoCapture(original_file_location)
            if not cap.isOpened(): raise Exception(f"Erro ao abrir vídeo: {original_file_location}")

            fps = cap.get(cv.CAP_PROP_FPS) or 25.0
            original_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or 1
            original_filesize_bytes = os.path.getsize(original_file_location)
            safe_original_filename = os.path.basename(filename)

            self._update_progress(video_id, 2, "Registrando metadados...")
            self.video_collection.update_one(
                {"_id": video_id},
                {"$set": {
                    "status": "processing", "original_filename": safe_original_filename,
                    "original_filepath": original_file_location, "original_filesize_bytes": original_filesize_bytes,
                    "original_width": original_width, "original_height": original_height, "fps": fps,
                    "processed_width": self.OUTPUT_WIDTH, "processed_height": self.OUTPUT_HEIGHT,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }},
            )

            self._update_progress(video_id, 5, "Enviando vídeo original para a nuvem...")
            original_cloud = self.storage.upload_video(
                file_path=original_file_location, public_id=f"{video_id}_original", folder="feelframe/videos_originais",
            )
            self.video_collection.update_one({"_id": video_id}, {"$set": {"original_url": original_cloud["secure_url"], "original_filepath": original_cloud["secure_url"]}})

            fourcc = cv.VideoWriter_fourcc(*"avc1")
            output_filename = f"quadro_fixo_{safe_original_filename}"
            if not output_filename.lower().endswith((".mp4", ".avi", ".mov")): output_filename += ".mp4"
            output_file_location = os.path.join(self.fixed_frame_videos_dir, output_filename)

            out = cv.VideoWriter(output_file_location, fourcc, fps, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT))

            input_frame_count = 0
            output_frame_count = 0

            self._update_progress(video_id, 10, "Processando frames com análise máxima...")

            while True:
                ret, frame = cap.read()
                if not ret: break

                input_frame_count += 1
                timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))

                raw_progress = int((input_frame_count / total_frames) * 75)
                progress_value = 10 + raw_progress
                if input_frame_count % max(1, total_frames // 20) == 0:
                    self._update_progress(video_id, progress_value, f"Analisando quadros em alta precisão... {progress_value}%")

                frame_out = black_frame.copy()

                if input_frame_count == 1 or fixed_crop_rect is None:
                    detected_rect = local_analyzer.detect_face_rect(frame)
                    if detected_rect: fixed_crop_rect = detected_rect

                if fixed_crop_rect: frame_out = self._crop_head_padded(frame, fixed_crop_rect)
                else: frame_out = self.frameResize(frame, self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT)

                if input_frame_count % self.frame_analysis_skip_rate == 0:
                    try:
                        from app.models.FrameAnalysis import EstimativaEngajamentoEnum
                        analysis_result_obj: FrameAnalysis = local_analyzer.analyze(
                            frame=frame_out, video_id=video_id, timestamp_ms=timestamp_ms, frame_number=input_frame_count,
                        )

                        if hasattr(analysis_result_obj, "estado_fluxo") and analysis_result_obj.estado_fluxo:
                            info = self._salvar_frame_destaque(frame, input_frame_count, timestamp_ms, video_id, analysis_result_obj, "fluxo")
                            if info: frames_destaque_info.append(info)
                        elif analysis_result_obj.estimativa_engajamento == EstimativaEngajamentoEnum.ALTAMENTE_DESENGAJADO:
                            info = self._salvar_frame_destaque(frame, input_frame_count, timestamp_ms, video_id, analysis_result_obj, "desengajamento")
                            if info: frames_destaque_info.append(info)

                        analysis_results_list.append(analysis_result_obj.to_dict())

                        if len(analysis_results_list) >= 150:
                            self.frame_analysis_collection.insert_many(analysis_results_list)
                            analysis_results_list = []

                    except Exception as e_analyze: print(f"Aviso: Falha ao analisar frame {input_frame_count}: {e_analyze}")

                out.write(frame_out)
                output_frame_count += 1

            if analysis_results_list: self.frame_analysis_collection.insert_many(analysis_results_list)
            if frames_destaque_info: self._salvar_metadados_destaque(video_id, frames_destaque_info)

            cap.release()
            out.release()
            cap = out = None

            self._update_progress(video_id, 85, "Enviando vídeo processado para a nuvem...")
            processed_cloud = self.storage.upload_video(
                file_path=output_file_location, public_id=f"{video_id}_processado", folder="feelframe/videos_processados",
            )

            duration_seconds = (input_frame_count / fps) if fps > 0 else 0.0
            final_message = self._get_final_message(fixed_crop_rect, frames_destaque_info)
            fluxo_count = len([f for f in frames_destaque_info if f["tipo"] == "fluxo"])
            desengajamento_count = len([f for f in frames_destaque_info if f["tipo"] == "desengajamento"])

            self.video_collection.update_one(
                {"_id": video_id},
                {"$set": {
                    "status": "success", "progress_percent": 100, "processing_message": final_message,
                    "processed_url": processed_cloud["secure_url"], "processed_filepath": processed_cloud["secure_url"],
                    "frame_count": input_frame_count, "duration_seconds": duration_seconds,
                    "fluxo_frames_count": fluxo_count, "desengajamento_frames_count": desengajamento_count,
                    "fixed_crop_rect_in_source": str(fixed_crop_rect) if fixed_crop_rect else "N/A",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }},
            )

            self._cleanup_local_files(original_file_location, output_file_location)
            return {
                "message": final_message, "video_id": video_id, "original_filename": safe_original_filename,
                "original_url": original_cloud["secure_url"], "processed_url": processed_cloud["secure_url"],
                "input_frames_read": input_frame_count, "fluxo_frames_captured": fluxo_count,
                "desengajamento_frames_captured": desengajamento_count,
            }

        except Exception as e:
            error_message_str = str(e)
            self.video_collection.update_one({"_id": video_id}, {"$set": {"status": "failed", "progress_percent": 0, "error_message": error_message_str, "updated_at": datetime.now(timezone.utc).isoformat()}})
            return {"filename": filename, "status": "falha", "error": error_message_str, "video_id": video_id}
        finally:
            if cap and cap.isOpened(): cap.release()
            if out and out.isOpened(): out.release()

    async def process_multiple_files(self, files: List[UploadFile]) -> List[Dict[str, Any]]:
        tasks = [self.processFile(file) for file in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [{"status": "falha", "error": str(r)} if isinstance(r, Exception) else r for r in results]

    def __del__(self):
        if hasattr(self, "thread_pool"): self.thread_pool.shutdown(wait=False)