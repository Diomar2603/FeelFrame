"""
videoService.py — REFATORADO para operação assíncrona e não bloqueante.

MUDANÇAS PRINCIPAIS vs versão original:
  1. `_process_video_sync()` — toda a lógica CPU-bound (loop de frames, análise
     de face, escrita de vídeo) foi movida para este método síncrono puro.
     Ele roda dentro do ThreadPoolExecutor via `run_in_executor`, liberando a
     event loop para atender outras requisições enquanto o vídeo é processado.

  2. `processFile()` e `processFile_by_id()` agora são corrotinas finas que:
       a) gravam o arquivo em disco (I/O via thread-pool),
       b) disparam `_process_video_sync` no pool sem bloquear a event loop,
       c) atualizam MongoDB com o resultado.

  3. ThreadPoolExecutor com `max_workers` configurável via variável de ambiente
     `VIDEO_POOL_WORKERS` (padrão 2).  Para tarefas CPU-bound em Python com GIL,
     2-4 workers é suficiente; acima disso só aumenta contenção sem ganho de
     throughput em processamento de vídeo + DeepFace.

  4. Upload para Firebase (`upload_video`) deslocado para `asyncio.to_thread()`
     pois é I/O puro (rede), não CPU — não precisa de worker do pool de vídeo.

  5. `_salvar_frame_destaque` e `_update_progress` continuam síncronos;
     chamados de dentro do worker thread, não da event loop.

  6. Exceções dentro do worker são capturadas pela `Future` retornada pelo
     `run_in_executor` e relançadas no `await`, garantindo que o processo
     principal nunca trave silenciosamente.
"""

import os
import asyncio
import cv2 as cv
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pymongo.collection import Collection
from fastapi import UploadFile
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

from app.models.VideoMetadata import VideoMetadata
from app.models.StorageSource import StorageSource
from app.services.faceAnalyzer import FaceAnalyzer
from app.models.FrameAnalysis import FrameAnalysis, EstimativaEngajamentoEnum
from app.utils.DatabaseConfig import DatabaseConfig
from app.services.interfaces.IStorageService import IStorageService


# Número de workers para tarefas CPU-bound (processamento de vídeo).
# Ajuste conforme vCPUs disponíveis no servidor; 2 é seguro em Spark/free-tier.
_VIDEO_POOL_WORKERS = int(os.getenv("VIDEO_POOL_WORKERS", "2"))


class VideoService:

    def __init__(self, filePath: str, db_config: DatabaseConfig,
                 storage: IStorageService):
        self.db = db_config.client[db_config.db_name]
        self.video_collection: Collection       = self.db["videos"]
        self.frame_analysis_collection: Collection = self.db["frame_analysis"]

        self.original_videos_dir       = os.path.join(filePath, "videos_originais")
        self.fixed_frame_videos_dir    = os.path.join(
            filePath, os.getenv("FIXED_FRAME_VIDEOS_DIR", "videos_quadro_fixo")
        )
        self.fluxo_frames_dir          = os.path.join(filePath, "cenas_fluxo")
        self.desengajamento_frames_dir = os.path.join(filePath, "cenas_desengajamento")

        self.OUTPUT_WIDTH  = int(os.getenv("OUTPUT_WIDTH",  "480"))
        self.OUTPUT_HEIGHT = int(os.getenv("OUTPUT_HEIGHT", "480"))
        self.frame_analysis_skip_rate = 1

        for d in (self.fixed_frame_videos_dir, self.original_videos_dir,
                  self.fluxo_frames_dir, self.desengajamento_frames_dir):
            os.makedirs(d, exist_ok=True)

        # Pool dedicado a tarefas CPU-bound (processamento de vídeo + ML).
        # max_workers limitado para evitar esgotamento de memória: cada worker
        # carrega um FaceAnalyzer completo (MediaPipe + YuNet + DeepFace).
        self.thread_pool = ThreadPoolExecutor(
            max_workers=_VIDEO_POOL_WORKERS,
            thread_name_prefix="video_worker"
        )

        self.storage: IStorageService = storage

    # ------------------------------------------------------------------
    # I/O de arquivos (chamados de corrotinas)
    # ------------------------------------------------------------------

    async def saveFile(self, file: UploadFile) -> str:
        safe_filename = os.path.basename(file.filename)
        location = os.path.join(self.original_videos_dir, safe_filename)
        # `file.read()` já é awaitable no FastAPI; a escrita em disco é leve.
        content = await file.read()
        # Delegamos a escrita em disco para uma thread para não bloquear.
        await asyncio.to_thread(self._write_bytes, location, content)
        return location

    async def saveFileFromBytes(self, file_bytes: bytes, filename: str) -> str:
        safe_filename = os.path.basename(filename)
        location = os.path.join(self.original_videos_dir, safe_filename)
        await asyncio.to_thread(self._write_bytes, location, file_bytes)
        return location

    @staticmethod
    def _write_bytes(path: str, data: bytes) -> None:
        """Escrita síncrona — sempre chamada via asyncio.to_thread."""
        with open(path, "wb") as f:
            f.write(data)

    # ------------------------------------------------------------------
    # Helpers síncronos (chamados dentro do worker thread)
    # ------------------------------------------------------------------

    def frameResize(self, frame, width, height):
        return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

    def _apply_crop(self, frame, face_rect: tuple,
                    pad_top_factor: float, pad_bottom_factor: float,
                    pad_side_factor: float) -> np.ndarray:
        """Lógica central de recorte com clipping e padding pretos."""
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = face_rect

        pad_top    = int(h * pad_top_factor)
        pad_bottom = int(h * pad_bottom_factor)
        pad_side   = int(w * pad_side_factor)

        x1 = x - pad_side;  y1 = y - pad_top
        x2 = x + w + pad_side;  y2 = y + h + pad_bottom

        # Força proporção quadrada para preservar o aspect ratio do OUTPUT
        crop_w = x2 - x1;  crop_h = y2 - y1
        if crop_h > crop_w:
            diff = crop_h - crop_w
            x1 -= diff // 2;  x2 += diff - diff // 2
        elif crop_w > crop_h:
            diff = crop_w - crop_h
            y1 -= diff // 2;  y2 += diff - diff // 2

        # Calcula padding preto para áreas fora do frame original
        pad_left_fill   = max(0, -x1)
        pad_top_fill    = max(0, -y1)
        pad_right_fill  = max(0, x2 - frame_w)
        pad_bottom_fill = max(0, y2 - frame_h)

        x1c = max(0, x1);  y1c = max(0, y1)
        x2c = min(frame_w, x2);  y2c = min(frame_h, y2)
        cropped = frame[y1c:y2c, x1c:x2c]

        if any([pad_left_fill, pad_top_fill, pad_right_fill, pad_bottom_fill]):
            cropped = cv.copyMakeBorder(
                cropped,
                pad_top_fill, pad_bottom_fill,
                pad_left_fill, pad_right_fill,
                cv.BORDER_CONSTANT, value=(0, 0, 0)
            )

        return cv.resize(cropped, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT),
                         interpolation=cv.INTER_LANCZOS4)

    def _crop_face_for_analysis(self, frame, face_rect: tuple) -> np.ndarray:
        """
        Recorte justo ao redor do rosto — usado EXCLUSIVAMENTE para análise
        (DeepFace / YuNet / MediaPipe).  Mantém o rosto grande no frame para
        preservar a qualidade de detecção de emoções.
        """
        return self._apply_crop(frame, face_rect,
                                pad_top_factor=0.80,
                                pad_bottom_factor=0.30,
                                pad_side_factor=0.40)

    def _crop_upper_body(self, frame, face_rect: tuple) -> np.ndarray:
        """
        Recorte de tronco superior — usado para o VÍDEO DE SAÍDA.
        Expande 2× a altura do rosto para baixo (tórax/ombros)
        e 1.5× a largura para os lados.
        O clipping garante que o recorte nunca saia das dimensões originais.
        """
        return self._apply_crop(frame, face_rect,
                                pad_top_factor=0.80,
                                pad_bottom_factor=2.00,
                                pad_side_factor=1.50)

    def _update_progress(self, video_id: str, percent: int,
                         message: Optional[str] = None) -> None:
        """
        Atualiza progresso no MongoDB.
        Chamado de dentro do worker thread — pymongo é thread-safe por padrão.
        """
        update: Dict[str, Any] = {
            "progress_percent": percent,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        if message:
            update["processing_message"] = message
        self.video_collection.update_one({"_id": video_id}, {"$set": update})

    def _salvar_frame_destaque(self, frame_original, frame_number, timestamp_ms,
                               video_id, analysis, tipo: str) -> Optional[dict]:
        """Upload de frame de destaque para Firebase. Chamado do worker thread."""
        try:
            timestamp_str = f"{timestamp_ms // 1000}s_{timestamp_ms % 1000:03d}ms"
            if tipo == "fluxo":
                filename = f"fluxo_{video_id}_{frame_number:06d}_{timestamp_str}.jpg"
                folder   = "feelframe/frames_destaque/fluxo"
            elif tipo == "desengajamento":
                filename = f"desengajamento_{video_id}_{frame_number:06d}_{timestamp_str}.jpg"
                folder   = "feelframe/frames_destaque/desengajamento"
            else:
                return None

            success, buffer = cv.imencode(".jpg", frame_original,
                                          [cv.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                raise RuntimeError(f"Falha ao codificar frame {frame_number} como JPEG.")

            result = self.storage.upload_image_from_bytes(
                buffer.tobytes(), filename, folder
            )

            return {
                "frame_number": frame_number,
                "timestamp_ms": timestamp_ms,
                "firebase_url": result["secure_url"],
                "tipo":         tipo,
                "emocao":       analysis.emocao,
                "engajamento":  analysis.estimativa_engajamento.value,
                "dimensao":     analysis.dimensao_comportamental.value,
                "confidence":   getattr(analysis, "emotion_confidence", 0.0),
            }
        except Exception as e:
            print(f"Erro ao fazer upload do frame de destaque ({tipo}): {e}")
            return None

    def _salvar_metadados_destaque(self, video_id: str,
                                   frames_destaque_info: list) -> None:
        try:
            if frames_destaque_info:
                destaque_collection = self.db["frames_destaque"]
                for frame_info in frames_destaque_info:
                    frame_info["video_id"]   = video_id
                    frame_info["created_at"] = datetime.now(timezone.utc).isoformat()
                destaque_collection.insert_many(frames_destaque_info)
        except Exception as e:
            print(f"Erro ao salvar metadados de destaque: {e}")

    def _get_final_message(self, fixed_crop_rect, frames_destaque_info) -> str:
        msg = "Vídeo processado com sucesso."
        if fixed_crop_rect is None:
            msg = "Nenhum rosto detectado com confiança. Quadros pretos foram usados."
        if frames_destaque_info:
            fluxo_count = len([f for f in frames_destaque_info if f["tipo"] == "fluxo"])
            deseng_count = len([f for f in frames_destaque_info if f["tipo"] == "desengajamento"])
            msg += f" {fluxo_count} cenas de fluxo e {deseng_count} de desengajamento detectadas."
        return msg

    def _cleanup_local_files(self, *paths: str) -> None:
        for path in paths:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Aviso: não foi possível remover {path}: {e}")

    # ------------------------------------------------------------------
    # Núcleo do processamento — SÍNCRONO, roda no ThreadPoolExecutor
    # ------------------------------------------------------------------

    @staticmethod
    def _user_folder(base: str, user_id: Optional[str]) -> str:
        """Retorna caminho de storage organizado por usuário quando disponível."""
        if user_id:
            return f"feelframe/users/{user_id}/{base}"
        return f"feelframe/{base}"

    def _process_video_sync(self, original_file_location: str,
                            video_id: str,
                            user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Loop principal de processamento de vídeo.

        DEVE ser chamado apenas via `loop.run_in_executor(self.thread_pool, ...)`.
        Nunca chame diretamente de uma corrotina — bloquearia a event loop.

        Retorna um dict com os metadados finais do processamento.
        Exceções propagam normalmente para o chamador (Future do executor).
        """
        cap = out = None
        output_file_location = ""
        analysis_results_list: list = []
        frames_destaque_info: list  = []
        fixed_crop_rect             = None

        # Instância EXCLUSIVA por thread — FaceAnalyzer não é thread-safe entre instâncias.
        local_analyzer = FaceAnalyzer()
        black_frame    = np.zeros((self.OUTPUT_HEIGHT, self.OUTPUT_WIDTH, 3), dtype=np.uint8)

        try:
            cap = cv.VideoCapture(original_file_location)
            if not cap.isOpened():
                raise RuntimeError(f"Não foi possível abrir vídeo: {original_file_location}")

            fps             = cap.get(cv.CAP_PROP_FPS) or 25.0
            original_width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
            total_frames    = int(cap.get(cv.CAP_PROP_FRAME_COUNT)) or 1
            original_filesize_bytes = os.path.getsize(original_file_location)

            self._update_progress(video_id, 2, "Registrando metadados...")
            self.video_collection.update_one(
                {"_id": video_id},
                {"$set": {
                    "status":                   "processing",
                    "storage_source":           self.storage.source.value,
                    "original_filepath":        original_file_location,
                    "original_filesize_bytes":  original_filesize_bytes,
                    "original_width":           original_width,
                    "original_height":          original_height,
                    "fps":                      fps,
                    "processed_width":          self.OUTPUT_WIDTH,
                    "processed_height":         self.OUTPUT_HEIGHT,
                    "updated_at":               datetime.now(timezone.utc).isoformat(),
                }},
            )

            # Upload do vídeo original (I/O de rede — síncrono dentro do worker)
            self._update_progress(video_id, 5, "Enviando vídeo original para a nuvem...")
            original_cloud = self.storage.upload_video(
                file_path=original_file_location,
                public_id=f"{video_id}_original",
                folder=self._user_folder("videos_originais", user_id),
            )
            self.video_collection.update_one(
                {"_id": video_id},
                {"$set": {
                    "original_url":      original_cloud["secure_url"],
                    "original_filepath": original_cloud["secure_url"],
                }},
            )

            # Prepara saída de vídeo processado
            safe_original_filename = os.path.basename(original_file_location)
            output_filename = f"quadro_fixo_{safe_original_filename}"
            if not output_filename.lower().endswith((".mp4", ".avi", ".mov")):
                output_filename += ".mp4"
            output_file_location = os.path.join(self.fixed_frame_videos_dir, output_filename)

            fourcc = cv.VideoWriter_fourcc(*"avc1")
            out    = cv.VideoWriter(output_file_location, fourcc, fps,
                                    (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT))

            input_frame_count = output_frame_count = 0
            self._update_progress(video_id, 10, "Processando frames com análise máxima...")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                input_frame_count += 1
                timestamp_ms = int(cap.get(cv.CAP_PROP_POS_MSEC))

                progress_value = 10 + int((input_frame_count / total_frames) * 75)
                if input_frame_count % max(1, total_frames // 20) == 0:
                    self._update_progress(
                        video_id, progress_value,
                        f"Analisando quadros em alta precisão... {progress_value}%"
                    )

                # Detecção de rosto fixo (primeiro frame ou rect ainda nulo)
                if input_frame_count == 1 or fixed_crop_rect is None:
                    detected_rect = local_analyzer.detect_face_rect(frame)
                    if detected_rect:
                        fixed_crop_rect = detected_rect

                if fixed_crop_rect:
                    # frame_out  → tronco superior (escrito no vídeo de saída)
                    # frame_analysis → rosto em close (qualidade máxima para DeepFace/YuNet)
                    frame_out      = self._crop_upper_body(frame, fixed_crop_rect)
                    frame_analysis = self._crop_face_for_analysis(frame, fixed_crop_rect)
                else:
                    frame_out = frame_analysis = self.frameResize(
                        frame, self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT
                    )

                if input_frame_count % self.frame_analysis_skip_rate == 0:
                    try:
                        analysis_result_obj: FrameAnalysis = local_analyzer.analyze(
                            frame=frame_analysis,
                            video_id=video_id,
                            timestamp_ms=timestamp_ms,
                            frame_number=input_frame_count,
                        )

                        if getattr(analysis_result_obj, "estado_fluxo", False):
                            info = self._salvar_frame_destaque(
                                frame, input_frame_count, timestamp_ms,
                                video_id, analysis_result_obj, "fluxo"
                            )
                            if info:
                                frames_destaque_info.append(info)
                        elif (analysis_result_obj.estimativa_engajamento
                              == EstimativaEngajamentoEnum.ALTAMENTE_DESENGAJADO):
                            info = self._salvar_frame_destaque(
                                frame, input_frame_count, timestamp_ms,
                                video_id, analysis_result_obj, "desengajamento"
                            )
                            if info:
                                frames_destaque_info.append(info)

                        analysis_results_list.append(analysis_result_obj.to_dict())

                        # Batch insert: evita acúmulo excessivo em memória
                        if len(analysis_results_list) >= 150:
                            self.frame_analysis_collection.insert_many(analysis_results_list)
                            analysis_results_list = []

                    except Exception as e_analyze:
                        # Falha em um frame não aborta o processamento completo.
                        print(f"Aviso: falha ao analisar frame {input_frame_count}: {e_analyze}")

                out.write(frame_out)
                output_frame_count += 1

            # Flush de resultados restantes
            if analysis_results_list:
                self.frame_analysis_collection.insert_many(analysis_results_list)
            if frames_destaque_info:
                self._salvar_metadados_destaque(video_id, frames_destaque_info)

            cap.release();  out.release()
            cap = out = None

            # Upload do vídeo processado
            self._update_progress(video_id, 85, "Enviando vídeo processado para a nuvem...")
            processed_cloud = self.storage.upload_video(
                file_path=output_file_location,
                public_id=f"{video_id}_processado",
                folder=self._user_folder("videos_processados", user_id),
            )

            duration_seconds  = (input_frame_count / fps) if fps > 0 else 0.0
            final_message     = self._get_final_message(fixed_crop_rect, frames_destaque_info)
            fluxo_count       = len([f for f in frames_destaque_info if f["tipo"] == "fluxo"])
            desengajamento_count = len([f for f in frames_destaque_info if f["tipo"] == "desengajamento"])

            self.video_collection.update_one(
                {"_id": video_id},
                {"$set": {
                    "status":                    "success",
                    "storage_source":            self.storage.source.value,
                    "progress_percent":          100,
                    "processing_message":        final_message,
                    "processed_url":             processed_cloud["secure_url"],
                    "processed_filepath":        processed_cloud["secure_url"],
                    "frame_count":               input_frame_count,
                    "duration_seconds":          duration_seconds,
                    "fluxo_frames_count":        fluxo_count,
                    "desengajamento_frames_count": desengajamento_count,
                    "fixed_crop_rect_in_source": str(fixed_crop_rect) if fixed_crop_rect else "N/A",
                    "updated_at":                datetime.now(timezone.utc).isoformat(),
                }},
            )

            self._cleanup_local_files(original_file_location, output_file_location)

            return {
                "video_id":                    video_id,
                "original_url":                original_cloud["secure_url"],
                "processed_url":               processed_cloud["secure_url"],
                "input_frames_read":           input_frame_count,
                "fluxo_frames_captured":       fluxo_count,
                "desengajamento_frames_captured": desengajamento_count,
                "message":                     final_message,
            }

        except Exception:
            # Re-lança para que o Future propagule o erro ao chamador assíncrono.
            raise

        finally:
            if cap and cap.isOpened():
                cap.release()
            if out and out.isOpened():
                out.release()

    # ------------------------------------------------------------------
    # Interface assíncrona pública
    # ------------------------------------------------------------------

    async def processFile(self, file: UploadFile) -> Dict[str, Any]:
        """
        Recebe um UploadFile do FastAPI, salva em disco e processa o vídeo
        de forma totalmente não bloqueante.

        O processamento pesado ocorre no ThreadPoolExecutor — a event loop
        fica livre para atender outras requisições durante todo o processo.
        """
        original_file_location = ""
        video_id               = None

        try:
            # 1. Salva arquivo (I/O delegado para thread)
            original_file_location = await self.saveFile(file)

            safe_original_filename  = os.path.basename(file.filename)
            original_filesize_bytes = os.path.getsize(original_file_location)

            # 2. Cria registro no MongoDB para tracking de progresso
            video_meta = VideoMetadata(
                original_filename=safe_original_filename,
                original_filepath=original_file_location,
                original_filesize_bytes=original_filesize_bytes,
                original_width=0, original_height=0, fps=0,
                processed_width=self.OUTPUT_WIDTH,
                processed_height=self.OUTPUT_HEIGHT,
                storage_source=self.storage.source,
            )
            self.video_collection.insert_one(video_meta.to_dict())
            video_id = video_meta._id

            self._update_progress(video_id, 0, "Iniciando processamento...")

            # 3. Delega o processamento CPU-bound para o pool — não bloqueia event loop
            loop   = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                self._process_video_sync,   # função síncrona
                original_file_location,
                video_id,
            )

            result["original_filename"] = safe_original_filename
            return result

        except Exception as e:
            error_msg = str(e)
            print(f"[VideoService.processFile] Erro: {error_msg}")
            if video_id:
                self.video_collection.update_one(
                    {"_id": video_id},
                    {"$set": {
                        "status":          "failed",
                        "progress_percent": 0,
                        "error_message":   error_msg,
                        "updated_at":      datetime.now(timezone.utc).isoformat(),
                    }},
                )
            return {
                "filename": getattr(file, "filename", ""),
                "status":   "falha",
                "error":    error_msg,
                "video_id": video_id,
            }

    async def processFile_by_id(self, file_bytes: bytes, filename: str,
                                video_id: str,
                                user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Processa vídeo a partir de bytes (sem criar VideoMetadata novo) —
        usado quando o registro já existe no banco (ex.: reprocessamento).
        """
        original_file_location = ""

        try:
            self._update_progress(video_id, 0, "Salvando arquivo...")
            original_file_location = await self.saveFileFromBytes(file_bytes, filename)

            # Preenche metadados básicos no registro existente
            safe_original_filename = os.path.basename(filename)
            self.video_collection.update_one(
                {"_id": video_id},
                {"$set": {
                    "original_filename": safe_original_filename,
                    "updated_at":        datetime.now(timezone.utc).isoformat(),
                }},
            )

            loop   = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                self.thread_pool,
                self._process_video_sync,
                original_file_location,
                video_id,
                user_id,
            )

            result["original_filename"] = safe_original_filename
            return result

        except Exception as e:
            error_msg = str(e)
            print(f"[VideoService.processFile_by_id] Erro: {error_msg}")
            self.video_collection.update_one(
                {"_id": video_id},
                {"$set": {
                    "status":          "failed",
                    "progress_percent": 0,
                    "error_message":   error_msg,
                    "updated_at":      datetime.now(timezone.utc).isoformat(),
                }},
            )
            return {
                "filename": filename,
                "status":   "falha",
                "error":    error_msg,
                "video_id": video_id,
            }

    async def process_multiple_files(self, files: List[UploadFile]) -> List[Dict[str, Any]]:
        """
        Processa múltiplos vídeos concorrentemente.
        O pool limita o paralelismo real; asyncio.gather apenas agenda as corrotinas.
        """
        tasks   = [self.processFile(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            {"status": "falha", "error": str(r)}
            if isinstance(r, Exception) else r
            for r in results
        ]

    def __del__(self):
        if hasattr(self, "thread_pool"):
            self.thread_pool.shutdown(wait=False)
