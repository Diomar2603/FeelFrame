import os
import cv2 as cv
import numpy as np
from fastapi import UploadFile
from app.services.faceAnalyzer import FaceAnalyzer

class VideoService:
    def __init__(self, filePath: str):
        self.base_operation_path = filePath
        self.original_videos_dir = os.path.join(self.base_operation_path, "videos_originais")
        self.fixed_frame_videos_dir = os.path.join(self.base_operation_path, "videos_quadro_fixo")

        os.makedirs(self.original_videos_dir, exist_ok=True)
        os.makedirs(self.fixed_frame_videos_dir, exist_ok=True)

        self.OUTPUT_WIDTH = 1080
        self.OUTPUT_HEIGHT = 840

        self.face_analyzer = FaceAnalyzer()

    @staticmethod
    def frameResize(frame, width=840, height=480):
        return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

    async def saveFile(self, file: UploadFile):
        file_location = os.path.join(self.original_videos_dir, file.filename)
        try:
            contents = await file.read()
            with open(file_location, "wb+") as f:
                f.write(contents)
        except Exception as e:
            raise Exception(f"Erro ao salvar vídeo: {str(e)}")
        return file_location

    async def processFile(self, file: UploadFile):
        output_file_location = ""
        original_file_location = ""
        fixed_crop_rect = None
        cap = out = None

        black_frame = np.zeros((self.OUTPUT_HEIGHT, self.OUTPUT_WIDTH, 3), dtype=np.uint8)

        try:
            original_file_location = await self.saveFile(file)
            cap = cv.VideoCapture(original_file_location)
            if not cap.isOpened():
                raise Exception(f"Erro ao abrir vídeo: {original_file_location}")

            fps = cap.get(cv.CAP_PROP_FPS) or 25
            output_filename = f"quadro_fixo_{file.filename}"
            if not output_filename.lower().endswith((".mp4", ".avi", ".mov")):
                output_filename += ".mp4"
            output_file_location = os.path.join(self.fixed_frame_videos_dir, output_filename)

            fourcc = cv.VideoWriter_fourcc(*'avc1')
            out = cv.VideoWriter(output_file_location, fourcc, fps, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT))
            if not out.isOpened():
                print(f"Aviso: Falha ao abrir VideoWriter com 'avc1'. Tentando com 'mp4v' para {output_filename}.")
                fourcc = cv.VideoWriter_fourcc(*'mp4v') 
                out = cv.VideoWriter(output_file_location, fourcc, fps, (self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT))
                if not out.isOpened():
                    raise Exception(f"Erro ao criar VideoWriter para {output_filename} com codecs 'avc1' e 'mp4v'.")


            input_frame_count = output_frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                input_frame_count += 1

                frame_resized = self.frameResize(frame, self.OUTPUT_WIDTH, self.OUTPUT_HEIGHT)

                if fixed_crop_rect is None:
                    faces = self.face_analyzer.detect_faces(frame_resized)
                    if len(faces) > 0:
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

                out.write(frame_out)
                output_frame_count += 1

            final_message = "Vídeo processado com sucesso."
            if input_frame_count == 0:
                if os.path.exists(output_file_location):
                    os.remove(output_file_location)
                raise Exception("Vídeo de entrada está vazio.")
            if fixed_crop_rect is None:
                final_message = "Nenhum rosto detectado. Quadros pretos foram usados."

            return {
                "message": final_message,
                "original_uploaded_filename": file.filename,
                "saved_original_location": original_file_location,
                "fixed_frame_video_filename": output_filename,
                "fixed_frame_video_location": output_file_location,
                "input_frames_read": input_frame_count,
                "output_frames_written": output_frame_count,
                "output_video_dimensions": f"{self.OUTPUT_WIDTH}x{self.OUTPUT_HEIGHT}",
                "fixed_crop_rect_in_source": str(fixed_crop_rect) if fixed_crop_rect else "N/A"
            }

        except Exception as e:
            if out and out.isOpened(): out.release()
            out = None 

            if output_file_location and os.path.exists(output_file_location):
                try:
                    os.remove(output_file_location)
                except Exception as remove_error:
                    print(f"Erro ao tentar remover o arquivo de saída após falha: {remove_error}")
            return {
                "filename": file.filename,
                "status": "falha",
                "error": str(e)
            }
        finally:
            if cap and cap.isOpened(): cap.release()
            if out and out.isOpened(): out.release()