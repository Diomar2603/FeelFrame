import cv2 as cv
import os

class FaceAnalyzer:
    def __init__(self):

        input_width = 480
        input_height = 480
        yunet_model_path = os.path.join("back-end","app","utils", "yunet", "face_detection_yunet_2023mar.onnx")

        if not os.path.exists(yunet_model_path):
            raise FileNotFoundError(f"Modelo YuNet não encontrado em: {yunet_model_path}")

        self.detector = cv.FaceDetectorYN.create(
            yunet_model_path,
            "",
            (input_width, input_height),
            score_threshold=0.65,
            nms_threshold=0.3,
            top_k=1
        )

    def detect_faces(self, frame):
        # A imagem deve ter tamanho fixo definido na criação do detector
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else []
