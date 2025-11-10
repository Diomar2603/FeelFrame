import os
import cv2 as cv
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import math
from app.models.FrameAnalysis import (
    DimensaoComportamentalEnum, EmocaoEnum, EstimativaEngajamentoEnum, 
    OlharDirecaoEnum, PoseCabecaEnum, FrameAnalysis, HeadPose, GazeDirection
)
    
DEEPFACE_EMOTION_MAP = {
    'happy': EmocaoEnum.FELIZ,
    'sad': EmocaoEnum.TRISTE,
    'neutral': EmocaoEnum.NEUTRO,
    'surprise': EmocaoEnum.SURPRESO,
    'fear': EmocaoEnum.MEDO,
    'angry': EmocaoEnum.INDEFINIDO,
    'disgust': EmocaoEnum.INDEFINIDO,
}

class FaceAnalyzer:
    """
    Uma classe para analisar um frame de vídeo e extrair emoção, 
    pose da cabeça e direção do olhar.
    """
    def __init__(self, max_faces=1):
        """
        Inicializa os modelos MediaPipe FaceMesh.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.face_3d_model_points = np.array([
            [0.0, 0.0, 0.0],     # 1: Ponta do nariz
            [0.0, -330.0, -65.0],  # 152: Queixo
            [-225.0, 170.0, -135.0], # 263: Canto externo olho esquerdo
            [225.0, 170.0, -135.0],  # 33: Canto externo olho direito
            [-150.0, -150.0, -125.0],# 291: Canto esquerdo da boca
            [150.0, -150.0, -125.0]  # 61: Canto direito da boca
        ], dtype=np.float64)
        
        self.pose_landmark_indices = [1, 152, 263, 33, 291, 61]
        
        # Olho esquerdo
        self.IRIS_LEFT_INDICES = [474, 475, 476, 477]
        self.EYE_LEFT_OUTER = 33
        self.EYE_LEFT_INNER = 133
        self.EYE_LEFT_TOP = 159
        self.EYE_LEFT_BOTTOM = 145
        
        # Olho direito
        self.IRIS_RIGHT_INDICES = [469, 470, 471, 472]
        self.EYE_RIGHT_OUTER = 362
        self.EYE_RIGHT_INNER = 263
        self.EYE_RIGHT_TOP = 386
        self.EYE_RIGHT_BOTTOM = 374

        self.OLHOS_FECHADOS_THRESHOLD = 0.2

        # Detector 
        input_width = 480
        input_height = 480
        
        # Tenta encontrar o modelo .onnx de forma mais robusta
        base_dir = os.path.dirname(os.path.abspath(__file__))
        yunet_model_path = os.path.join(base_dir, "..", "utils", "yunet", "face_detection_yunet_2023mar.onnx")

        if not os.path.exists(yunet_model_path):
             yunet_model_path = os.path.join("back-end","app","utils", "yunet", "face_detection_yunet_2023mar.onnx")

        if not os.path.exists(yunet_model_path):
             yunet_model_path_alt = os.path.join(base_dir, "..", "..", "utils", "yunet", "face_detection_yunet_2023mar.onnx")
             if os.path.exists(yunet_model_path_alt):
                 yunet_model_path = yunet_model_path_alt
             else:
                raise FileNotFoundError(f"Modelo YuNet não encontrado. Tentativa 1: {yunet_model_path}")

        self.detector = cv.FaceDetectorYN.create(
            yunet_model_path,
            "",
            (input_width, input_height),
            score_threshold=0.65,
            nms_threshold=0.3,
            top_k=1
        )

    def detect_faces(self, frame):
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else []

    def _get_2d_coords(self, lm, shape):
        h, w = shape
        return int(lm.x * w), int(lm.y * h)

    def _get_emotion(self, frame, landmarks_list, shape):
        h, w = shape
        try:
            all_x = [int(lm.x * w) for lm in landmarks_list]
            all_y = [int(lm.y * h) for lm in landmarks_list]
            x, y = min(all_x), min(all_y)
            x_max, y_max = max(all_x), max(all_y)
            
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w_bbox = min(w - x, (x_max - x) + 2 * padding)
            h_bbox = min(h - y, (y_max - y) + 2 * padding)

            cropped_face = frame[y : y + h_bbox, x : x + w_bbox]

            if cropped_face.size == 0:
                return "Indeterminado (rosto pequeno)"

            analysis = DeepFace.analyze(
                img_path=cropped_face,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(analysis, list):
                analysis = analysis[0]

            return analysis['dominant_emotion']

        except Exception as e:
            return "Indeterminado"

    def _get_head_pose(self, landmarks_list, shape):
        h, w = shape
        
        image_points_2d = np.array(
            [self._get_2d_coords(landmarks_list[i], shape) for i in self.pose_landmark_indices],
            dtype="double"
        )
        
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv.solvePnP(
            self.face_3d_model_points,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv.SOLVEPNP_ITERATIVE
        )

        rotation_matrix, _ = cv.Rodrigues(rotation_vector)
        
        cameraMatrixOut, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ, eulerAngles = cv.decomposeProjectionMatrix(
            np.hstack((rotation_matrix, translation_vector))
        )

        yaw = eulerAngles[1][0]
        pitch = eulerAngles[0][0]
        roll = eulerAngles[2][0]

        if yaw > 15:
            pos_h = "Direita"
        elif yaw < -15:
            pos_h = "Esquerda"
        else:
            pos_h = "Frente"

        if pitch > 10:
            pos_v = "Baixo"
        elif pitch < -10:
            pos_v = "Cima"
        else:
            pos_v = "Frente"
            
        proximidade_z = translation_vector[2][0]

        return {
            "direcao_horizontal": pos_h,
            "direcao_vertical": pos_v,
            "proximidade_z": proximidade_z,
            "raw_pitch": pitch,
            "raw_yaw": yaw,
            "raw_roll": roll
        }
        
    def _get_gaze_direction(self, landmarks_list, shape):
        h, w = shape
        
        # Olho Esquerdo
        left_iris_coords = [landmarks_list[i] for i in self.IRIS_LEFT_INDICES]
        left_iris_center_x = np.mean([lm.x for lm in left_iris_coords])
        left_iris_center_y = np.mean([lm.y for lm in left_iris_coords])
        
        eye_left_outer_x = landmarks_list[self.EYE_LEFT_OUTER].x
        eye_left_inner_x = landmarks_list[self.EYE_LEFT_INNER].x
        eye_left_top_y = landmarks_list[self.EYE_LEFT_TOP].y
        eye_left_bottom_y = landmarks_list[self.EYE_LEFT_BOTTOM].y

        # Olho Direito
        right_iris_coords = [landmarks_list[i] for i in self.IRIS_RIGHT_INDICES]
        right_iris_center_x = np.mean([lm.x for lm in right_iris_coords])
        right_iris_center_y = np.mean([lm.y for lm in right_iris_coords])
        
        eye_right_outer_x = landmarks_list[self.EYE_RIGHT_OUTER].x
        eye_right_inner_x = landmarks_list[self.EYE_RIGHT_INNER].x
        eye_right_top_y = landmarks_list[self.EYE_RIGHT_TOP].y
        eye_right_bottom_y = landmarks_list[self.EYE_RIGHT_BOTTOM].y

        epsilon = 1e-6
        
        ratio_h_left = (left_iris_center_x - eye_left_outer_x) / (eye_left_inner_x - eye_left_outer_x + epsilon)
        ratio_h_right = (right_iris_center_x - eye_right_outer_x) / (eye_right_inner_x - eye_right_outer_x + epsilon)
        ratio_h_avg = (ratio_h_left + ratio_h_right) / 2.0

        ratio_v_left = (left_iris_center_y - eye_left_top_y) / (eye_left_bottom_y - eye_left_top_y + epsilon)
        ratio_v_right = (right_iris_center_y - eye_right_top_y) / (eye_right_bottom_y - eye_right_top_y + epsilon)
        ratio_v_avg = (ratio_v_left + ratio_v_right) / 2.0
        
        if ratio_h_avg > 0.65:
            gaze_h = "Direita"
        elif ratio_h_avg < 0.35:
            gaze_h = "Esquerda"
        else:
            gaze_h = "Frente"

        if ratio_v_avg > 0.60:
            gaze_v = "Baixo"
        elif ratio_v_avg < 0.40:
            gaze_v = "Cima"
        else:
            gaze_v = "Frente"
            
        return {
            "direcao_horizontal": gaze_h,
            "direcao_vertical": gaze_v,
            "raw_ratio_h": ratio_h_avg,
            "raw_ratio_v": ratio_v_avg
        }
    
    def analyze_frame(self, frame, timestamp_ms: int, video_id: str, frame_number: int) -> FrameAnalysis:
        h, w, _ = frame.shape
        shape_2d = (h, w)
        
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks_list = face_landmarks.landmark

            raw_emotion_str = self._get_emotion(frame, landmarks_list, shape_2d)
            raw_pose_dict = self._get_head_pose(landmarks_list, shape_2d)
            raw_gaze_dict = self._get_gaze_direction(landmarks_list, shape_2d)

            # Lógica de Olhos Fechados
            v_dist_left = abs(landmarks_list[self.EYE_LEFT_BOTTOM].y - landmarks_list[self.EYE_LEFT_TOP].y)
            v_dist_right = abs(landmarks_list[self.EYE_RIGHT_BOTTOM].y - landmarks_list[self.EYE_RIGHT_TOP].y)
            h_dist_left = abs(landmarks_list[self.EYE_LEFT_INNER].x - landmarks_list[self.EYE_LEFT_OUTER].x)
            h_dist_right = abs(landmarks_list[self.EYE_RIGHT_OUTER].x - landmarks_list[self.EYE_RIGHT_INNER].x)
            
            epsilon = 1e-6
            ear_left = v_dist_left / (h_dist_left + epsilon)
            ear_right = v_dist_right / (h_dist_right + epsilon)
            avg_ear = (ear_left + ear_right) / 2.0
            olhos_fechados = avg_ear < self.OLHOS_FECHADOS_THRESHOLD
            
            # Mapeamento para Enums e Dataclasses
            emocao_enum = DEEPFACE_EMOTION_MAP.get(raw_emotion_str, EmocaoEnum.INDEFINIDO)

            pose_obj = HeadPose(
                direcao_horizontal=raw_pose_dict['direcao_horizontal'],
                raw_yaw=raw_pose_dict['raw_yaw'],
                direcao_vertical=raw_pose_dict['direcao_vertical'],
                raw_pitch=raw_pose_dict['raw_pitch'],
                proximidade_z=raw_pose_dict['proximidade_z']
            )
            gaze_obj = GazeDirection(
                direcao_horizontal=raw_gaze_dict['direcao_horizontal'],
                raw_ratio_h=raw_gaze_dict['raw_ratio_h'],
                direcao_vertical=raw_gaze_dict['direcao_vertical'],
                raw_ratio_v=raw_gaze_dict['raw_ratio_v']
            )

            if pose_obj.direcao_vertical == "Baixo":
                pose_enum = PoseCabecaEnum.BAIXO
            elif pose_obj.direcao_horizontal != "Frente":
                pose_enum = PoseCabecaEnum.LADOS
            elif pose_obj.direcao_vertical == "Frente":
                pose_enum = PoseCabecaEnum.FRENTE
            else:
                pose_enum = PoseCabecaEnum.INDEFINIDO

            if gaze_obj.direcao_horizontal == "Frente":
                olhar_enum = OlharDirecaoEnum.FRENTE
            else:
                olhar_enum = OlharDirecaoEnum.LADOS

            dimensao_enum = self._calcular_dimensao_comportamental(
                pose=pose_enum,
                olhar=olhar_enum,
                olhos_fechados=olhos_fechados
            )
            
            estimativa_enum = self._calcular_estimativa_engajamento(
                dimensao=dimensao_enum,
                emocao=emocao_enum
            )

            return FrameAnalysis(
                video_id=video_id,
                timestamp_ms=timestamp_ms,
                frame_number=frame_number,
                emocao=emocao_enum.value,
                pose_cabeca=pose_obj,
                olhar=gaze_obj,
                dimensao_comportamental=dimensao_enum,
                estimativa_engajamento=estimativa_enum
            )
        
        else:
            # CASO: NENHUM ROSTO DETECTADO
            default_pose = HeadPose(
                direcao_horizontal="Indefinido", raw_yaw=0.0,
                direcao_vertical="Indefinido", raw_pitch=0.0,
                proximidade_z=0.0
            )
            default_gaze = GazeDirection(
                direcao_horizontal="Indefinido", raw_ratio_h=0.0,
                direcao_vertical="Indefinido", raw_ratio_v=0.0
            )

            return FrameAnalysis(
                video_id=video_id,
                timestamp_ms=timestamp_ms,
                frame_number=frame_number,
                emocao=EmocaoEnum.INDEFINIDO.value,
                pose_cabeca=default_pose,
                olhar=default_gaze,
                dimensao_comportamental=DimensaoComportamentalEnum.INDEFINIDO,
                estimativa_engajamento=EstimativaEngajamentoEnum.INDEFINIDO
            )

    def _calcular_dimensao_comportamental(
        self, 
        pose: PoseCabecaEnum, 
        olhar: OlharDirecaoEnum, 
        olhos_fechados: bool
    ) -> DimensaoComportamentalEnum:
        
        if pose == PoseCabecaEnum.BAIXO:
            if olhos_fechados:
                return DimensaoComportamentalEnum.DISTRAIDO
            else:
                if olhar == OlharDirecaoEnum.FRENTE:
                    return DimensaoComportamentalEnum.CONCENTRADO
                elif olhar == OlharDirecaoEnum.LADOS:
                    return DimensaoComportamentalEnum.INDEFINIDO_DISTRAIDO
                else:
                    return DimensaoComportamentalEnum.CONCENTRADO

        elif pose == PoseCabecaEnum.LADOS:
            if olhos_fechados:
                return DimensaoComportamentalEnum.DISTRAIDO
            else:
                if olhar == OlharDirecaoEnum.FRENTE:
                    return DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO
                elif olhar == OlharDirecaoEnum.LADOS:
                    return DimensaoComportamentalEnum.DISTRAIDO
                else:
                    return DimensaoComportamentalEnum.DISTRAIDO
                
        elif pose == PoseCabecaEnum.FRENTE:
            if olhos_fechados:
                return DimensaoComportamentalEnum.DISTRAIDO
            else:
                if olhar == OlharDirecaoEnum.FRENTE:
                    return DimensaoComportamentalEnum.CONCENTRADO
                elif olhar == OlharDirecaoEnum.LADOS:
                    return DimensaoComportamentalEnum.INDEFINIDO_DISTRAIDO
                else:
                    return DimensaoComportamentalEnum.CONCENTRADO
        
        return DimensaoComportamentalEnum.INDEFINIDO

    
    def _calcular_estimativa_engajamento(
        self, 
        dimensao: DimensaoComportamentalEnum, 
        emocao: EmocaoEnum
    ) -> EstimativaEngajamentoEnum:
        
        if dimensao == DimensaoComportamentalEnum.CONCENTRADO:
            if emocao in [EmocaoEnum.FELIZ, EmocaoEnum.SURPRESO]:
                return EstimativaEngajamentoEnum.ALTAMENTE_ENGAJADO
            if emocao == EmocaoEnum.NEUTRO:
                return EstimativaEngajamentoEnum.ENGAJADO
            if emocao == EmocaoEnum.MEDO:
                return EstimativaEngajamentoEnum.DESENGAJADO
            if emocao == EmocaoEnum.TRISTE:
                return EstimativaEngajamentoEnum.ALTAMENTE_DESENGAJADO

        if dimensao == DimensaoComportamentalEnum.INDEFINIDO_DISTRAIDO:
            if emocao in [EmocaoEnum.FELIZ, EmocaoEnum.SURPRESO]:
                return EstimativaEngajamentoEnum.ENGAJADO
            if emocao == EmocaoEnum.NEUTRO:
                return EstimativaEngajamentoEnum.DESENGAJADO
            if emocao in [EmocaoEnum.MEDO, EmocaoEnum.TRISTE]:
                return EstimativaEngajamentoEnum.ALTAMENTE_DESENGAJADO
        
        if dimensao == DimensaoComportamentalEnum.DISTRAIDO:
            if emocao in [EmocaoEnum.NEUTRO, EmocaoEnum.FELIZ, EmocaoEnum.SURPRESO]:
                return EstimativaEngajamentoEnum.DESENGAJADO
            if emocao in [EmocaoEnum.MEDO, EmocaoEnum.TRISTE]:
                return EstimativaEngajamentoEnum.ALTAMENTE_DESENGAJADO

        if dimensao == DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO:
            if emocao in [EmocaoEnum.NEUTRO, EmocaoEnum.FELIZ, EmocaoEnum.SURPRESO]:
                return EstimativaEngajamentoEnum.ENGAJADO
            if emocao == EmocaoEnum.MEDO:
                return EstimativaEngajamentoEnum.DESENGAJADO
            if emocao == EmocaoEnum.TRISTE:
                return EstimativaEngajamentoEnum.ALTAMENTE_DESENGAJADO

        return EstimativaEngajamentoEnum.INDEFINIDO