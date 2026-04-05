import os
import cv2 as cv
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import math
from collections import deque
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
    Classe ajustada para evitar comportamentos indefinidos puros e 
    extrair emoções secundárias quando o neutro predomina.
    """
    
    def __init__(self, max_faces=1):
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Modelo 3D para pose
        self.face_3d_model_points = np.array([
            [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
        ], dtype=np.float64)
        
        self.pose_landmark_indices = [1, 152, 263, 33, 291, 61]
        
        # Landmarks para olhos
        self.IRIS_LEFT_INDICES = [474, 475, 476, 477]
        self.EYE_LEFT_OUTER = 33
        self.EYE_LEFT_INNER = 133
        self.EYE_LEFT_TOP = 159
        self.EYE_LEFT_BOTTOM = 145
        
        self.IRIS_RIGHT_INDICES = [469, 470, 471, 472]
        self.EYE_RIGHT_OUTER = 362
        self.EYE_RIGHT_INNER = 263
        self.EYE_RIGHT_TOP = 386
        self.EYE_RIGHT_BOTTOM = 374

        self.OLHOS_FECHADOS_THRESHOLD = 0.2
        
        # Cache e histórico
        self._detector = None
        self.emotion_history = deque(maxlen=10)
        self.pose_history = deque(maxlen=5)
        
        # Parâmetros
        self.SAD_TO_NEUTRAL_THRESHOLD = 25
        self.MAX_SAD_CONFIDENCE = 0.5 
        # Threshold para aceitar a segunda emoção se a primeira for neutra
        self.SECOND_EMOTION_THRESHOLD = 15.0 

        self._init_detector()

    def _init_detector(self):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            yunet_model_path = os.path.join(base_dir, "..", "utils", "yunet", "face_detection_yunet_2023mar.onnx")

            if not os.path.exists(yunet_model_path):
                yunet_model_path = os.path.join("back-end", "app", "utils", "yunet", "face_detection_yunet_2023mar.onnx")

            if not os.path.exists(yunet_model_path):
                yunet_model_path_alt = os.path.join(base_dir, "..", "..", "utils", "yunet", "face_detection_yunet_2023mar.onnx")
                if os.path.exists(yunet_model_path_alt):
                    yunet_model_path = yunet_model_path_alt
                else:
                    raise FileNotFoundError("Modelo YuNet não encontrado")

            self._detector = cv.FaceDetectorYN.create(
                yunet_model_path, "", (480, 480),
                score_threshold=0.65, nms_threshold=0.3, top_k=1
            )
        except Exception as e:
            print(f"Aviso: Não foi possível carregar YuNet: {e}")
            self._detector = None

    @property
    def detector(self):
        return self._detector

    def detect_faces(self, frame):
        if self.detector is None:
            return []
        self.detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.detector.detect(frame)
        return faces if faces is not None else []

    def _get_2d_coords(self, lm, shape):
        h, w = shape
        return int(lm.x * w), int(lm.y * h)

    def _extract_face_regions(self, frame, landmarks_list, shape):
        h, w = shape
        regions = []
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

            full_face = frame[y:y+h_bbox, x:x+w_bbox]
            if full_face.size > 0: regions.append(('full', full_face))

            eye_region_y = max(0, y - padding)
            eye_region_h = min(h - eye_region_y, int(h_bbox * 0.6))
            eye_region = frame[eye_region_y:eye_region_y+eye_region_h, x:x+w_bbox]
            if eye_region.size > 0: regions.append(('eyes', eye_region))

            mouth_y = y + int(h_bbox * 0.6)
            mouth_h = min(h - mouth_y, int(h_bbox * 0.4))
            mouth_region = frame[mouth_y:mouth_y+mouth_h, x:x+w_bbox]
            if mouth_region.size > 0: regions.append(('mouth', mouth_region))

        except Exception as e:
            print(f"Erro ao extrair regiões: {e}")
            
        return regions

    def _corrigir_tendencia_tristeza(self, emotion_data, dominant_emotion):
        if dominant_emotion != 'sad':
            return dominant_emotion
        
        sad_score = emotion_data.get('sad', 0)
        neutral_score = emotion_data.get('neutral', 0)
        happy_score = emotion_data.get('happy', 0)
        
        if sad_score < 40: return 'neutral'
        if sad_score - neutral_score < self.SAD_TO_NEUTRAL_THRESHOLD: return 'neutral'
        if happy_score > 30 and sad_score - happy_score < 20: return 'happy'
        if sad_score < 60 and neutral_score > 20: return 'neutral'
        if sad_score > 70 and sad_score - max(neutral_score, happy_score) > 30: return 'sad'
        
        return 'neutral'

    def _verificar_segunda_emocao(self, emotion_data, dominant_emotion):
        """
        NOVO: Se a emoção dominante for NEUTRO, tenta puxar a segunda emoção
        se ela for forte o suficiente (ex: felicidade ou surpresa).
        """
        if dominant_emotion != 'neutral':
            return dominant_emotion

        # Ordena as emoções por score
        sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)
        
        # Se não tiver dados suficientes
        if len(sorted_emotions) < 2:
            return dominant_emotion
            
        # Pega a segunda colocada
        second_emotion, second_score = sorted_emotions[1]

        # Lista de emoções que valem a pena "resgatar" do neutro
        emocoes_resgataveis = ['happy', 'surprise', 'fear'] 
        
        # Se a segunda emoção for relevante e tiver score decente (> 15%)
        if second_emotion in emocoes_resgataveis and second_score > self.SECOND_EMOTION_THRESHOLD:
            return second_emotion
            
        return dominant_emotion

    def _analisar_expressao_facial(self, landmarks_list):
        try:
            mouth_left = landmarks_list[61]
            mouth_right = landmarks_list[291]
            mouth_top = landmarks_list[13]
            mouth_bottom = landmarks_list[14]
            left_eyebrow = landmarks_list[65]
            right_eyebrow = landmarks_list[295]
            
            mouth_width = abs(mouth_right.x - mouth_left.x)
            mouth_height = abs(mouth_bottom.y - mouth_top.y)
            eyebrow_height = (left_eyebrow.y + right_eyebrow.y) / 2
            
            smile_ratio = mouth_height / (mouth_width + 1e-6)
            eyebrow_raise_factor = 0.25 - eyebrow_height 
            
            if smile_ratio > 0.18: return "happy", 0.8
            elif smile_ratio > 0.12: return "happy", 0.6
            elif eyebrow_raise_factor > 0.15: return "surprise", 0.7
            else: return "neutral", 0.6
            
        except Exception:
            return "neutral", 0.5

    def _analyze_emotion_robust(self, frame, landmarks_list, shape):
        try:
            regions = self._extract_face_regions(frame, landmarks_list, shape)
            if not regions:
                return "neutral", 0.6 

            emotions = []
            confidences = []
            region_weights = {'full': 0.5, 'eyes': 0.2, 'mouth': 0.3}

            for region_name, region_img in regions:
                try:
                    analysis = DeepFace.analyze(
                        img_path=region_img,
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True,
                        detector_backend='opencv'
                    )
                    
                    if isinstance(analysis, list): analysis = analysis[0]

                    raw_dominant = analysis['dominant_emotion']
                    emotion_data = analysis['emotion']
                    
                    # 1. Tenta resgatar emoção escondida no neutro
                    dominant_emotion = self._verificar_segunda_emocao(emotion_data, raw_dominant)

                    # 2. Aplica correção agressiva para tristeza (se sobrou tristeza)
                    dominant_emotion = self._corrigir_tendencia_tristeza(emotion_data, dominant_emotion)
                    
                    # Cálculo de confiança
                    sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)
                    if len(sorted_emotions) >= 2:
                        confidence = (sorted_emotions[0][1] - sorted_emotions[1][1]) / 100.0
                    else:
                        confidence = sorted_emotions[0][1] / 100.0
                    
                    weight = region_weights.get(region_name, 0.33)
                    weighted_confidence = confidence * weight
                    
                    min_dim = min(region_img.shape[:2])
                    if min_dim < 40: weighted_confidence *= 0.5
                    
                    emotions.append(dominant_emotion)
                    confidences.append(weighted_confidence)

                except Exception as e:
                    continue

            if not emotions:
                return self._analisar_expressao_facial(landmarks_list)

            emotion_scores = {}
            for emotion, conf in zip(emotions, confidences):
                emotion_scores[emotion] = emotion_scores.get(emotion, 0) + conf
            
            final_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            
            avg_confidence = np.mean(confidences) if confidences else 0.0
            consistency = emotions.count(final_emotion) / len(emotions)
            final_confidence = avg_confidence * (0.6 + 0.4 * consistency)
            
            if final_emotion == 'sad' and final_confidence < self.MAX_SAD_CONFIDENCE:
                return "neutral", final_confidence * 0.9

            return final_emotion, final_confidence

        except Exception as e:
            print(f"Erro na análise robusta: {e}")
            return "neutral", 0.5

    def _get_emotion(self, frame, landmarks_list, shape):
        emotion, confidence = self._analyze_emotion_robust(frame, landmarks_list, shape)
        
        if confidence < 0.3:
            emotion_geo, confidence_geo = self._analisar_expressao_facial(landmarks_list)
            if confidence_geo > confidence:
                emotion, confidence = emotion_geo, confidence_geo
        
        self.emotion_history.append(emotion)
        
        if len(self.emotion_history) >= 3:
            emotion_counts = {}
            for e in self.emotion_history:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            
            most_common = max(emotion_counts.items(), key=lambda x: x[1])[0]
            
            if emotion != most_common:
                if emotion == 'sad' and most_common != 'sad':
                    emotion = most_common
                    confidence *= 0.8
                elif most_common == 'neutral':
                    # Se o histórico é neutro, permite flutuação se a confiança atual for boa
                    if confidence < 0.6: 
                        emotion = 'neutral'
                        confidence = min(1.0, confidence * 1.1)
        
        if emotion == 'sad' and confidence < 0.7:
            emotion = 'neutral'
            confidence = confidence * 0.9
        
        return emotion, confidence

    def _get_head_pose(self, landmarks_list, shape):
        h, w = shape
        image_points_2d = np.array([self._get_2d_coords(landmarks_list[i], shape) for i in self.pose_landmark_indices], dtype="double")
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv.solvePnP(self.face_3d_model_points, image_points_2d, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
        rotation_matrix, _ = cv.Rodrigues(rotation_vector)
        _, _, _, _, _, _, eulerAngles = cv.decomposeProjectionMatrix(np.hstack((rotation_matrix, translation_vector)))

        yaw = eulerAngles[1][0]
        pitch = eulerAngles[0][0]
        roll = eulerAngles[2][0]

        self.pose_history.append((yaw, pitch, roll))
        if len(self.pose_history) >= 2:
            yaw = np.mean([p[0] for p in self.pose_history])
            pitch = np.mean([p[1] for p in self.pose_history])
            roll = np.mean([p[2] for p in self.pose_history])

        # Ajuste de tolerância para evitar "lados" muito facilmente
        if yaw > 18: pos_h = "Direita"
        elif yaw < -18: pos_h = "Esquerda"
        else: pos_h = "Frente"

        if pitch > 12: pos_v = "Baixo"
        elif pitch < -12: pos_v = "Cima"
        else: pos_v = "Frente"
            
        proximidade_z = translation_vector[2][0]

        return {"direcao_horizontal": pos_h, "direcao_vertical": pos_v, "proximidade_z": proximidade_z, 
                "raw_pitch": pitch, "raw_yaw": yaw, "raw_roll": roll}
        
    def _get_gaze_direction(self, landmarks_list, shape):
        # ... (código inalterado para cálculo de iris) ...
        # (Mantendo a lógica existente para brevidade, mas você pode usar o código anterior aqui)
        h, w = shape
        left_iris_coords = [landmarks_list[i] for i in self.IRIS_LEFT_INDICES]
        left_iris_center_x = np.mean([lm.x for lm in left_iris_coords])
        left_iris_center_y = np.mean([lm.y for lm in left_iris_coords])
        
        eye_left_outer_x = landmarks_list[self.EYE_LEFT_OUTER].x
        eye_left_inner_x = landmarks_list[self.EYE_LEFT_INNER].x
        eye_left_top_y = landmarks_list[self.EYE_LEFT_TOP].y
        eye_left_bottom_y = landmarks_list[self.EYE_LEFT_BOTTOM].y

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
        
        if ratio_h_avg > 0.65: gaze_h = "Direita"
        elif ratio_h_avg < 0.35: gaze_h = "Esquerda"
        else: gaze_h = "Frente"

        if ratio_v_avg > 0.60: gaze_v = "Baixo"
        elif ratio_v_avg < 0.40: gaze_v = "Cima"
        else: gaze_v = "Frente"
            
        return {"direcao_horizontal": gaze_h, "direcao_vertical": gaze_v, 
                "raw_ratio_h": ratio_h_avg, "raw_ratio_v": ratio_v_avg}

    def _get_eye_aspect_ratio(self, landmarks_list):
        v_dist_left = abs(landmarks_list[self.EYE_LEFT_BOTTOM].y - landmarks_list[self.EYE_LEFT_TOP].y)
        h_dist_left = abs(landmarks_list[self.EYE_LEFT_INNER].x - landmarks_list[self.EYE_LEFT_OUTER].x)
        v_dist_right = abs(landmarks_list[self.EYE_RIGHT_BOTTOM].y - landmarks_list[self.EYE_RIGHT_TOP].y)
        h_dist_right = abs(landmarks_list[self.EYE_RIGHT_OUTER].x - landmarks_list[self.EYE_RIGHT_INNER].x)
        epsilon = 1e-6
        return ((v_dist_left / (h_dist_left + epsilon)) + (v_dist_right / (h_dist_right + epsilon))) / 2.0

    def _determinar_pose_enum(self, pose_obj):
        # Mapeamento levemente mais flexível
        if pose_obj.direcao_vertical == "Baixo": return PoseCabecaEnum.BAIXO
        elif pose_obj.direcao_horizontal != "Frente": return PoseCabecaEnum.LADOS
        elif pose_obj.direcao_vertical == "Frente": return PoseCabecaEnum.FRENTE
        # Default para Frente se for algo estranho mas não extremo
        return PoseCabecaEnum.FRENTE 

    def _determinar_olhar_enum(self, gaze_obj):
        if gaze_obj.direcao_horizontal == "Frente": return OlharDirecaoEnum.FRENTE
        else: return OlharDirecaoEnum.LADOS

    def _detectar_estado_fluxo(self, emocao, dimensao, confidence, pose, olhar):
        concentrado = dimensao in [DimensaoComportamentalEnum.CONCENTRADO, DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO]
        if not concentrado: return False
            
        # Aceita neutro como fluxo se a postura for muito estável
        emocoa_adequada = emocao in [EmocaoEnum.FELIZ, EmocaoEnum.SURPRESO, EmocaoEnum.NEUTRO]
        if not emocoa_adequada: return False
            
        postura_adequada = (
            pose.direcao_horizontal == "Frente" and
            pose.direcao_vertical in ["Frente", "Baixo"] and
            abs(pose.raw_yaw) < 20 and abs(pose.raw_pitch) < 25
        )
        if not postura_adequada: return False
            
        olhar_focalizado = (
            olhar.direcao_horizontal == "Frente" and
            olhar.direcao_vertical in ["Frente", "Baixo"]
        )
        if not olhar_focalizado: return False
        
        # Reduzi threshold de confiança para fluxo
        if confidence < 0.4: return False
            
        return True

    def _calcular_dimensao_comportamental(self, pose: PoseCabecaEnum, olhar: OlharDirecaoEnum, olhos_fechados: bool) -> DimensaoComportamentalEnum:
        """
        Lógica ajustada para EVITAR 'INDEFINIDO' puro.
        Prioriza INDEFINIDO_CONCENTRADO ou INDEFINIDO_DISTRAIDO.
        """
        
        # Caso 1: Olhos Fechados = Distração quase certa (sono/tédio)
        if olhos_fechados:
            return DimensaoComportamentalEnum.DISTRAIDO

        # Caso 2: Olhando para baixo (leitura/anotação)
        if pose == PoseCabecaEnum.BAIXO:
            if olhar == OlharDirecaoEnum.FRENTE or olhar == OlharDirecaoEnum.INDEFINIDO:
                return DimensaoComportamentalEnum.CONCENTRADO
            else:
                return DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO # Assume trabalho na mesa

        # Caso 3: Olhando para os lados
        elif pose == PoseCabecaEnum.LADOS:
            if olhar == OlharDirecaoEnum.LADOS:
                return DimensaoComportamentalEnum.DISTRAIDO
            elif olhar == OlharDirecaoEnum.FRENTE:
                # Cabeça virada mas olho na tela = Atenção parcial
                return DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO
            else:
                return DimensaoComportamentalEnum.INDEFINIDO_DISTRAIDO

        # Caso 4: Cabeça de Frente
        elif pose == PoseCabecaEnum.FRENTE:
            if olhar == OlharDirecaoEnum.FRENTE:
                return DimensaoComportamentalEnum.CONCENTRADO
            elif olhar == OlharDirecaoEnum.LADOS:
                return DimensaoComportamentalEnum.INDEFINIDO_DISTRAIDO
            else:
                # Se pose é frente mas olhar falhou, assume concentrado
                return DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO
        
        # FALLBACK FINAL ROBUSTO:
        # Se chegou aqui (pose indefinida), tenta salvar baseado no olhar
        if olhar == OlharDirecaoEnum.FRENTE:
            return DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO
        elif olhar == OlharDirecaoEnum.LADOS:
            return DimensaoComportamentalEnum.INDEFINIDO_DISTRAIDO
        
        # Se tudo falhar, assume concentrado "leve" em vez de nulo
        return DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO

    def _get_emotion_score(self, emocao: EmocaoEnum, confidence: float) -> float:
        base_scores = {
            EmocaoEnum.FELIZ: 45,
            EmocaoEnum.SURPRESO: 40,
            EmocaoEnum.NEUTRO: 35,
            EmocaoEnum.MEDO: 10,
            EmocaoEnum.TRISTE: 5, # Tristeza pontua pouco, não zero
            EmocaoEnum.INDEFINIDO: 15
        }
        return base_scores.get(emocao, 0) * confidence

    def _calcular_estimativa_engajamento(self, dimensao, emocao, confidence, estado_fluxo=False):
        if estado_fluxo: return EstimativaEngajamentoEnum.ALTAMENTE_ENGAJADO
            
        score = 0
        dimensao_scores = {
            DimensaoComportamentalEnum.CONCENTRADO: 70,
            DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO: 55, # Aumentado
            DimensaoComportamentalEnum.INDEFINIDO_DISTRAIDO: 30, # Aumentado
            DimensaoComportamentalEnum.DISTRAIDO: 5,
            DimensaoComportamentalEnum.INDEFINIDO: 20
        }
        score += dimensao_scores.get(dimensao, 20)
        score += self._get_emotion_score(emocao, confidence)
        
        if score >= 80: return EstimativaEngajamentoEnum.ALTAMENTE_ENGAJADO
        elif score >= 55: return EstimativaEngajamentoEnum.ENGAJADO
        elif score >= 30: return EstimativaEngajamentoEnum.DESENGAJADO
        else: return EstimativaEngajamentoEnum.ALTAMENTE_DESENGAJADO

    def analyze_frame(self, frame, timestamp_ms: int, video_id: str, frame_number: int) -> FrameAnalysis:
        h, w, _ = frame.shape
        shape_2d = (h, w)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks_list = face_landmarks.landmark

            raw_emotion_str, emotion_confidence = self._get_emotion(frame, landmarks_list, shape_2d)
            raw_pose_dict = self._get_head_pose(landmarks_list, shape_2d)
            raw_gaze_dict = self._get_gaze_direction(landmarks_list, shape_2d)
            avg_ear = self._get_eye_aspect_ratio(landmarks_list)
            olhos_fechados = avg_ear < self.OLHOS_FECHADOS_THRESHOLD
            
            if emotion_confidence > 0.2:
                emocao_enum = DEEPFACE_EMOTION_MAP.get(raw_emotion_str, EmocaoEnum.NEUTRO)
            else:
                emocao_enum = EmocaoEnum.NEUTRO

            # Verifica e ajusta tristeza final
            if emocao_enum == EmocaoEnum.TRISTE:
                if emotion_confidence < 0.7:
                    emocao_enum = EmocaoEnum.NEUTRO
                    emotion_confidence = emotion_confidence * 0.9

            pose_obj = HeadPose(
                direcao_horizontal=raw_pose_dict['direcao_horizontal'],
                raw_yaw=raw_pose_dict['raw_yaw'],
                direcao_vertical=raw_pose_dict['direcao_vertical'],
                raw_pitch=raw_pose_dict['raw_pitch'],
                proximidade_z=raw_pose_dict['proximidade_z'],
                raw_roll=raw_pose_dict['raw_roll']
            )
            
            gaze_obj = GazeDirection(
                direcao_horizontal=raw_gaze_dict['direcao_horizontal'],
                raw_ratio_h=raw_gaze_dict['raw_ratio_h'],
                direcao_vertical=raw_gaze_dict['direcao_vertical'],
                raw_ratio_v=raw_gaze_dict['raw_ratio_v']
            )

            pose_enum = self._determinar_pose_enum(pose_obj)
            olhar_enum = self._determinar_olhar_enum(gaze_obj)

            # Aqui está a mágica da dimensão comportamental sem Indefinido puro
            dimensao_enum = self._calcular_dimensao_comportamental(pose_enum, olhar_enum, olhos_fechados)
            
            estado_fluxo = self._detectar_estado_fluxo(
                emocao_enum, dimensao_enum, emotion_confidence, pose_obj, gaze_obj
            )

            estimativa_enum = self._calcular_estimativa_engajamento(
                dimensao_enum, emocao_enum, emotion_confidence, estado_fluxo
            )

            return FrameAnalysis(
                video_id=video_id, timestamp_ms=timestamp_ms, frame_number=frame_number,
                emocao=emocao_enum.value, pose_cabeca=pose_obj, olhar=gaze_obj,
                dimensao_comportamental=dimensao_enum, estimativa_engajamento=estimativa_enum,
                emotion_confidence=emotion_confidence, estado_fluxo=estado_fluxo
            )
        
        else:
            # Fallback para quando não acha rosto (aqui mantemos indefinido puro pois não há dados)
            default_pose = HeadPose("Indefinido", 0.0, "Indefinido", 0.0, 0.0, 0.0)
            default_gaze = GazeDirection("Indefinido", 0.0, "Indefinido", 0.0)

            return FrameAnalysis(
                video_id=video_id, timestamp_ms=timestamp_ms, frame_number=frame_number,
                emocao=EmocaoEnum.NEUTRO.value, pose_cabeca=default_pose, olhar=default_gaze,
                dimensao_comportamental=DimensaoComportamentalEnum.INDEFINIDO,
                estimativa_engajamento=EstimativaEngajamentoEnum.INDEFINIDO,
                emotion_confidence=0.0, estado_fluxo=False
            )