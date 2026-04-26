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
    'happy':   EmocaoEnum.FELIZ,
    'sad':     EmocaoEnum.TRISTE,
    'neutral': EmocaoEnum.NEUTRO,
    'surprise':EmocaoEnum.SURPRESO,
    'fear':    EmocaoEnum.MEDO,
    # angry e disgust são mapeados para TRISTE: semanticamente indicam tensão/
    # desconforto, e o modelo não tem enum próprio para raiva/nojo. Preservar
    # o dado é melhor que descartar para INDEFINIDO.
    'angry':   EmocaoEnum.TRISTE,
    'disgust': EmocaoEnum.TRISTE,
}

class FaceAnalyzer:
    """
    Classe ajustada para evitar comportamentos indefinidos puros e 
    extrair emoções secundárias quando o neutro predomina.
    """
    
    def __init__(self, max_faces=1):
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,        # <--- A MÁGICA ACONTECE AQUI
            max_num_faces=max_faces,
            refine_landmarks=True,       
            min_detection_confidence=0.1,  # <--- Reduzido ao extremo para aceitar rostos parciais
            min_tracking_confidence=0.1    
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
        self.pose_history = deque(maxlen=5)   # cada entrada: (yaw, pitch, roll, proximidade_z)
        self.gaze_history = deque(maxlen=5)   # cada entrada: (ratio_h, ratio_v)
        
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
        """
        Corrige APENAS falsos positivos flagrantes de tristeza — quando o score é
        muito baixo ou outra emoção positiva é claramente mais forte.
        Não zera tristeza moderada: isso era a principal causa de perda de dados.
        """
        if dominant_emotion not in ('sad', 'angry', 'disgust'):
            return dominant_emotion

        sad_score    = emotion_data.get('sad', 0) + emotion_data.get('angry', 0) * 0.5 + emotion_data.get('disgust', 0) * 0.5
        neutral_score = emotion_data.get('neutral', 0)
        happy_score  = emotion_data.get('happy', 0)

        # Só corrige se outra emoção positiva domina claramente
        if happy_score > sad_score + 20:
            return 'happy'

        # Score combinado negativo muito fraco: provável ruído
        if sad_score < 20:
            return 'neutral'

        # Mantém a tristeza — deixa o threshold de confiança decidir no analyze_frame
        return dominant_emotion

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

        # Todas as emoções não-neutras valem ser resgatadas se tiverem score suficiente
        emocoes_resgataveis = ['happy', 'surprise', 'fear', 'sad', 'angry', 'disgust']

        if second_emotion in emocoes_resgataveis and second_score > self.SECOND_EMOTION_THRESHOLD:
            return second_emotion
            
        return dominant_emotion

    def _analisar_expressao_facial(self, landmarks_list):
        """
        Análise geométrica por landmarks como fallback do DeepFace.
        Detecta: feliz, surpreso, triste, medo, raiva e neutro.
        """
        try:
            # Boca
            mouth_left   = landmarks_list[61]
            mouth_right  = landmarks_list[291]
            mouth_top    = landmarks_list[13]
            mouth_bottom = landmarks_list[14]
            # Cantos da boca (para sorriso vs caído)
            mouth_corner_left  = landmarks_list[57]
            mouth_corner_right = landmarks_list[287]
            # Sobrancelhas
            left_eyebrow_inner  = landmarks_list[65]
            right_eyebrow_inner = landmarks_list[295]
            left_eyebrow_outer  = landmarks_list[46]
            right_eyebrow_outer = landmarks_list[276]
            # Olhos (abertura)
            left_eye_top    = landmarks_list[159]
            left_eye_bottom = landmarks_list[145]
            right_eye_top   = landmarks_list[386]
            right_eye_bottom= landmarks_list[374]
            # Nariz (ponto central para referência vertical)
            nose_tip = landmarks_list[1]

            mouth_width  = abs(mouth_right.x - mouth_left.x)
            mouth_height = abs(mouth_bottom.y - mouth_top.y)
            mouth_open_ratio = mouth_height / (mouth_width + 1e-6)

            # Cantos da boca: positivo = sorriso, negativo = boca caída
            mouth_corner_avg_y = (mouth_corner_left.y + mouth_corner_right.y) / 2
            mouth_center_y     = (mouth_top.y + mouth_bottom.y) / 2
            mouth_corner_lift  = mouth_center_y - mouth_corner_avg_y  # >0 = sorriso

            # Sobrancelhas: altura relativa ao nariz
            eyebrow_inner_avg_y = (left_eyebrow_inner.y + right_eyebrow_inner.y) / 2
            eyebrow_outer_avg_y = (left_eyebrow_outer.y + right_eyebrow_outer.y) / 2
            eyebrow_raise       = nose_tip.y - eyebrow_inner_avg_y   # >0 = levantadas
            eyebrow_furrow      = eyebrow_outer_avg_y - eyebrow_inner_avg_y  # >0 = franzidas

            # Abertura dos olhos
            left_eye_open  = abs(left_eye_bottom.y  - left_eye_top.y)
            right_eye_open = abs(right_eye_bottom.y - right_eye_top.y)
            eye_open_avg   = (left_eye_open + right_eye_open) / 2

            # --- Decisão ---
            # Surpresa: boca aberta + sobrancelhas levantadas + olhos abertos
            if mouth_open_ratio > 0.22 and eyebrow_raise > 0.20 and eye_open_avg > 0.03:
                return "surprise", 0.75

            # Feliz: cantos levantados + boca moderadamente aberta
            if mouth_corner_lift > 0.005 and mouth_open_ratio > 0.08:
                return "happy", 0.80
            if mouth_corner_lift > 0.003:
                return "happy", 0.60

            # Medo: sobrancelhas levantadas + olhos muito abertos + boca entreaberta
            if eyebrow_raise > 0.18 and eye_open_avg > 0.04 and mouth_open_ratio > 0.10:
                return "fear", 0.65

            # Raiva/tensão: sobrancelhas franzidas + boca fechada ou comprimida
            if eyebrow_furrow > 0.015 and mouth_open_ratio < 0.08:
                return "angry", 0.60

            # Triste: cantos caídos + sobrancelhas levemente franzidas
            if mouth_corner_lift < -0.003 and eyebrow_raise < 0.14:
                return "sad", 0.60

            # Neutro: nenhum sinal expressivo claro
            return "neutral", 0.45

        except Exception:
            return "neutral", 0.40

    def _analyze_emotion_robust(self, frame, landmarks_list, shape):
        """
        Analisa a emoção usando o crop facial completo extraído via landmarks do MediaPipe.

        Estratégia:
          1. Extrai o crop do rosto a partir dos landmarks (já sabemos que o rosto existe).
          2. Passa o crop para o DeepFace com detector_backend='skip' — assim o DeepFace
             analisa a imagem diretamente sem tentar (e falhar em) detectar rosto de novo.
          3. Se o DeepFace falhar, usa análise geométrica por landmarks como fallback.

        Por que 'skip' e não 'opencv'?
          - Quando recebe um crop sem contexto extra, o detector OpenCV/RetinaFace do DeepFace
            frequentemente não encontra o rosto e lança exceção, que era silenciada e causava
            retorno de "neutral" padrão. Com 'skip', o DeepFace confia que o input já é o rosto.
        """
        try:
            h, w = shape
            all_x = [int(lm.x * w) for lm in landmarks_list]
            all_y = [int(lm.y * h) for lm in landmarks_list]
            x1 = max(0, min(all_x) - 20)
            y1 = max(0, min(all_y) - 20)
            x2 = min(w, max(all_x) + 20)
            y2 = min(h, max(all_y) + 20)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0 or min(face_crop.shape[:2]) < 48:
                # Crop inválido ou muito pequeno: usa geometria pura
                return self._analisar_expressao_facial(landmarks_list)

            analysis = DeepFace.analyze(
                img_path=face_crop,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='skip',   # crop já é o rosto — pula detecção interna
                silent=True,
            )

            if isinstance(analysis, list):
                analysis = analysis[0]

            raw_dominant = analysis['dominant_emotion']
            emotion_data = analysis['emotion']

            # Tenta resgatar emoção real quando neutro domina superficialmente
            dominant_emotion = self._verificar_segunda_emocao(emotion_data, raw_dominant)
            # Corrige viés de tristeza
            dominant_emotion = self._corrigir_tendencia_tristeza(emotion_data, dominant_emotion)

            # Confiança = margem entre 1ª e 2ª emoção (mais discriminativa que score absoluto)
            sorted_emotions = sorted(emotion_data.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_emotions) >= 2:
                confidence = (sorted_emotions[0][1] - sorted_emotions[1][1]) / 100.0
            else:
                confidence = sorted_emotions[0][1] / 100.0

            # Penaliza crop muito pequeno
            if min(face_crop.shape[:2]) < 80:
                confidence *= 0.75

            if dominant_emotion == 'sad' and confidence < self.MAX_SAD_CONFIDENCE:
                return "neutral", confidence * 0.9

            return dominant_emotion, float(confidence)

        except Exception as e:
            print(f"Aviso _analyze_emotion_robust: {e}")
            # Fallback: análise geométrica por landmarks
            return self._analisar_expressao_facial(landmarks_list)

    def _get_emotion(self, frame, landmarks_list, shape):
        """
        Obtém a emoção do frame atual, combinando DeepFace com análise geométrica
        e aplicando suavização temporal leve para reduzir flicker sem suprimir
        mudanças legítimas de emoção.
        """
        emotion, confidence = self._analyze_emotion_robust(frame, landmarks_list, shape)

        # Se DeepFace retornou baixa confiança, tenta complementar com geometria
        # Threshold reduzido para 0.15: antes de 0.3 descartava resultados válidos
        if confidence < 0.15:
            emotion_geo, confidence_geo = self._analisar_expressao_facial(landmarks_list)
            if confidence_geo > confidence:
                emotion, confidence = emotion_geo, confidence_geo

        self.emotion_history.append((emotion, confidence))

        # Suavização temporal: usa média ponderada por confiança em vez de voto puro.
        # Isso evita que um histórico de neutros de baixa confiança suprima uma emoção
        # clara e recente de alta confiança.
        if len(self.emotion_history) >= 3:
            # Acumula score ponderado por emoção
            weighted_scores: dict = {}
            for e, c in self.emotion_history:
                weighted_scores[e] = weighted_scores.get(e, 0.0) + c

            smoothed_emotion = max(weighted_scores.items(), key=lambda x: x[1])[0]

            # Só substitui se a emoção suavizada tiver confiança maior que a atual
            if smoothed_emotion != emotion:
                smoothed_conf = weighted_scores[smoothed_emotion] / len(self.emotion_history)
                if smoothed_conf > confidence * 1.5:  # exige margem de 50% para sobrescrever
                    emotion = smoothed_emotion
                    confidence = smoothed_conf

        # Não filtramos mais tristeza aqui — o único filtro fica em analyze_frame
        # com threshold mínimo (0.05). Dupla filtragem era a causa de perda de dado.
        return emotion, confidence

    def _get_head_pose(self, landmarks_list, shape):
        h, w = shape

        try:
            image_points_2d = np.array(
                [self._get_2d_coords(landmarks_list[i], shape) for i in self.pose_landmark_indices],
                dtype="double"
            )
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                dtype="double"
            )
            dist_coeffs = np.zeros((4, 1))

            success, rotation_vector, translation_vector = cv.solvePnP(
                self.face_3d_model_points, image_points_2d,
                camera_matrix, dist_coeffs,
                flags=cv.SOLVEPNP_ITERATIVE
            )

            # Se o solvePnP falhou, usa o último valor válido do histórico
            if not success:
                return self._pose_from_history()

            rotation_matrix, _ = cv.Rodrigues(rotation_vector)
            _, _, _, _, _, _, eulerAngles = cv.decomposeProjectionMatrix(
                np.hstack((rotation_matrix, translation_vector))
            )

            yaw   = float(eulerAngles[1][0])
            pitch = float(eulerAngles[0][0])
            roll  = float(eulerAngles[2][0])

            # Sanidade: ângulos absurdos indicam solução inválida — usa histórico
            if abs(yaw) > 90 or abs(pitch) > 90:
                return self._pose_from_history()

            proximidade_z = float(translation_vector[2][0])

        except Exception as e:
            print(f"Aviso _get_head_pose: {e}")
            return self._pose_from_history()

        # Suaviza com histórico de poses válidas
        self.pose_history.append((yaw, pitch, roll, proximidade_z))
        if len(self.pose_history) >= 2:
            yaw          = float(np.mean([p[0] for p in self.pose_history]))
            pitch        = float(np.mean([p[1] for p in self.pose_history]))
            roll         = float(np.mean([p[2] for p in self.pose_history]))
            proximidade_z = float(np.mean([p[3] for p in self.pose_history]))

        if yaw > 18:        pos_h = "Direita"
        elif yaw < -18:     pos_h = "Esquerda"
        else:               pos_h = "Frente"

        if pitch > 12:      pos_v = "Baixo"
        elif pitch < -12:   pos_v = "Cima"
        else:               pos_v = "Frente"

        return {
            "direcao_horizontal": pos_h,
            "direcao_vertical": pos_v,
            "proximidade_z": proximidade_z,
            "raw_pitch": pitch,
            "raw_yaw": yaw,
            "raw_roll": roll,
        }

    def _pose_from_history(self) -> dict:
        """
        Retorna a média do histórico de poses válidas como fallback.
        Se não há histórico ainda, assume posição frontal (a mais provável).
        """
        if self.pose_history:
            yaw          = float(np.mean([p[0] for p in self.pose_history]))
            pitch        = float(np.mean([p[1] for p in self.pose_history]))
            roll         = float(np.mean([p[2] for p in self.pose_history]))
            proximidade_z = float(np.mean([p[3] for p in self.pose_history]))
        else:
            yaw = pitch = roll = proximidade_z = 0.0

        if yaw > 18:        pos_h = "Direita"
        elif yaw < -18:     pos_h = "Esquerda"
        else:               pos_h = "Frente"

        if pitch > 12:      pos_v = "Baixo"
        elif pitch < -12:   pos_v = "Cima"
        else:               pos_v = "Frente"

        return {
            "direcao_horizontal": pos_h,
            "direcao_vertical": pos_v,
            "proximidade_z": proximidade_z,
            "raw_pitch": pitch,
            "raw_yaw": yaw,
            "raw_roll": roll,
        }
        
    def _get_gaze_direction(self, landmarks_list, shape):
        h, w = shape

        try:
            total_landmarks = len(landmarks_list)

            # Índices de íris (469-477) só existem com refine_landmarks=True.
            # Se o MediaPipe não os forneceu, cai no fallback geométrico.
            iris_indices_needed = self.IRIS_LEFT_INDICES + self.IRIS_RIGHT_INDICES
            if total_landmarks <= max(iris_indices_needed):
                return self._gaze_from_history()

            left_iris_coords  = [landmarks_list[i] for i in self.IRIS_LEFT_INDICES]
            right_iris_coords = [landmarks_list[i] for i in self.IRIS_RIGHT_INDICES]

            left_iris_center_x  = float(np.mean([lm.x for lm in left_iris_coords]))
            left_iris_center_y  = float(np.mean([lm.y for lm in left_iris_coords]))
            right_iris_center_x = float(np.mean([lm.x for lm in right_iris_coords]))
            right_iris_center_y = float(np.mean([lm.y for lm in right_iris_coords]))

            eye_left_outer_x  = landmarks_list[self.EYE_LEFT_OUTER].x
            eye_left_inner_x  = landmarks_list[self.EYE_LEFT_INNER].x
            eye_left_top_y    = landmarks_list[self.EYE_LEFT_TOP].y
            eye_left_bottom_y = landmarks_list[self.EYE_LEFT_BOTTOM].y

            eye_right_outer_x  = landmarks_list[self.EYE_RIGHT_OUTER].x
            eye_right_inner_x  = landmarks_list[self.EYE_RIGHT_INNER].x
            eye_right_top_y    = landmarks_list[self.EYE_RIGHT_TOP].y
            eye_right_bottom_y = landmarks_list[self.EYE_RIGHT_BOTTOM].y

            epsilon = 1e-6

            # Sanidade: se a abertura do olho for quase zero (olho fechado/ocluído),
            # o ratio fica instável — descarta esse olho individualmente
            left_eye_width  = abs(eye_left_inner_x  - eye_left_outer_x)
            right_eye_width = abs(eye_right_inner_x - eye_right_outer_x)
            left_eye_height  = abs(eye_left_bottom_y  - eye_left_top_y)
            right_eye_height = abs(eye_right_bottom_y - eye_right_top_y)

            ratios_h = []
            ratios_v = []

            if left_eye_width > 0.01:
                ratios_h.append(
                    (left_iris_center_x - eye_left_outer_x) / (eye_left_inner_x - eye_left_outer_x + epsilon)
                )
            if right_eye_width > 0.01:
                ratios_h.append(
                    (right_iris_center_x - eye_right_outer_x) / (eye_right_inner_x - eye_right_outer_x + epsilon)
                )
            if left_eye_height > 0.005:
                ratios_v.append(
                    (left_iris_center_y - eye_left_top_y) / (eye_left_bottom_y - eye_left_top_y + epsilon)
                )
            if right_eye_height > 0.005:
                ratios_v.append(
                    (right_iris_center_y - eye_right_top_y) / (eye_right_bottom_y - eye_right_top_y + epsilon)
                )

            # Se nenhum olho ficou confiável, usa histórico
            if not ratios_h and not ratios_v:
                return self._gaze_from_history()

            ratio_h_avg = float(np.mean(ratios_h)) if ratios_h else 0.5
            ratio_v_avg = float(np.mean(ratios_v)) if ratios_v else 0.5

            # Clamp para evitar valores fora de [0,1] por oclusão parcial
            ratio_h_avg = max(0.0, min(1.0, ratio_h_avg))
            ratio_v_avg = max(0.0, min(1.0, ratio_v_avg))

        except Exception as e:
            print(f"Aviso _get_gaze_direction: {e}")
            return self._gaze_from_history()

        # Salva no histórico de olhar
        self.gaze_history.append((ratio_h_avg, ratio_v_avg))
        if len(self.gaze_history) >= 2:
            ratio_h_avg = float(np.mean([g[0] for g in self.gaze_history]))
            ratio_v_avg = float(np.mean([g[1] for g in self.gaze_history]))

        if ratio_h_avg > 0.65:   gaze_h = "Direita"
        elif ratio_h_avg < 0.35: gaze_h = "Esquerda"
        else:                    gaze_h = "Frente"

        if ratio_v_avg > 0.60:   gaze_v = "Baixo"
        elif ratio_v_avg < 0.40: gaze_v = "Cima"
        else:                    gaze_v = "Frente"

        return {
            "direcao_horizontal": gaze_h,
            "direcao_vertical": gaze_v,
            "raw_ratio_h": ratio_h_avg,
            "raw_ratio_v": ratio_v_avg,
        }

    def _gaze_from_history(self) -> dict:
        """
        Retorna a média do histórico de olhar como fallback.
        Se não há histórico, assume olhar para frente.
        """
        if self.gaze_history:
            ratio_h_avg = float(np.mean([g[0] for g in self.gaze_history]))
            ratio_v_avg = float(np.mean([g[1] for g in self.gaze_history]))
        else:
            ratio_h_avg = ratio_v_avg = 0.5  # centro = "Frente"

        if ratio_h_avg > 0.65:   gaze_h = "Direita"
        elif ratio_h_avg < 0.35: gaze_h = "Esquerda"
        else:                    gaze_h = "Frente"

        if ratio_v_avg > 0.60:   gaze_v = "Baixo"
        elif ratio_v_avg < 0.40: gaze_v = "Cima"
        else:                    gaze_v = "Frente"

        return {
            "direcao_horizontal": gaze_h,
            "direcao_vertical": gaze_v,
            "raw_ratio_h": ratio_h_avg,
            "raw_ratio_v": ratio_v_avg,
        }

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
        """
        Classifica o olhar considerando horizontal E vertical.
        Olhar para baixo/cima conta como FRENTE (foco na tarefa ou leitura),
        não como LADOS (distração). Só olhar lateralmente é distração.
        """
        if gaze_obj.direcao_horizontal != "Frente":
            return OlharDirecaoEnum.LADOS
        # Horizontal = Frente (independente de cima/baixo = foco na tarefa)
        return OlharDirecaoEnum.FRENTE

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

    def _calcular_dimensao_comportamental(
        self,
        pose: PoseCabecaEnum,
        olhar: OlharDirecaoEnum,
        olhos_fechados: bool,
        ear: float = 0.25,
    ) -> DimensaoComportamentalEnum:
        """
        Calcula a dimensão comportamental usando pose, olhar E abertura dos olhos (EAR).

        Gradiente de atenção:
          EAR < 0.15               → olhos fechados         → DISTRAIDO
          0.15 ≤ EAR < 0.20       → olhos semicerrados      → INDEFINIDO_DISTRAIDO
          EAR ≥ 0.20 + pose/olhar → análise combinada normal
        """
        # Olhos completamente fechados: distração/sono
        if olhos_fechados or ear < 0.15:
            return DimensaoComportamentalEnum.DISTRAIDO

        # Olhos semicerrados: atenção reduzida — degrada um nível abaixo do normal
        semicerrado = ear < 0.20

        # Cabeça olhando para baixo (leitura, anotação, celular)
        if pose == PoseCabecaEnum.BAIXO:
            if olhar == OlharDirecaoEnum.FRENTE:
                return DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO if semicerrado else DimensaoComportamentalEnum.CONCENTRADO
            else:
                return DimensaoComportamentalEnum.INDEFINIDO_DISTRAIDO if semicerrado else DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO

        # Cabeça virada para o lado
        elif pose == PoseCabecaEnum.LADOS:
            if olhar == OlharDirecaoEnum.LADOS:
                return DimensaoComportamentalEnum.DISTRAIDO
            else:
                # Cabeça virada mas olhos na frente = atenção parcial
                return DimensaoComportamentalEnum.INDEFINIDO_DISTRAIDO if semicerrado else DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO

        # Cabeça de frente (caso mais comum)
        elif pose == PoseCabecaEnum.FRENTE:
            if olhar == OlharDirecaoEnum.FRENTE:
                return DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO if semicerrado else DimensaoComportamentalEnum.CONCENTRADO
            else:
                return DimensaoComportamentalEnum.DISTRAIDO if semicerrado else DimensaoComportamentalEnum.INDEFINIDO_DISTRAIDO

        # Fallback: usa apenas o olhar
        if olhar == OlharDirecaoEnum.FRENTE:
            return DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO
        return DimensaoComportamentalEnum.INDEFINIDO_DISTRAIDO

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

            # Cada extração é isolada: falha em uma não contamina as outras
            try:
                raw_emotion_str, emotion_confidence = self._get_emotion(frame, landmarks_list, shape_2d)
            except Exception as e:
                print(f"Aviso _get_emotion frame {frame_number}: {e}")
                raw_emotion_str, emotion_confidence = "neutral", 0.5

            try:
                raw_pose_dict = self._get_head_pose(landmarks_list, shape_2d)
            except Exception as e:
                print(f"Aviso _get_head_pose frame {frame_number}: {e}")
                raw_pose_dict = self._pose_from_history()

            try:
                raw_gaze_dict = self._get_gaze_direction(landmarks_list, shape_2d)
            except Exception as e:
                print(f"Aviso _get_gaze_direction frame {frame_number}: {e}")
                raw_gaze_dict = self._gaze_from_history()

            try:
                avg_ear = self._get_eye_aspect_ratio(landmarks_list)
                olhos_fechados = avg_ear < self.OLHOS_FECHADOS_THRESHOLD
            except Exception:
                avg_ear = 0.25
                olhos_fechados = False

            # Threshold reduzido: 0.2 descartava emoções sutis mas reais.
            # Agora aceita qualquer resultado com confiança mínima > 0.08.
            if emotion_confidence > 0.08:
                emocao_enum = DEEPFACE_EMOTION_MAP.get(raw_emotion_str, EmocaoEnum.NEUTRO)
            else:
                emocao_enum = EmocaoEnum.INDEFINIDO

            # Tristeza só é descartada se a confiança for muito baixa (ruído puro).
            # O filtro agressivo anterior (0.55) era duplicado com _get_emotion e
            # eliminava tristeza/raiva legítima. Apenas confiança quase zero é descartada.
            if emocao_enum == EmocaoEnum.TRISTE and emotion_confidence < 0.05:
                emocao_enum = EmocaoEnum.NEUTRO

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

            pose_enum    = self._determinar_pose_enum(pose_obj)
            olhar_enum   = self._determinar_olhar_enum(gaze_obj)
            dimensao_enum = self._calcular_dimensao_comportamental(pose_enum, olhar_enum, olhos_fechados, avg_ear)

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
            # Rosto não detectado pelo FaceMesh: usa histórico se disponível,
            # evitando gravar zeros onde havia dados confiáveis nos frames anteriores
            pose_dict = self._pose_from_history()
            gaze_dict = self._gaze_from_history()

            pose_obj = HeadPose(
                direcao_horizontal=pose_dict['direcao_horizontal'],
                raw_yaw=pose_dict['raw_yaw'],
                direcao_vertical=pose_dict['direcao_vertical'],
                raw_pitch=pose_dict['raw_pitch'],
                proximidade_z=pose_dict['proximidade_z'],
                raw_roll=pose_dict['raw_roll']
            )
            gaze_obj = GazeDirection(
                direcao_horizontal=gaze_dict['direcao_horizontal'],
                raw_ratio_h=gaze_dict['raw_ratio_h'],
                direcao_vertical=gaze_dict['direcao_vertical'],
                raw_ratio_v=gaze_dict['raw_ratio_v']
            )

            # Se há histórico, mantém o engajamento anterior (indefinido só na primeira vez)
            if self.pose_history or self.gaze_history:
                pose_enum    = self._determinar_pose_enum(pose_obj)
                olhar_enum   = self._determinar_olhar_enum(gaze_obj)
                dimensao_enum = self._calcular_dimensao_comportamental(pose_enum, olhar_enum, False)
                estimativa_enum = self._calcular_estimativa_engajamento(
                    dimensao_enum, EmocaoEnum.INDEFINIDO, 0.0
                )
            else:
                dimensao_enum   = DimensaoComportamentalEnum.INDEFINIDO
                estimativa_enum = EstimativaEngajamentoEnum.INDEFINIDO

            # Rosto não detectado pelo FaceMesh: emoção é INDEFINIDO (não NEUTRO),
            # pois não temos informação — neutro implicaria expressão detectada.
            return FrameAnalysis(
                video_id=video_id, timestamp_ms=timestamp_ms, frame_number=frame_number,
                emocao=EmocaoEnum.INDEFINIDO.value, pose_cabeca=pose_obj, olhar=gaze_obj,
                dimensao_comportamental=dimensao_enum, estimativa_engajamento=estimativa_enum,
                emotion_confidence=0.0, estado_fluxo=False
            )

    def analyze(self, frame, video_id: str, timestamp_ms: int, frame_number: int) -> FrameAnalysis:
        """
        Alias de analyze_frame com a assinatura esperada pelo VideoService.
        VideoService chama: self.face_analyzer.analyze(frame, video_id, timestamp_ms, frame_number)
        """
        return self.analyze_frame(
            frame=frame,
            timestamp_ms=timestamp_ms,
            video_id=video_id,
            frame_number=frame_number,
        )

    def detect_face_rect(self, frame) -> tuple | None:
        """
        Detecta o rosto no frame usando YuNet e retorna o retângulo (x, y, w, h)
        para ser usado como quadro fixo de recorte no VideoService.

        Retorna None se nenhum rosto for detectado com confiança suficiente.
        """
        faces = self.detect_faces(frame)

        if len(faces) == 0:
            return None

        # Pega o rosto com maior score de confiança
        best_face = max(faces, key=lambda f: f[14])  # coluna 14 = score do YuNet

        x, y, w, h = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])

        # Valida que o retângulo está dentro dos limites do frame
        frame_h, frame_w = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)

        if w <= 0 or h <= 0:
            return None

        return (x, y, w, h)