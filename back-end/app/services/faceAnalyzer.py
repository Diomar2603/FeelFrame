"""
faceAnalyzer.py — REFATORADO para thread-safety e operação não bloqueante.

MUDANÇAS PRINCIPAIS:
  1. threading.Lock() em `emotion_history`, `pose_history` e `gaze_history`
     → evita condição de corrida quando múltiplas threads usam o mesmo analyzer.
  2. `static_image_mode=True` já estava correto (sem estado entre frames),
     portanto o face_mesh é stateless e pode ser chamado de threads diferentes
     sem lock adicional.
  3. Todos os métodos públicos são síncronos (CPU-bound); quem os chama é
     responsável por executá-los via `loop.run_in_executor(pool, ...)`.
  4. `analyze` / `analyze_frame` agora recebem cópias locais do frame para
     evitar que o chamador modifique o array durante o processamento.
"""

import os
import threading
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
    'angry':   EmocaoEnum.TRISTE,
    'disgust': EmocaoEnum.TRISTE,
}


class FaceAnalyzer:
    """
    Classe ajustada para evitar comportamentos indefinidos puros e
    extrair emoções secundárias quando o neutro predomina.

    THREAD-SAFETY
    -------------
    • `emotion_history`, `pose_history` e `gaze_history` são protegidos por
      `_history_lock`.  Qualquer leitura/escrita nesses deques passa pelo lock.
    • O face_mesh MediaPipe é instanciado com `static_image_mode=True`, o que
      elimina o estado interno entre chamadas — é seguro usar de threads
      diferentes sem lock adicional.
    • O detector YuNet (`cv.FaceDetectorYN`) NÃO é thread-safe internamente;
      para uso multi-thread, cada thread deve ter sua própria instância de
      FaceAnalyzer (padrão já adotado pelo VideoService com `local_analyzer`).
    """

    def __init__(self, max_faces=1):
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1
        )

        self.face_3d_model_points = np.array([
            [0.0, 0.0, 0.0], [0.0, -330.0, -65.0], [-225.0, 170.0, -135.0],
            [225.0, 170.0, -135.0], [-150.0, -150.0, -125.0], [150.0, -150.0, -125.0]
        ], dtype=np.float64)

        self.pose_landmark_indices = [1, 152, 263, 33, 291, 61]

        self.IRIS_LEFT_INDICES  = [474, 475, 476, 477]
        self.EYE_LEFT_OUTER     = 33
        self.EYE_LEFT_INNER     = 133
        self.EYE_LEFT_TOP       = 159
        self.EYE_LEFT_BOTTOM    = 145

        self.IRIS_RIGHT_INDICES = [469, 470, 471, 472]
        self.EYE_RIGHT_OUTER    = 362
        self.EYE_RIGHT_INNER    = 263
        self.EYE_RIGHT_TOP      = 386
        self.EYE_RIGHT_BOTTOM   = 374

        self.OLHOS_FECHADOS_THRESHOLD = 0.2

        self._detector = None

        # ── THREAD-SAFETY: lock único para todos os históricos ────────────────
        self._history_lock = threading.Lock()
        self.emotion_history = deque(maxlen=10)
        self.pose_history    = deque(maxlen=5)
        self.gaze_history    = deque(maxlen=5)
        # ─────────────────────────────────────────────────────────────────────

        self.SAD_TO_NEUTRAL_THRESHOLD  = 25
        self.MAX_SAD_CONFIDENCE        = 0.5
        self.SECOND_EMOTION_THRESHOLD  = 15.0

        self._init_detector()

    # ------------------------------------------------------------------
    # Inicialização do detector YuNet
    # ------------------------------------------------------------------

    def _init_detector(self):
        try:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            yunet_model_path = os.path.join(
                base_dir, "..", "utils", "yunet", "face_detection_yunet_2023mar.onnx"
            )

            if not os.path.exists(yunet_model_path):
                yunet_model_path = os.path.join(
                    "back-end", "app", "utils", "yunet", "face_detection_yunet_2023mar.onnx"
                )

            if not os.path.exists(yunet_model_path):
                alt = os.path.join(
                    base_dir, "..", "..", "utils", "yunet", "face_detection_yunet_2023mar.onnx"
                )
                if os.path.exists(alt):
                    yunet_model_path = alt
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

    # ------------------------------------------------------------------
    # Helpers para histórico thread-safe
    # ------------------------------------------------------------------

    def _append_emotion_history(self, value):
        """Adiciona ao histórico de emoções com proteção de lock."""
        with self._history_lock:
            self.emotion_history.append(value)

    def _append_pose_history(self, value):
        with self._history_lock:
            self.pose_history.append(value)

    def _append_gaze_history(self, value):
        with self._history_lock:
            self.gaze_history.append(value)

    def _read_pose_history(self):
        """Retorna cópia thread-safe do histórico de pose."""
        with self._history_lock:
            return list(self.pose_history)

    def _read_gaze_history(self):
        with self._history_lock:
            return list(self.gaze_history)

    def _read_emotion_history(self):
        with self._history_lock:
            return list(self.emotion_history)

    # ------------------------------------------------------------------
    # Detecção de faces
    # ------------------------------------------------------------------

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
            x, y   = min(all_x), min(all_y)
            x_max, y_max = max(all_x), max(all_y)

            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w_bbox = min(w - x, (x_max - x) + 2 * padding)
            h_bbox = min(h - y, (y_max - y) + 2 * padding)

            full_face = frame[y:y+h_bbox, x:x+w_bbox]
            if full_face.size > 0:
                regions.append(('full', full_face))

            eye_region_y = max(0, y - padding)
            eye_region_h = min(h - eye_region_y, int(h_bbox * 0.6))
            eye_region   = frame[eye_region_y:eye_region_y+eye_region_h, x:x+w_bbox]
            if eye_region.size > 0:
                regions.append(('eyes', eye_region))

            mouth_y = y + int(h_bbox * 0.6)
            mouth_h = min(h - mouth_y, int(h_bbox * 0.4))
            mouth_region = frame[mouth_y:mouth_y+mouth_h, x:x+w_bbox]
            if mouth_region.size > 0:
                regions.append(('mouth', mouth_region))

        except Exception as e:
            print(f"Erro ao extrair regiões: {e}")

        return regions

    def _corrigir_tendencia_tristeza(self, emotion_data, dominant_emotion):
        if dominant_emotion not in ('sad', 'angry', 'disgust'):
            return dominant_emotion

        sad_score    = (emotion_data.get('sad', 0)
                        + emotion_data.get('angry', 0) * 0.5
                        + emotion_data.get('disgust', 0) * 0.5)
        happy_score  = emotion_data.get('happy', 0)

        if happy_score > sad_score + 20:
            return 'happy'
        if sad_score < 20:
            return 'neutral'
        return dominant_emotion

    def _verificar_segunda_emocao(self, emotion_data, dominant_emotion):
        if dominant_emotion != 'neutral':
            return dominant_emotion

        sorted_emotions = sorted(
            emotion_data.items(), key=lambda x: x[1], reverse=True
        )
        if len(sorted_emotions) >= 2:
            second_emotion, second_score = sorted_emotions[1]
            if second_score >= self.SECOND_EMOTION_THRESHOLD:
                return second_emotion
        return dominant_emotion

    # ------------------------------------------------------------------
    # Análise de emoção (CPU-bound — deve ser chamada via executor)
    # ------------------------------------------------------------------

    def _get_emotion(self, frame, landmarks_list, shape):
        """
        Analisa emoção usando DeepFace.
        Operação CPU-bound pesada; não chame da event loop diretamente.
        """
        regions = self._extract_face_regions(frame, landmarks_list, shape)
        if not regions:
            return "neutral", 0.5

        best_emotion = "neutral"
        best_confidence = 0.0

        for region_name, region_img in regions:
            if region_img.size == 0:
                continue
            try:
                # DeepFace libera o GIL em partes do processamento,
                # mas o overhead de Python puro ainda bloqueia; usar executor é obrigatório.
                result = DeepFace.analyze(
                    region_img,
                    actions=['emotion'],
                    enforce_detection=False,
                    detector_backend='skip',
                    silent=True
                )
                if isinstance(result, list):
                    result = result[0]

                emotion_data   = result.get('emotion', {})
                dominant_raw   = result.get('dominant_emotion', 'neutral')

                dominant_raw   = self._corrigir_tendencia_tristeza(emotion_data, dominant_raw)
                dominant_raw   = self._verificar_segunda_emocao(emotion_data, dominant_raw)

                confidence = emotion_data.get(dominant_raw, 0) / 100.0

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_emotion    = dominant_raw

            except Exception as e:
                print(f"DeepFace erro ({region_name}): {e}")

        self._append_emotion_history((best_emotion, best_confidence))
        return best_emotion, best_confidence

    # ------------------------------------------------------------------
    # Pose e Gaze (mantidos síncronos — são leves)
    # ------------------------------------------------------------------

    def _get_head_pose(self, landmarks_list, shape):
        h, w = shape
        image_points = np.array([
            self._get_2d_coords(landmarks_list[i], shape)
            for i in self.pose_landmark_indices
        ], dtype=np.float64)

        focal_length  = w
        center        = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vec, translation_vec = cv.solvePnP(
            self.face_3d_model_points, image_points,
            camera_matrix, dist_coeffs
        )
        if not success:
            return self._pose_from_history()

        rotation_mat, _ = cv.Rodrigues(rotation_vec)
        pose_mat        = cv.hconcat([rotation_mat, translation_vec])
        _, _, _, _, _, _, euler = cv.decomposeProjectionMatrix(pose_mat)

        pitch = float(euler[0])
        yaw   = float(euler[1])
        roll  = float(euler[2])
        prox  = float(translation_vec[2][0])

        self._append_pose_history((yaw, pitch, roll, prox))

        return self._build_pose_dict(yaw, pitch, roll, prox)

    def _build_pose_dict(self, yaw, pitch, roll, prox):
        if yaw < -15:
            h_dir = PoseCabecaEnum.ESQUERDA
        elif yaw > 15:
            h_dir = PoseCabecaEnum.DIREITA
        else:
            h_dir = PoseCabecaEnum.CENTRO

        if pitch < -15:
            v_dir = PoseCabecaEnum.CIMA
        elif pitch > 15:
            v_dir = PoseCabecaEnum.BAIXO
        else:
            v_dir = PoseCabecaEnum.CENTRO

        return {
            'direcao_horizontal': h_dir,
            'raw_yaw':            yaw,
            'direcao_vertical':   v_dir,
            'raw_pitch':          pitch,
            'proximidade_z':      prox,
            'raw_roll':           roll,
        }

    def _pose_from_history(self):
        history = self._read_pose_history()
        if not history:
            return self._build_pose_dict(0.0, 0.0, 0.0, 0.0)
        yaw, pitch, roll, prox = history[-1]
        return self._build_pose_dict(yaw, pitch, roll, prox)

    def _get_gaze_direction(self, landmarks_list, shape):
        h, w = shape

        def iris_center(indices):
            pts = [landmarks_list[i] for i in indices]
            cx  = sum(p.x for p in pts) / len(pts)
            cy  = sum(p.y for p in pts) / len(pts)
            return cx * w, cy * h

        lx, ly = iris_center(self.IRIS_LEFT_INDICES)
        rx, ry = iris_center(self.IRIS_RIGHT_INDICES)

        def ratio_h(iris_x, outer_lm, inner_lm):
            ox = outer_lm.x * w
            ix = inner_lm.x * w
            span = abs(ix - ox)
            return (iris_x - ox) / span if span > 0 else 0.5

        def ratio_v(iris_y, top_lm, bot_lm):
            ty = top_lm.y * h
            by = bot_lm.y * h
            span = abs(by - ty)
            return (iris_y - ty) / span if span > 0 else 0.5

        lrh = ratio_h(lx, landmarks_list[self.EYE_LEFT_OUTER], landmarks_list[self.EYE_LEFT_INNER])
        lrv = ratio_v(ly, landmarks_list[self.EYE_LEFT_TOP],   landmarks_list[self.EYE_LEFT_BOTTOM])
        rrh = ratio_h(rx, landmarks_list[self.EYE_RIGHT_INNER], landmarks_list[self.EYE_RIGHT_OUTER])
        rrv = ratio_v(ry, landmarks_list[self.EYE_RIGHT_TOP],   landmarks_list[self.EYE_RIGHT_BOTTOM])

        avg_h = (lrh + rrh) / 2
        avg_v = (lrv + rrv) / 2

        self._append_gaze_history((avg_h, avg_v))

        return self._build_gaze_dict(avg_h, avg_v)

    def _build_gaze_dict(self, avg_h, avg_v):
        if avg_h < 0.40:
            h_dir = OlharDirecaoEnum.ESQUERDA
        elif avg_h > 0.60:
            h_dir = OlharDirecaoEnum.DIREITA
        else:
            h_dir = OlharDirecaoEnum.CENTRO

        if avg_v < 0.35:
            v_dir = OlharDirecaoEnum.CIMA
        elif avg_v > 0.65:
            v_dir = OlharDirecaoEnum.BAIXO
        else:
            v_dir = OlharDirecaoEnum.CENTRO

        return {
            'direcao_horizontal': h_dir,
            'raw_ratio_h':        avg_h,
            'direcao_vertical':   v_dir,
            'raw_ratio_v':        avg_v,
        }

    def _gaze_from_history(self):
        history = self._read_gaze_history()
        if not history:
            return self._build_gaze_dict(0.5, 0.5)
        avg_h, avg_v = history[-1]
        return self._build_gaze_dict(avg_h, avg_v)

    def _get_eye_aspect_ratio(self, landmarks_list):
        def ear(top_idx, bot_idx, outer_idx, inner_idx):
            top = landmarks_list[top_idx]
            bot = landmarks_list[bot_idx]
            outer = landmarks_list[outer_idx]
            inner = landmarks_list[inner_idx]
            vert = math.hypot(top.x - bot.x, top.y - bot.y)
            horz = math.hypot(outer.x - inner.x, outer.y - inner.y)
            return vert / horz if horz > 0 else 0.0

        left_ear  = ear(self.EYE_LEFT_TOP, self.EYE_LEFT_BOTTOM,
                        self.EYE_LEFT_OUTER, self.EYE_LEFT_INNER)
        right_ear = ear(self.EYE_RIGHT_TOP, self.EYE_RIGHT_BOTTOM,
                        self.EYE_RIGHT_OUTER, self.EYE_RIGHT_INNER)
        return (left_ear + right_ear) / 2

    # ------------------------------------------------------------------
    # Enums derivados
    # ------------------------------------------------------------------

    def _determinar_pose_enum(self, pose_obj):
        yaw   = abs(pose_obj.raw_yaw)
        pitch = abs(pose_obj.raw_pitch)
        if yaw <= 15 and pitch <= 15:
            return PoseCabecaEnum.FRENTE
        if yaw > pitch:
            return pose_obj.direcao_horizontal
        return pose_obj.direcao_vertical

    def _determinar_olhar_enum(self, gaze_obj):
        if (gaze_obj.direcao_horizontal == OlharDirecaoEnum.CENTRO and
                gaze_obj.direcao_vertical == OlharDirecaoEnum.CENTRO):
            return OlharDirecaoEnum.CENTRO
        if gaze_obj.direcao_horizontal != OlharDirecaoEnum.CENTRO:
            return gaze_obj.direcao_horizontal
        return gaze_obj.direcao_vertical

    def _calcular_dimensao_comportamental(self, pose_enum, olhar_enum,
                                          olhos_fechados, avg_ear=0.25):
        cabeca_centrada = (pose_enum == PoseCabecaEnum.FRENTE)
        olhar_centrado  = (olhar_enum == OlharDirecaoEnum.CENTRO)

        if olhos_fechados:
            return DimensaoComportamentalEnum.INDEFINIDO_DISTRAIDO

        if cabeca_centrada and olhar_centrado:
            return DimensaoComportamentalEnum.CONCENTRADO
        if cabeca_centrada or olhar_centrado:
            return DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO
        return DimensaoComportamentalEnum.DISTRAIDO

    def _detectar_estado_fluxo(self, emocao, dimensao, confidence, pose_obj, gaze_obj):
        if dimensao != DimensaoComportamentalEnum.CONCENTRADO:
            return False
        if emocao not in (EmocaoEnum.FELIZ, EmocaoEnum.SURPRESO, EmocaoEnum.NEUTRO):
            return False
        if confidence < 0.35:
            return False
        return True

    def _get_emotion_score(self, emocao, confidence):
        base = {
            EmocaoEnum.FELIZ:    40,
            EmocaoEnum.SURPRESO: 30,
            EmocaoEnum.NEUTRO:   20,
            EmocaoEnum.TRISTE:   -10,
            EmocaoEnum.MEDO:     -10,
            EmocaoEnum.INDEFINIDO: 0,
        }.get(emocao, 0)
        return int(base * min(confidence / 0.5, 1.0))

    def _calcular_estimativa_engajamento(self, dimensao, emocao, confidence,
                                         estado_fluxo=False):
        if estado_fluxo:
            return EstimativaEngajamentoEnum.ALTAMENTE_ENGAJADO

        score = 0
        dimensao_scores = {
            DimensaoComportamentalEnum.CONCENTRADO:            70,
            DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO: 55,
            DimensaoComportamentalEnum.INDEFINIDO_DISTRAIDO:   30,
            DimensaoComportamentalEnum.DISTRAIDO:               5,
            DimensaoComportamentalEnum.INDEFINIDO:             20,
        }
        score += dimensao_scores.get(dimensao, 20)
        score += self._get_emotion_score(emocao, confidence)

        if score >= 80:
            return EstimativaEngajamentoEnum.ALTAMENTE_ENGAJADO
        elif score >= 55:
            return EstimativaEngajamentoEnum.ENGAJADO
        elif score >= 30:
            return EstimativaEngajamentoEnum.DESENGAJADO
        else:
            return EstimativaEngajamentoEnum.ALTAMENTE_DESENGAJADO

    # ------------------------------------------------------------------
    # Ponto de entrada principal
    # ------------------------------------------------------------------

    def analyze_frame(self, frame, timestamp_ms: int, video_id: str,
                      frame_number: int) -> FrameAnalysis:
        """
        Analisa um único frame.

        IMPORTANTE: este método é síncrono e CPU-bound.
        Chame-o exclusivamente via `loop.run_in_executor(pool, ...)` para não
        bloquear a event loop do asyncio.

        O frame é copiado internamente para evitar condição de corrida com o
        chamador que pode reutilizar o buffer do OpenCV.
        """
        # Cópia defensiva: o chamador pode modificar o array durante o processamento
        frame = frame.copy()

        h, w, _ = frame.shape
        shape_2d = (h, w)
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks_list = face_landmarks.landmark

            try:
                raw_emotion_str, emotion_confidence = self._get_emotion(
                    frame, landmarks_list, shape_2d
                )
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
                avg_ear      = self._get_eye_aspect_ratio(landmarks_list)
                olhos_fechados = avg_ear < self.OLHOS_FECHADOS_THRESHOLD
            except Exception:
                avg_ear, olhos_fechados = 0.25, False

            emocao_enum = (
                DEEPFACE_EMOTION_MAP.get(raw_emotion_str, EmocaoEnum.NEUTRO)
                if emotion_confidence > 0.08
                else EmocaoEnum.INDEFINIDO
            )

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
            dimensao_enum = self._calcular_dimensao_comportamental(
                pose_enum, olhar_enum, olhos_fechados, avg_ear
            )
            estado_fluxo = self._detectar_estado_fluxo(
                emocao_enum, dimensao_enum, emotion_confidence, pose_obj, gaze_obj
            )
            estimativa_enum = self._calcular_estimativa_engajamento(
                dimensao_enum, emocao_enum, emotion_confidence, estado_fluxo
            )

            return FrameAnalysis(
                video_id=video_id,
                timestamp_ms=timestamp_ms,
                frame_number=frame_number,
                emocao=emocao_enum.value,
                pose_cabeca=pose_obj,
                olhar=gaze_obj,
                dimensao_comportamental=dimensao_enum,
                estimativa_engajamento=estimativa_enum,
                emotion_confidence=emotion_confidence,
                estado_fluxo=estado_fluxo
            )

        else:
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

            if self._read_pose_history() or self._read_gaze_history():
                pose_enum    = self._determinar_pose_enum(pose_obj)
                olhar_enum   = self._determinar_olhar_enum(gaze_obj)
                dimensao_enum = self._calcular_dimensao_comportamental(
                    pose_enum, olhar_enum, False
                )
                estimativa_enum = self._calcular_estimativa_engajamento(
                    dimensao_enum, EmocaoEnum.INDEFINIDO, 0.0
                )
            else:
                dimensao_enum   = DimensaoComportamentalEnum.INDEFINIDO
                estimativa_enum = EstimativaEngajamentoEnum.INDEFINIDO

            return FrameAnalysis(
                video_id=video_id,
                timestamp_ms=timestamp_ms,
                frame_number=frame_number,
                emocao=EmocaoEnum.INDEFINIDO.value,
                pose_cabeca=pose_obj,
                olhar=gaze_obj,
                dimensao_comportamental=dimensao_enum,
                estimativa_engajamento=estimativa_enum,
                emotion_confidence=0.0,
                estado_fluxo=False
            )

    def analyze(self, frame, video_id: str, timestamp_ms: int,
                frame_number: int) -> FrameAnalysis:
        """Alias de analyze_frame com a assinatura esperada pelo VideoService."""
        return self.analyze_frame(
            frame=frame,
            timestamp_ms=timestamp_ms,
            video_id=video_id,
            frame_number=frame_number,
        )

    def detect_face_rect(self, frame) -> tuple | None:
        """
        Detecta o rosto no frame usando YuNet.
        Também é CPU-bound leve; pode ser chamado via executor se necessário.
        """
        faces = self.detect_faces(frame)
        if len(faces) == 0:
            return None

        best_face = max(faces, key=lambda f: f[14])
        x, y, w, h = int(best_face[0]), int(best_face[1]), int(best_face[2]), int(best_face[3])

        frame_h, frame_w = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame_w - x)
        h = min(h, frame_h - y)

        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)
