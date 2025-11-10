import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional
from enum import Enum

# --- Enums ---

class PoseCabecaEnum(str, Enum):
    FRENTE = "FRENTE"
    LADOS = "LADOS"
    BAIXO = "BAIXO"
    INDEFINIDO = "INDEFINIDO"

class OlharDirecaoEnum(str, Enum):
    FRENTE = "FRENTE"
    LADOS = "LADOS"
    INDEFINIDO = "INDEFINIDO"

class DimensaoComportamentalEnum(str, Enum):
    CONCENTRADO = "Concentrado"
    DISTRAIDO = "Distraído"
    INDEFINIDO_DISTRAIDO = "Indefinido: Parece Distraído"
    INDEFINIDO_CONCENTRADO = "Indefinido: Parece Concentrado"
    INDEFINIDO = "Indefinido"

class EstimativaEngajamentoEnum(str, Enum):
    ALTAMENTE_ENGAJADO = "Altamente Engajado"
    ENGAJADO = "Engajado"
    DESENGAJADO = "Desengajado"
    ALTAMENTE_DESENGAJADO = "Altamente Desengajado"
    INDEFINIDO = "Indefinido"

class EmocaoEnum(str, Enum):
    NEUTRO = "Neutro"
    FELIZ = "Feliz"
    SURPRESO = "Surpreso"
    MEDO = "Medo"
    TRISTE = "Triste"
    INDEFINIDO = "Indefinido" 

# --- Sub-Entidades ---

@dataclass
class HeadPose:
    """Armazena os dados de pose da cabeça."""
    direcao_horizontal: str
    raw_yaw: float
    direcao_vertical: str
    raw_pitch: float
    proximidade_z: float

@dataclass
class GazeDirection:
    """Armazena os dados de direção do olhar."""
    direcao_horizontal: str
    raw_ratio_h: float
    direcao_vertical: str
    raw_ratio_v: float

# --- Entidade Principal ---

@dataclass
class FrameAnalysis:
    """
    Entidade para armazenar o resultado completo da análise de um ÚNICO frame,
    incluindo os novos campos de engajamento e comportamento.
    """
    
    # --- Metadados de Vínculo ---
    video_id: str        
    timestamp_ms: int    
    frame_number: int    

    # --- Dados da Análise (do seu exemplo) ---
    emocao: str
    pose_cabeca: HeadPose
    olhar: GazeDirection
    dimensao_comportamental: DimensaoComportamentalEnum
    estimativa_engajamento: EstimativaEngajamentoEnum
    
    _id: str = field(default_factory=lambda: str(uuid.uuid4()))


    def to_dict(self) -> Dict[str, Any]:
        """
        Converte esta entidade em um dicionário pronto para o MongoDB,
        resolvendo corretamente os sub-objetos (dataclasses aninhadas).
        """
        return asdict(self)