import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any

@dataclass
class VideoMetadata:
    """
    Representa um vídeo processado e seus metadados associados.
    
    Campos sem valor padrão (obrigatórios no __init__) devem
    vir ANTES dos campos com valor padrão.
    """

    original_filename: str
    original_filepath: str
    original_filesize_bytes: int
    original_width: int
    original_height: int
    fps: float
    
    _id: str = field(default_factory=lambda: str(uuid.uuid4()))

    status: str = "processing" 
    processing_message: Optional[str] = None
    error_message: Optional[str] = None
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    processed_filepath: Optional[str] = None
    processed_width: Optional[int] = None
    processed_height: Optional[int] = None
    
    frame_count: int = 0
    duration_seconds: float = 0.0
    
    fixed_crop_rect_in_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Converte a instância da classe em um dicionário, 
        pronto para ser inserido no MongoDB.
        """
        data = asdict(self)
        
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    def mark_as_updated(self):
        """Atualiza o timestamp 'updated_at'."""
        self.updated_at = datetime.now(timezone.utc)