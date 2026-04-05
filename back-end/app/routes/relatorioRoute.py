import os
import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks
from starlette.responses import FileResponse
from dotenv import load_dotenv
from app.services.relatorioService import (
        RelatorioService
    )
from app.utils.DatabaseConfig import DatabaseConfig

router = APIRouter(prefix="/relatorios", tags=["Relatorios"])

# 1. Carrega as variáveis do arquivo .env para o ambiente (os.environ)
load_dotenv()

relatorio_service_instance: RelatorioService | None = None
db_config_instance: DatabaseConfig | None = None

try:
    # 4. Inicializa a configuração do DB PRIMEIRO
    db_config_instance = DatabaseConfig()
    
    relatorio_service_instance = RelatorioService(
        db_config=db_config_instance
    )
except Exception as e:
    # Se qualquer coisa acima falhar (ex: DB offline), o serviço ficará como 'None'
    print(f"ERRO FATAL: Falha ao instanciar serviços: {e}")
    # (video_service_instance permanece None)

# --- Função de Limpeza ---
def _cleanup_file(path: str):
    """Função de background para remover o arquivo após o envio."""
    try:
        if os.path.exists(path):
            os.remove(path)
            print(f"Arquivo temporário removido: {path}")
    except Exception as e:
        print(f"Erro ao remover arquivo temporário {path}: {e}")

@router.get(
    "/{video_id}",
    response_class=FileResponse, # Informa ao FastAPI que a resposta é um arquivo
    summary="Gera e baixa um relatório em PDF para um vídeo"
)
def gerar_relatorio_por_video(
    video_id: str, 
    background_tasks: BackgroundTasks
):
    """
    Gera um relatório completo em PDF com os 6 gráficos de análise
    para o `video_id` fornecido e o retorna para download.
    
    Se o vídeo não for encontrado, retorna um erro 404.
    """
    
    # 1. Buscar os dados do banco
    print(f"Iniciando relatório para: {video_id}")
    lista_analises = relatorio_service_instance.buscar_frames_do_banco_de_dados(video_id)
    
    # 2. Verificar se os dados foram encontrados
    if not lista_analises:
        raise HTTPException(
            status_code=404, 
            detail=f"Nenhum dado de análise encontrado para o video_id: {video_id}"
        )
    
    # 4. Definir um nome de arquivo temporário único (no servidor)
    temp_filename = f"temp_report_{uuid.uuid4()}.pdf"
    
    # 5. Gerar o PDF
    try:
        relatorio_service_instance.gerar_relatorio_pdf(temp_filename, lista_analises, video_id)
    except Exception as e:
        # Se a geração do PDF falhar, limpa o arquivo (se existir)
        _cleanup_file(temp_filename)
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar o arquivo PDF: {e}"
        )

    # 6. Adicionar tarefa de background para limpar o arquivo DEPOIS
    #    que a resposta for enviada.
    background_tasks.add_task(_cleanup_file, temp_filename)
    
    # 7. Buscar o nome original e formatar para o download
    nome_video = relatorio_service_instance.obter_nome_arquivo_video(video_id)
    
    # Remove a extensão original (ex: .mp4) e adiciona .pdf
    nome_base, _ = os.path.splitext(nome_video)
    nome_download = f"Relatorio_{nome_base}.pdf"

    # 8. Retornar o arquivo como resposta
    return FileResponse(
        path=temp_filename,
        media_type='application/pdf',
        # Este é o nome que o usuário verá no prompt de download
        filename=nome_download 
    )