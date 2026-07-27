import os
import uuid
import requests
from fastapi import APIRouter, HTTPException, BackgroundTasks
from starlette.responses import FileResponse, Response
from dotenv import load_dotenv
from app.services.relatorioService import (
        RelatorioService
    )
from app.services.storageFactory import create_storage_service
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
async def _cleanup_file(path: str):
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
async def gerar_relatorio_por_video(
    video_id: str, 
    background_tasks: BackgroundTasks
):
    """
    Gera um relatório completo em PDF com os 6 gráficos de análise
    para o `video_id` fornecido e o retorna para download.
    
    Se o vídeo não for encontrado, retorna um erro 404.
    """
    
    print(f"Iniciando relatório para: {video_id}")

    video_doc = relatorio_service_instance.video_collection.find_one({"_id": video_id})
    if video_doc is None:
        raise HTTPException(status_code=404, detail=f"Vídeo '{video_id}' não encontrado.")

    nome_video = video_doc.get("original_filename", f"video_{video_id}")
    nome_base, _ = os.path.splitext(nome_video)
    nome_download = f"Relatorio_{nome_base}.pdf"

    # 1. Reaproveita o relatório já gerado se nenhuma emoção/marcador mudou
    #    desde a última geração (ver RelatorioService.cache_relatorio_valido).
    if relatorio_service_instance.cache_relatorio_valido(video_doc):
        print(f"Reutilizando relatório em cache para {video_id}.")
        cached = requests.get(video_doc["relatorio_cache_url"], timeout=30)
        cached.raise_for_status()
        return Response(
            content=cached.content,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{nome_download}"'},
        )

    # 2. Buscar os dados do banco (thread — não bloqueia a event loop)
    lista_analises = await relatorio_service_instance.buscar_frames_do_banco_async(video_id)
    if not lista_analises:
        raise HTTPException(
            status_code=404,
            detail=f"Nenhum dado de análise encontrado para o video_id: {video_id}"
        )

    temp_filename = f"temp_report_{uuid.uuid4()}.pdf"

    # 3. Gerar o PDF (thread — CPU/I/O intensivo, não pode rodar direto na event loop)
    try:
        await relatorio_service_instance.gerar_relatorio_pdf_async(temp_filename, lista_analises, video_id)
    except Exception as e:
        await _cleanup_file(temp_filename)
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao gerar o arquivo PDF: {e}"
        )

    # 4. Envia para o Firebase e registra o cache para os próximos downloads.
    #    Falha no cache não deve impedir a entrega do PDF já gerado.
    try:
        storage_service = create_storage_service()
        upload_result = await storage_service.upload_pdf_async(
            temp_filename, f"relatorio_{video_id}", "feelframe/relatorios"
        )
        relatorio_service_instance.salvar_cache_relatorio(video_id, upload_result["secure_url"])
    except Exception as e:
        print(f"[AVISO] Falha ao salvar cache do relatório: {e}")

    background_tasks.add_task(_cleanup_file, temp_filename)

    return FileResponse(
        path=temp_filename,
        media_type='application/pdf',
        filename=nome_download
    )