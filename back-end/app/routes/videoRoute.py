import os
import asyncio
import uuid
from io import BytesIO
from typing import List, AsyncGenerator, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from app.services.videoService import VideoService
from app.services.interfaces.IStorageService import IStorageService
from app.services.storageFactory import create_storage_service
from app.models.StorageSource import StorageSource
from app.utils.DatabaseConfig import DatabaseConfig
from app.utils.auth_deps import get_current_user

# ---------------------------------------------------------------------------
# Schemas de entrada
# ---------------------------------------------------------------------------

class MarkerIn(BaseModel):
    time: float
    label: Optional[str] = ""
    color: Optional[str] = None


class MarkerUpdate(BaseModel):
    label: Optional[str] = None
    color: Optional[str] = None


_VALID_EMOTIONS = frozenset({"Feliz", "Triste", "Surpreso", "Medo", "Neutro", "Indefinido"})


class BulkReplaceIn(BaseModel):
    start_time: float
    end_time: float
    new_emotion: str


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/arquivos", tags=["Arquivos"])

# ---------------------------------------------------------------------------
# Inicialização dos serviços
# ---------------------------------------------------------------------------

load_dotenv()

video_service_instance: VideoService | None = None
db_config_instance: DatabaseConfig | None = None
cloudinary_instance: IStorageService | None = None

try:
    upload_subfolder = os.environ.get("VIDEO_UPLOAD_SUBFOLDER", "VideosFeelFrame")
    user_home_directory = os.path.expanduser("~")
    VIDEO_UPLOAD_DIRECTORY = os.path.join(user_home_directory, upload_subfolder)

    if not os.path.exists(VIDEO_UPLOAD_DIRECTORY):
        os.makedirs(VIDEO_UPLOAD_DIRECTORY)
        print(f"Diretório de upload criado: {VIDEO_UPLOAD_DIRECTORY}")

    db_config_instance = DatabaseConfig()
    cloudinary_instance = create_storage_service()
    video_service_instance = VideoService(
        filePath=VIDEO_UPLOAD_DIRECTORY,
        db_config=db_config_instance,
        storage=cloudinary_instance,
    )

    print(f"Serviços inicializados. DB: '{db_config_instance.db_name}', Caminho: '{VIDEO_UPLOAD_DIRECTORY}'")

except Exception as e:
    print(f"ERRO FATAL ao instanciar serviços: {e}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_video_service():
    if video_service_instance is None:
        raise HTTPException(status_code=503, detail="Serviço de vídeo não inicializado.")

def _require_cloudinary():
    if cloudinary_instance is None:
        raise HTTPException(status_code=503, detail="Serviço de armazenamento não inicializado.")


async def _register_pending_video(filename: str, user_id: str | None = None) -> str:
    """
    Cria um documento 'pending' no MongoDB antes de iniciar o processamento,
    garantindo que o video_id já exista quando o cliente começar a fazer polling.
    Retorna o video_id gerado.
    """
    video_id = str(uuid.uuid4())
    db = db_config_instance.client[db_config_instance.db_name]
    db["videos"].insert_one({
        "_id": video_id,
        "original_filename": filename,
        "user_id": user_id,
        "status": "pending",
        "storage_source": cloudinary_instance.source.value if cloudinary_instance else StorageSource.FIREBASE.value,
        "progress_percent": 0,
        "processing_message": "Aguardando início do processamento...",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    return video_id


async def _background_process_single(file_bytes: bytes, filename: str,
                                     video_id: str, user_id: str | None = None):
    """Tarefa de background: processa um único vídeo usando o video_id já registrado."""
    try:
        await video_service_instance.processFile_by_id(
            file_bytes=file_bytes,
            filename=filename,
            video_id=video_id,
            user_id=user_id,
        )
    except Exception as e:
        # Atualiza o status para 'failed' em caso de erro inesperado fora do service
        db = db_config_instance.client[db_config_instance.db_name]
        db["videos"].update_one(
            {"_id": video_id},
            {"$set": {
                "status": "failed",
                "error_message": str(e),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }},
        )


async def _background_process_multiple(files_data: list[dict], video_ids: list[str]):
    """Tarefa de background: processa múltiplos vídeos em paralelo."""
    tasks = [
        video_service_instance.processFile_by_id(
            file_bytes=f["bytes"],
            filename=f["filename"],
            video_id=vid,
        )
        for f, vid in zip(files_data, video_ids)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    db = db_config_instance.client[db_config_instance.db_name]
    for video_id, result in zip(video_ids, results):
        if isinstance(result, Exception):
            db["videos"].update_one(
                {"_id": video_id},
                {"$set": {
                    "status": "failed",
                    "error_message": str(result),
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }},
            )


# ---------------------------------------------------------------------------
# Rotas de vídeo
# ---------------------------------------------------------------------------

@router.post("/enviar/", status_code=202)
async def upload_single_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Aceita o vídeo e inicia o processamento em background.

    Retorna imediatamente com o `video_id` e status 202 (Accepted).
    Use o endpoint SSE `/files/processing-progress/{video_id}` para acompanhar
    o progresso em tempo real, ou `/files/urls/{video_id}` para buscar as URLs
    quando concluído.
    """
    _require_video_service()

    # Lê os bytes enquanto o arquivo ainda está disponível na requisição
    file_bytes = await file.read()
    user_id    = current_user["user_id"]

    # Registra o documento no MongoDB antes de enfileirar o background task,
    # evitando race condition entre o cliente começar o polling e o task iniciar.
    video_id = await _register_pending_video(filename=file.filename, user_id=user_id)

    background_tasks.add_task(
        _background_process_single,
        file_bytes=file_bytes,
        filename=file.filename,
        video_id=video_id,
        user_id=user_id,
    )

    return {
        "message": "Upload recebido. O processamento foi iniciado em background.",
        "video_id": video_id,
        "status": "pending",
        "progress_url": f"/arquivos/progresso/{video_id}",
        "result_url": f"/arquivos/dados/{video_id}",
    }


@router.post("/enviar-multiplos/", status_code=202)
async def upload_multiple_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Aceita múltiplos vídeos e inicia o processamento em batch no background.

    Retorna imediatamente com a lista de `video_id`s (um por arquivo) e status 202.
    Acompanhe cada vídeo individualmente via SSE em `/files/processing-progress/{video_id}`.
    """
    _require_video_service()

    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado.")

    MAX_CONCURRENT = 4
    if len(files) > MAX_CONCURRENT:
        raise HTTPException(
            status_code=400,
            detail=f"Máximo de {MAX_CONCURRENT} arquivos por vez. Enviados: {len(files)}",
        )

    user_id = current_user["user_id"]

    # Lê todos os bytes antes de sair da requisição
    files_data = [
        {"bytes": await f.read(), "filename": f.filename}
        for f in files
    ]

    # Registra todos os vídeos como 'pending' de forma concorrente
    video_ids = await asyncio.gather(*[
        _register_pending_video(filename=f["filename"], user_id=user_id)
        for f in files_data
    ])

    background_tasks.add_task(
        _background_process_multiple,
        files_data=files_data,
        video_ids=list(video_ids),
    )

    return {
        "message": f"{len(files)} vídeo(s) recebidos. O processamento foi iniciado em background.",
        "total_files": len(files),
        "videos": [
            {
                "video_id": vid,
                "filename": f["filename"],
                "status": "pending",
                "progress_url": f"/arquivos/progresso/{vid}",
                "result_url": f"/arquivos/dados/{vid}",
            }
            for vid, f in zip(video_ids, files_data)
        ],
    }


# ---------------------------------------------------------------------------
# SSE — progresso de processamento em tempo real
# ---------------------------------------------------------------------------

@router.get("/progresso/{video_id}")
async def processing_progress(video_id: str):
    """
    Server-Sent Events (SSE) que transmite o percentual de progresso do processamento.

    Como usar no frontend (JavaScript):
        const source = new EventSource(`/files/processing-progress/${videoId}`);
        source.onmessage = (e) => {
            const data = JSON.parse(e.data);
            console.log(data.progress_percent, data.status, data.message);
            if (data.status === 'success' || data.status === 'failed') source.close();
        };

    Eventos emitidos a cada segundo até status final (success / failed).
    """
    _require_video_service()

    async def event_generator() -> AsyncGenerator[str, None]:
        db = db_config_instance.client[db_config_instance.db_name]
        collection = db["videos"]

        while True:
            doc = collection.find_one({"_id": video_id}, {
                "status": 1,
                "progress_percent": 1,
                "processing_message": 1,
                "error_message": 1,
                "original_url": 1,
                "processed_url": 1,
            })

            if doc is None:
                payload = (
                    f"data: {{\"error\": \"video_id '{video_id}' não encontrado\"}}\n\n"
                )
                yield payload
                return

            status = doc.get("status", "processing")
            progress = doc.get("progress_percent", 0)
            message = doc.get("processing_message") or doc.get("error_message") or ""

            event_data = {
                "video_id": video_id,
                "status": status,
                "progress_percent": progress,
                "message": message,
            }

            # Quando concluído, inclui as URLs finais
            if status == "success":
                event_data["original_url"] = doc.get("original_url")
                event_data["processed_url"] = doc.get("processed_url")

            import json
            yield f"data: {json.dumps(event_data)}\n\n"

            # Encerra o stream quando o processamento terminar
            if status in ("success", "failed"):
                return

            await asyncio.sleep(1)  # Polling a cada 1 segundo

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # Essencial para Nginx não bufferizar SSE
        },
    )


# ---------------------------------------------------------------------------
# Recuperação de dados do vídeo (URLs + timeline de análise)
# ---------------------------------------------------------------------------

_NEUTRO_THRESHOLD_MS = 100  # frames abaixo deste timestamp são forçados a Neutro/Indefinido


async def _build_timeline_blocks(frames: list, field: str, default_below_threshold: str) -> list:
    """
    Recebe os documentos de frame_analysis ordenados por timestamp_ms e
    agrupa frames consecutivos com o mesmo valor de `field` em blocos
    { start, end, <field> } (tempo em segundos, com 3 casas decimais).

    Frames com timestamp_ms < _NEUTRO_THRESHOLD_MS recebem `default_below_threshold`
    em vez do valor real.
    """
    blocks = []
    if not frames:
        return blocks

    def _value(doc):
        raw = doc.get(field, "Indefinido")
        if doc.get("timestamp_ms", 0) < _NEUTRO_THRESHOLD_MS:
            return default_below_threshold
        return raw if raw else "Indefinido"

    current_value = _value(frames[0])
    current_start_ms = frames[0].get("timestamp_ms", 0)
    last_ms = current_start_ms

    for doc in frames[1:]:
        ts = doc.get("timestamp_ms", last_ms)
        val = _value(doc)

        if val != current_value:
            blocks.append({
                "start": round(current_start_ms / 1000, 3),
                "end": round(ts / 1000, 3),
                field: current_value,
            })
            current_value = val
            current_start_ms = ts

        last_ms = ts

    # Fecha o ultimo bloco
    blocks.append({
        "start": round(current_start_ms / 1000, 3),
        "end": round(last_ms / 1000, 3),
        field: current_value,
    })

    return blocks


@router.get("/dados/{video_id}")
async def get_video_data(video_id: str):
    """
    Retorna as URLs do video processado e tres timelines de analise agrupadas
    por blocos de valores consecutivos iguais.

    Resposta (status 200 - processamento concluido):
    {
        "video_id": "...",
        "status": "success",
        "original_url": "https://...",
        "processed_url": "https://...",
        "analysis": {
            "emocao":                  [{ "start": 0.0,  "end": 1.4,  "emocao": "Neutro" }, ...],
            "dimensao_comportamental": [{ "start": 0.0,  "end": 2.1,  "dimensao_comportamental": "Concentrado" }, ...],
            "estimativa_engajamento":  [{ "start": 0.0,  "end": 1.4,  "estimativa_engajamento": "Engajado" }, ...]
        }
    }

    Frames com timestamp_ms < 100 ms sao forcados a:
      - emocao                  -> "Neutro"
      - dimensao_comportamental -> "Indefinido"
      - estimativa_engajamento  -> "Indefinido"

    Retorna 202 enquanto o video ainda esta sendo processado.
    """
    _require_video_service()

    db = db_config_instance.client[db_config_instance.db_name]

    video_doc = db["videos"].find_one(
        {"_id": video_id},
        {"status": 1, "original_url": 1, "processed_url": 1,
         "error_message": 1, "storage_source": 1},
    )

    if video_doc is None:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' nao encontrado.")

    status = video_doc.get("status")

    if status == "failed":
        raise HTTPException(
            status_code=422,
            detail=f"O processamento deste video falhou: {video_doc.get('error_message')}",
        )

    if status in ("processing", "pending"):
        return JSONResponse(
            status_code=202,
            content={
                "video_id": video_id,
                "status": status,
                "storage_source": video_doc.get("storage_source", StorageSource.FIREBASE.value),
                "message": "Video ainda em processamento. Tente novamente em instantes.",
                "original_url": video_doc.get("original_url"),
                "processed_url": None,
                "analysis": None,
            },
        )

    frames = list(
        db["frame_analysis"]
        .find(
            {"video_id": video_id},
            {
                "timestamp_ms": 1,
                "frame_number": 1,
                "emocao": 1,
                "dimensao_comportamental": 1,
                "estimativa_engajamento": 1,
                "_id": 0,
            },
        )
        .sort([("timestamp_ms", 1), ("frame_number", 1)])
    )

    analysis = {
        "emocao": await _build_timeline_blocks(
            frames,
            field="emocao",
            default_below_threshold="Neutro",
        ),
        "dimensao_comportamental": await _build_timeline_blocks(
            frames,
            field="dimensao_comportamental",
            default_below_threshold="Indefinido",
        ),
        "estimativa_engajamento": await _build_timeline_blocks(
            frames,
            field="estimativa_engajamento",
            default_below_threshold="Indefinido",
        ),
    }

    return {
        "video_id": video_id,
        "status": "success",
        "storage_source": video_doc.get("storage_source", StorageSource.FIREBASE.value),
        "original_url": video_doc.get("original_url"),
        "processed_url": video_doc.get("processed_url"),
        "analysis": analysis,
    }



# ---------------------------------------------------------------------------
# Listagem de vídeos processados
# ---------------------------------------------------------------------------

@router.get("/videos/")
async def list_processed_videos(current_user: dict = Depends(get_current_user)):
    """
    Retorna todos os vídeos processados com sucesso do usuário autenticado.
    """
    _require_video_service()

    db      = db_config_instance.client[db_config_instance.db_name]
    user_id = current_user["user_id"]
    docs = list(
        db["videos"].find(
            {"status": "success", "user_id": user_id},
            {"_id": 1, "original_filename": 1},
        ).sort("created_at", -1)
    )

    videos = [
        {"video_id": doc["_id"], "original_filename": doc.get("original_filename", "")}
        for doc in docs
    ]

    return {"total": len(videos), "videos": videos}


# ---------------------------------------------------------------------------
# Exclusão de projeto (cascata)
# ---------------------------------------------------------------------------

@router.delete("/videos/{video_id}", status_code=200)
async def delete_video(video_id: str, current_user: dict = Depends(get_current_user)):
    """
    Remove permanentemente um projeto de vídeo e todos os dados associados
    (frame_analysis, markers, relatorios) — equivalente a ON DELETE CASCADE.

    Retorna 404 se o vídeo não existir ou não pertencer ao usuário autenticado.
    """
    _require_video_service()

    db      = db_config_instance.client[db_config_instance.db_name]
    user_id = current_user["user_id"]

    video_doc = db["videos"].find_one({"_id": video_id, "user_id": user_id})
    if video_doc is None:
        raise HTTPException(status_code=404, detail=f"Projeto '{video_id}' não encontrado.")

    db["frame_analysis"].delete_many({"video_id": video_id})
    db["markers"].delete_many({"video_id": video_id})
    db["relatorios"].delete_many({"video_id": video_id})
    db["videos"].delete_one({"_id": video_id})

    return {"message": f"Projeto '{video_id}' excluído com sucesso."}


# ---------------------------------------------------------------------------
# Marcadores na timeline
# ---------------------------------------------------------------------------

@router.get("/videos/{video_id}/marcadores")
async def get_video_markers(video_id: str, current_user: dict = Depends(get_current_user)):
    """
    Retorna todos os marcadores de um vídeo, ordenados por tempo (segundos).
    """
    _require_video_service()

    db      = db_config_instance.client[db_config_instance.db_name]
    user_id = current_user["user_id"]

    if db["videos"].find_one({"_id": video_id, "user_id": user_id}, {"_id": 1}) is None:
        raise HTTPException(status_code=404, detail=f"Vídeo '{video_id}' não encontrado.")

    docs = list(
        db["markers"]
        .find({"video_id": video_id}, {"_id": 1, "time": 1, "label": 1, "color": 1, "created_at": 1})
        .sort("time", 1)
    )
    markers = [
        {
            "marker_id":  d["_id"],
            "time":       d["time"],
            "label":      d.get("label", ""),
            "color":      d.get("color"),
            "created_at": d.get("created_at"),
        }
        for d in docs
    ]
    return {"video_id": video_id, "markers": markers}


@router.post("/videos/{video_id}/marcadores", status_code=201)
async def add_video_marker(
    video_id: str,
    body: MarkerIn,
    current_user: dict = Depends(get_current_user),
):
    """
    Adiciona um marcador em um instante específico do vídeo.

    Body JSON: { "time": 12.5, "label": "Ponto de atenção" }
    """
    _require_video_service()

    db      = db_config_instance.client[db_config_instance.db_name]
    user_id = current_user["user_id"]

    if db["videos"].find_one({"_id": video_id, "user_id": user_id}, {"_id": 1}) is None:
        raise HTTPException(status_code=404, detail=f"Vídeo '{video_id}' não encontrado.")

    marker_id = str(uuid.uuid4())
    now       = datetime.now(timezone.utc).isoformat()
    db["markers"].insert_one({
        "_id":        marker_id,
        "video_id":   video_id,
        "time":       body.time,
        "label":      body.label or "",
        "color":      body.color,
        "created_at": now,
    })

    return {
        "marker_id":  marker_id,
        "video_id":   video_id,
        "time":       body.time,
        "label":      body.label or "",
        "color":      body.color,
        "created_at": now,
    }


# ---------------------------------------------------------------------------
# Atualização de marcador (cor / rótulo)
# ---------------------------------------------------------------------------

@router.patch("/marcadores/{marker_id}", status_code=200)
async def update_marker(
    marker_id: str,
    body: MarkerUpdate,
    current_user: dict = Depends(get_current_user),
):
    """
    Atualiza os campos `label` e/ou `color` de um marcador existente.
    Verifica a propriedade por meio do video_id associado.
    """
    _require_video_service()

    db      = db_config_instance.client[db_config_instance.db_name]
    user_id = current_user["user_id"]

    marker_doc = db["markers"].find_one({"_id": marker_id})
    if marker_doc is None:
        raise HTTPException(status_code=404, detail=f"Marcador '{marker_id}' não encontrado.")

    video_doc = db["videos"].find_one({"_id": marker_doc["video_id"], "user_id": user_id}, {"_id": 1})
    if video_doc is None:
        raise HTTPException(status_code=403, detail="Acesso não autorizado a este marcador.")

    updates: dict = {}
    if body.label is not None:
        updates["label"] = body.label
    if body.color is not None:
        updates["color"] = body.color

    if not updates:
        raise HTTPException(status_code=400, detail="Nenhum campo para atualizar.")

    db["markers"].update_one({"_id": marker_id}, {"$set": updates})

    updated = db["markers"].find_one({"_id": marker_id})
    return {
        "marker_id":  updated["_id"],
        "video_id":   updated["video_id"],
        "time":       updated["time"],
        "label":      updated.get("label", ""),
        "color":      updated.get("color"),
        "created_at": updated.get("created_at"),
    }


# ---------------------------------------------------------------------------
# Substituição em lote de emoções por intervalo de tempo
# ---------------------------------------------------------------------------

import logging as _logging

@router.patch("/videos/{video_id}/substituir-emocoes", status_code=200)
async def bulk_replace_emotions(
    video_id: str,
    body: BulkReplaceIn,
    current_user: dict = Depends(get_current_user),
):
    """
    Substitui a emoção de todos os frames dentro do intervalo
    [start_time, end_time] (segundos) por `new_emotion`.

    Executa dentro de uma transação MongoDB (requer replica set).
    Em ambientes sem replica set (desenvolvimento), usa atualização direta
    com fallback automático.

    Retorna o total de frames atualizados e a timeline de emoções reconstruída.
    """
    _require_video_service()

    if body.new_emotion not in _VALID_EMOTIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Emoção inválida. Aceitas: {sorted(_VALID_EMOTIONS)}",
        )

    db      = db_config_instance.client[db_config_instance.db_name]
    user_id = current_user["user_id"]

    video_doc = db["videos"].find_one({"_id": video_id, "user_id": user_id, "status": "success"})
    if video_doc is None:
        raise HTTPException(status_code=404, detail=f"Vídeo '{video_id}' não encontrado.")

    start_ms = int(body.start_time * 1000)
    end_ms   = int(body.end_time   * 1000)
    query    = {"video_id": video_id, "timestamp_ms": {"$gte": start_ms, "$lte": end_ms}}
    new_val  = {"$set": {"emocao": body.new_emotion}}

    # Tenta executar dentro de uma transação ACID (requer replica set no MongoDB)
    try:
        with db_config_instance.client.start_session() as session:
            with session.start_transaction():
                result = db["frame_analysis"].update_many(query, new_val, session=session)
    except Exception as txn_err:
        # Fallback: instância standalone sem replica set
        _logging.warning("Transação não suportada; usando atualização direta. Causa: %s", txn_err)
        result = db["frame_analysis"].update_many(query, new_val)

    # Reconstrói a timeline de emoções com os dados atualizados
    frames = list(
        db["frame_analysis"]
        .find(
            {"video_id": video_id},
            {"timestamp_ms": 1, "frame_number": 1, "emocao": 1, "_id": 0},
        )
        .sort([("timestamp_ms", 1), ("frame_number", 1)])
    )

    emocao_blocks = await _build_timeline_blocks(
        frames, field="emocao", default_below_threshold="Neutro"
    )

    return {
        "updated_count": result.modified_count,
        "emocao": emocao_blocks,
    }


# ---------------------------------------------------------------------------
# Upload e armazenamento de relatórios PDF
# ---------------------------------------------------------------------------

@router.post("/relatorios/enviar/")
async def upload_relatorio_pdf(file: UploadFile = File(...)):
    """
    Faz upload de um PDF de relatório para o Cloudinary e retorna a URL pública.
    Também salva o registro no MongoDB para consultas futuras.

    Use esta rota quando o PDF já foi gerado e você quer armazená-lo.
    Para geração + upload automático, integre o CloudinaryService diretamente
    no seu serviço de geração de PDF (veja upload_relatorio_gerado abaixo).

    Resposta:
        {
            "relatorio_id": "...",
            "filename": "relatorio_xyz.pdf",
            "url": "https://res.cloudinary.com/...",
            "created_at": "..."
        }
    """
    _require_cloudinary()

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Somente arquivos .pdf são aceitos nesta rota.")

    try:
        pdf_bytes = await file.read()
        cloud_result = cloudinary_instance.upload_pdf_from_bytes(
            pdf_bytes=pdf_bytes,
            filename=file.filename,
            folder="feelframe/relatorios",
        )

        # Salva registro no MongoDB
        from datetime import datetime, timezone
        import uuid
        relatorio_id = str(uuid.uuid4())
        db = db_config_instance.client[db_config_instance.db_name]
        db["relatorios"].insert_one({
            "_id": relatorio_id,
            "filename": file.filename,
            "cloudinary_public_id": cloud_result["public_id"],
            "url": cloud_result["secure_url"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        return {
            "relatorio_id": relatorio_id,
            "filename": file.filename,
            "url": cloud_result["secure_url"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao fazer upload do relatório: {str(e)}")


@router.post("/relatorios/gerar-e-salvar/")
async def gerar_e_salvar_relatorio(video_id: str):
    """
    Exemplo de endpoint que GERA o PDF de um relatório e já o salva no Cloudinary.
    Adapte a lógica de geração ao seu serviço atual de PDF.

    Resposta:
        {
            "relatorio_id": "...",
            "video_id": "...",
            "url": "https://res.cloudinary.com/...",
        }
    """
    _require_cloudinary()

    db = db_config_instance.client[db_config_instance.db_name]
    video_doc = db["videos"].find_one({"_id": video_id})
    if not video_doc:
        raise HTTPException(status_code=404, detail=f"Vídeo '{video_id}' não encontrado.")

    try:
        # ----------------------------------------------------------------
        # SUBSTITUA O BLOCO ABAIXO PELA CHAMADA AO SEU GERADOR DE PDF
        # Exemplo: pdf_bytes = seu_servico_pdf.gerar(video_doc)
        # ----------------------------------------------------------------
        from reportlab.pdfgen import canvas as rl_canvas
        buffer = BytesIO()
        c = rl_canvas.Canvas(buffer)
        c.drawString(100, 750, f"Relatório - Vídeo ID: {video_id}")
        c.drawString(100, 720, f"Status: {video_doc.get('status')}")
        c.drawString(100, 690, f"Frames: {video_doc.get('frame_count', 0)}")
        c.save()
        pdf_bytes = buffer.getvalue()
        # ----------------------------------------------------------------

        filename = f"relatorio_{video_id}.pdf"
        cloud_result = cloudinary_instance.upload_pdf_from_bytes(
            pdf_bytes=pdf_bytes,
            filename=filename,
            folder="feelframe/relatorios",
        )

        from datetime import datetime, timezone
        import uuid
        relatorio_id = str(uuid.uuid4())
        db["relatorios"].insert_one({
            "_id": relatorio_id,
            "video_id": video_id,
            "filename": filename,
            "cloudinary_public_id": cloud_result["public_id"],
            "url": cloud_result["secure_url"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        })

        # Atualiza o documento do vídeo com a URL do relatório
        db["videos"].update_one(
            {"_id": video_id},
            {"$set": {"relatorio_url": cloud_result["secure_url"]}},
        )

        return {
            "relatorio_id": relatorio_id,
            "video_id": video_id,
            "url": cloud_result["secure_url"],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar/salvar relatório: {str(e)}")


@router.get("/relatorios/urls/{relatorio_id}")
async def get_relatorio_url(relatorio_id: str):
    """
    Retorna a URL pública do relatório PDF armazenado no Cloudinary.

    Resposta:
        {
            "relatorio_id": "...",
            "filename": "...",
            "url": "https://res.cloudinary.com/...",
            "video_id": "..."   (se vinculado a um vídeo)
        }
    """
    db = db_config_instance.client[db_config_instance.db_name]
    doc = db["relatorios"].find_one({"_id": relatorio_id})

    if not doc:
        raise HTTPException(status_code=404, detail=f"Relatório '{relatorio_id}' não encontrado.")

    return {
        "relatorio_id": relatorio_id,
        "filename": doc.get("filename"),
        "url": doc.get("url"),
        "video_id": doc.get("video_id"),
        "created_at": doc.get("created_at"),
    }


# ---------------------------------------------------------------------------
# Status do serviço
# ---------------------------------------------------------------------------

@router.get("/status/")
async def get_processing_status():
    """Retorna o status atual do serviço de processamento."""
    if video_service_instance is None:
        return {"status": "service_unavailable", "message": "Serviço não inicializado"}

    return {
        "status": "service_available",
        "message": "Serviço funcionando normalmente",
        "storage": "Firebase Storage",
        "max_concurrent_processing": 4,
        "supported_video_formats": [".mp4", ".avi", ".mov", ".mkv"],
    }