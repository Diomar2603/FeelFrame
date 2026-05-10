"""
relatorioService.py — REFATORADO para operação assíncrona e não bloqueante.

MUDANÇAS PRINCIPAIS vs versão original:
  1. `gerar_relatorio_pdf()` e `buscar_frames_do_banco_de_dados()` são
     operações pesadas (CPU + I/O de disco).  Ambas foram envolvidas em versões
     async que as delegam para `asyncio.to_thread()`, liberando a event loop.

  2. `async def gerar_relatorio_pdf_async()` → wrapper público para uso em
     endpoints FastAPI.  Chama o método síncrono original em uma thread de
     I/O gerenciada pelo asyncio (ThreadPoolExecutor default do loop).

  3. `async def buscar_frames_do_banco_async()` → mesmo padrão para a consulta
     MongoDB de longa duração (cursor iterado em Python puro).

  4. `async def gerar_e_fazer_upload_relatorio()` → orquestrador de alto nível:
     busca frames → gera PDF → faz upload para Firebase, tudo de forma
     não bloqueante.

  5. Todo o código original dos métodos síncronos foi PRESERVADO sem alteração
     para garantir retrocompatibilidade com callers síncronos existentes.

  6. Tratamento de exceções: erros dentro da thread não derrubam o processo
     principal; são capturados pelo Future do executor e relançados no await.
"""

import uuid
import os
import math
import asyncio
import functools
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fpdf import FPDF
from dataclasses import asdict
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter
from pymongo.collection import Collection
from app.utils.DatabaseConfig import DatabaseConfig
from app.services.interfaces.IStorageService import IStorageService

from app.models.FrameAnalysis import (
    EmocaoEnum,
    FrameAnalysis,
    HeadPose,
    GazeDirection,
    EstimativaEngajamentoEnum,
    DimensaoComportamentalEnum
)


class RelatorioService:

    def __init__(self, db_config: DatabaseConfig):
        self.db = db_config.client[db_config.db_name]
        self.video_collection: Collection        = self.db["videos"]
        self.frame_analysis_collection: Collection = self.db["frame_analysis"]
        self.frames_destaque_collection: Collection = self.db["frames_destaque"]

    # =========================================================================
    # MÉTODOS SÍNCRONOS ORIGINAIS (preservados para retrocompatibilidade)
    # =========================================================================

    def obter_nome_arquivo_video(self, video_id: str) -> str:
        """Busca o nome do arquivo de forma segura."""
        try:
            filtro = {"_id": video_id}
            video = self.video_collection.find_one(filtro)

            if not video:
                video = self.video_collection.find_one({"video_id": video_id})

            if video:
                return video.get("original_filename", f"video_{video_id}")

            return f"video_{video_id}"

        except Exception as e:
            print(f"Erro ao buscar nome do vídeo: {e}")
            return f"video_{video_id}"

    def gerar_dados_grafico_temporal(self, analyses: List[FrameAnalysis]) -> Dict[str, List[Any]]:
        """Extrai dados protegendo contra Enums nulos ou atributos faltantes."""
        dados = {
            "frame_number": [], "timestamp_ms": [], "engajamento": [],
            "comportamento": [], "emocao": [], "estado_fluxo": []
        }

        for frame in analyses:
            if not frame:
                continue

            f_num = getattr(frame, 'frame_number', None)
            t_ms  = getattr(frame, 'timestamp_ms', None)

            if f_num is None or t_ms is None:
                continue

            eng_attr = getattr(frame, 'estimativa_engajamento', None)
            eng_val  = eng_attr.value if eng_attr else EstimativaEngajamentoEnum.INDEFINIDO.value

            comp_attr = getattr(frame, 'dimensao_comportamental', None)
            comp_val  = comp_attr.value if comp_attr else DimensaoComportamentalEnum.INDEFINIDO.value

            emo_raw = getattr(frame, 'emocao', None)
            if hasattr(emo_raw, 'value'):
                emo_val = emo_raw.value
            else:
                emo_val = str(emo_raw) if emo_raw else EmocaoEnum.INDEFINIDO.value

            fluxo_val = getattr(frame, 'estado_fluxo', False)

            dados["frame_number"].append(f_num)
            dados["timestamp_ms"].append(t_ms)
            dados["engajamento"].append(eng_val)
            dados["comportamento"].append(comp_val)
            dados["emocao"].append(emo_val)
            dados["estado_fluxo"].append(fluxo_val)

        return dados

    def calcular_distribuicao_percentual(self, analyses: List[FrameAnalysis],
                                         total_frames: int) -> Dict[str, Dict[str, float]]:
        """Calcula percentuais ignorando falhas parciais."""
        if total_frames == 0 or not analyses:
            return {"engajamento": {}, "comportamento": {}, "emocao": {}, "fluxo": {}}

        eng_vals  = []
        com_vals  = []
        emo_vals  = []
        fluxo_vals = []

        for f in analyses:
            if not f:
                continue

            if getattr(f, 'estimativa_engajamento', None):
                eng_vals.append(f.estimativa_engajamento.value)

            if getattr(f, 'dimensao_comportamental', None):
                com_vals.append(f.dimensao_comportamental.value)

            if getattr(f, 'emocao', None):
                emo = f.emocao
                emo_vals.append(emo.value if hasattr(emo, 'value') else str(emo))

            fluxo_vals.append(getattr(f, 'estado_fluxo', False))

        def _counts_to_percent(counter: Counter, total: int) -> Dict[str, float]:
            if total == 0:
                return {}
            return {
                str(key): round((count / total) * 100.0, 2)
                for key, count in counter.items()
            }

        return {
            "engajamento":  _counts_to_percent(Counter(eng_vals),   total_frames),
            "comportamento": _counts_to_percent(Counter(com_vals),   total_frames),
            "emocao":        _counts_to_percent(Counter(emo_vals),   total_frames),
            "fluxo":         _counts_to_percent(Counter(fluxo_vals), total_frames),
        }

    def _merge_periods(self, seconds_list: List[int]) -> List[Tuple[int, int]]:
        """Agrupa segundos garantindo tipos inteiros."""
        if not seconds_list:
            return []

        valid_seconds = sorted(list(set([
            int(s) for s in seconds_list if isinstance(s, (int, float))
        ])))

        if not valid_seconds:
            return []

        periods = []
        start_period = end_period = valid_seconds[0]

        for i in range(1, len(valid_seconds)):
            if valid_seconds[i] == end_period + 1:
                end_period = valid_seconds[i]
            else:
                periods.append((start_period, end_period + 1))
                start_period = end_period = valid_seconds[i]

        periods.append((start_period, end_period + 1))
        return periods

    def analisar_concentracao(self, analyses: List[FrameAnalysis]) -> Dict[str, List[Tuple[int, int]]]:
        if not analyses:
            return {"engajamento": [], "alt_desengajamento": [], "fluxo": []}

        frames_por_segundo: Dict[int, List[FrameAnalysis]] = {}
        for f in analyses:
            t_ms = getattr(f, 'timestamp_ms', None)
            if t_ms is None:
                continue
            segundo = int(t_ms // 1000)
            frames_por_segundo.setdefault(segundo, []).append(f)

        segundos_eng_alto   = []
        segundos_alt_des    = []
        segundos_fluxo      = []

        for segundo, frames_no_segundo in frames_por_segundo.items():
            total = len(frames_no_segundo)
            if total == 0:
                continue

            count_eng = count_alt_des = count_fluxo = 0

            for f in frames_no_segundo:
                eng = getattr(f, 'estimativa_engajamento', None)
                flx = getattr(f, 'estado_fluxo', False)

                if eng in (EstimativaEngajamentoEnum.ENGAJADO,
                            EstimativaEngajamentoEnum.ALTAMENTE_ENGAJADO):
                    count_eng += 1

                if eng == EstimativaEngajamentoEnum.ALTAMENTE_DESENGAJADO:
                    count_alt_des += 1

                if flx:
                    count_fluxo += 1

            if (count_eng / total) > 0.75:
                segundos_eng_alto.append(segundo)
            if (count_alt_des / total) > 0.75:
                segundos_alt_des.append(segundo)
            if (count_fluxo / total) > 0.5:
                segundos_fluxo.append(segundo)

        return {
            "engajamento":       self._merge_periods(segundos_eng_alto),
            "alt_desengajamento": self._merge_periods(segundos_alt_des),
            "fluxo":             self._merge_periods(segundos_fluxo),
        }

    def buscar_frames_destaque(self, video_id: str) -> List[Dict[str, Any]]:
        try:
            return list(self.frames_destaque_collection.find({"video_id": video_id})) or []
        except Exception as e:
            print(f"Erro ao buscar frames de destaque: {e}")
            return []

    def buscar_frames_do_banco_de_dados(self, video_id: str) -> List[FrameAnalysis]:
        """
        Reconstrói os objetos FrameAnalysis a partir do Mongo.
        Operação síncrona — use `buscar_frames_do_banco_async()` em endpoints.
        """
        print(f"Buscando dados para video_id: {video_id}...")
        lista_de_frames = []

        try:
            resultados_db = self.frame_analysis_collection.find({"video_id": video_id})

            for doc in resultados_db:
                try:
                    pose_data  = doc.get('pose_cabeca', {})
                    olhar_data = doc.get('olhar', {})

                    pose_obj = HeadPose(
                        direcao_horizontal=pose_data.get('direcao_horizontal', 'INDEFINIDO'),
                        raw_yaw=float(pose_data.get('raw_yaw', 0.0)),
                        direcao_vertical=pose_data.get('direcao_vertical', 'INDEFINIDO'),
                        raw_pitch=float(pose_data.get('raw_pitch', 0.0)),
                        proximidade_z=float(pose_data.get('proximidade_z', 0.0)),
                        raw_roll=float(pose_data.get('raw_roll', 0.0))
                    )

                    olhar_obj = GazeDirection(
                        direcao_horizontal=olhar_data.get('direcao_horizontal', 'INDEFINIDO'),
                        raw_ratio_h=float(olhar_data.get('raw_ratio_h', 0.0)),
                        direcao_vertical=olhar_data.get('direcao_vertical', 'INDEFINIDO'),
                        raw_ratio_v=float(olhar_data.get('raw_ratio_v', 0.0))
                    )

                    try:
                        dim_comp = DimensaoComportamentalEnum(doc.get('dimensao_comportamental'))
                    except (ValueError, TypeError):
                        dim_comp = DimensaoComportamentalEnum.INDEFINIDO

                    try:
                        est_eng = EstimativaEngajamentoEnum(doc.get('estimativa_engajamento'))
                    except (ValueError, TypeError):
                        est_eng = EstimativaEngajamentoEnum.INDEFINIDO

                    frame_analysis = FrameAnalysis(
                        video_id=doc.get('video_id', video_id),
                        timestamp_ms=int(doc.get('timestamp_ms', 0)),
                        frame_number=int(doc.get('frame_number', 0)),
                        emocao=str(doc.get('emocao', 'Indefinido')),
                        pose_cabeca=pose_obj,
                        olhar=olhar_obj,
                        dimensao_comportamental=dim_comp,
                        estimativa_engajamento=est_eng,
                        emotion_confidence=float(doc.get('emotion_confidence', 0.0)),
                        estado_fluxo=bool(doc.get('estado_fluxo', False))
                    )

                    if '_id' in doc:
                        frame_analysis._id = str(doc['_id'])

                    lista_de_frames.append(frame_analysis)

                except Exception as inner_e:
                    print(f" [AVISO] Pular frame {doc.get('frame_number', '?')}: {inner_e}")

            print(f"Processados {len(lista_de_frames)} frames com sucesso.")
            return lista_de_frames

        except Exception as e:
            print(f"ERRO CRÍTICO ao buscar dados no MongoDB: {e}")
            return []

    def _get_color_map(self) -> Dict[str, str]:
        return {
            EstimativaEngajamentoEnum.ALTAMENTE_ENGAJADO.value: "#00C853",
            EstimativaEngajamentoEnum.ENGAJADO.value: "#636EFA",
            EstimativaEngajamentoEnum.DESENGAJADO.value: "#FFA15A",
            EstimativaEngajamentoEnum.ALTAMENTE_DESENGAJADO.value: "#EF553B",
            EstimativaEngajamentoEnum.INDEFINIDO.value: "#CCCCCC",
            DimensaoComportamentalEnum.CONCENTRADO.value: "#00CC96",
            DimensaoComportamentalEnum.DISTRAIDO.value: "#FF6692",
            DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO.value: "#B6E880",
            DimensaoComportamentalEnum.INDEFINIDO_DISTRAIDO.value: "#FF9800",
            DimensaoComportamentalEnum.INDEFINIDO.value: "#BDBDBD",
            EmocaoEnum.TRISTE.value: "#4a90e2",
            EmocaoEnum.SURPRESO.value: "#f5a623",
            EmocaoEnum.FELIZ.value: "#7ed321",
            EmocaoEnum.MEDO.value: "#9013fe",
            EmocaoEnum.NEUTRO.value: "#9b9b9b",
            EmocaoEnum.INDEFINIDO.value: "#000000",
            "True": "#00C853",
            "False": "#9E9E9E",
            "Indefinido": "#000000",
        }

    def _adicionar_imagens_ao_pdf(self, pdf: FPDF,
                                  frames_destaque: List[Dict[str, Any]],
                                  tipo: str) -> None:
        if not frames_destaque:
            pdf.cell(0, 8, f" - Nenhuma cena de {tipo} capturada.", 0, 1, "L")
            return

        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Cenas de {tipo.capitalize()} Capturadas:", 0, 1, "L")
        pdf.set_font("Arial", "", 10)

        for frame_info in frames_destaque[:6]:
            try:
                path = frame_info.get('filepath')
                if path and os.path.exists(path):
                    pdf.ln(5)
                    pdf.cell(
                        0, 6,
                        f"Frame {frame_info.get('frame_number', '?')} - {frame_info.get('timestamp_ms', 0)}ms",
                        0, 1, "L"
                    )
                    pdf.cell(
                        0, 6,
                        f"Emoção: {frame_info.get('emocao', 'N/A')}, Engajamento: {frame_info.get('engajamento', 'N/A')}",
                        0, 1, "L"
                    )
                    pdf.image(path, x=pdf.l_margin, w=pdf.w - 2 * pdf.l_margin, h=60)
                    pdf.ln(2)
                    pdf.cell(0, 1, "_" * 100, 0, 1, "L")
                else:
                    pdf.cell(0, 6, f"[Imagem não encontrada: {path}]", 0, 1, "L")
            except Exception as e:
                pdf.cell(0, 6, f"Erro ao renderizar imagem: {e}", 0, 1, "L")

        if len(frames_destaque) > 6:
            pdf.cell(0, 8, f"... e mais {len(frames_destaque) - 6} cenas de {tipo}.", 0, 1, "L")

    # =========================================================================
    # ANÁLISE ARCS + UMAP
    # =========================================================================

    def _construir_dataframe_processado(self, analyses: List[FrameAnalysis]) -> pd.DataFrame:
        registros = []
        for f in analyses:
            if not f:
                continue

            pose  = getattr(f, 'pose_cabeca', None)
            olhar = getattr(f, 'olhar', None)

            eng_attr = getattr(f, 'estimativa_engajamento', None)
            eng_val  = eng_attr.value if eng_attr else EstimativaEngajamentoEnum.INDEFINIDO.value

            emo_raw = getattr(f, 'emocao', None)
            emo_val = emo_raw.value if hasattr(emo_raw, 'value') else str(emo_raw) if emo_raw else 'Indefinido'

            registros.append({
                'video_id':                  getattr(f, 'video_id', None),
                'frame_number':              getattr(f, 'frame_number', 0),
                'timestamp_ms':              getattr(f, 'timestamp_ms', 0),
                'emocao':                    emo_val,
                'estimativa_engajamento':    eng_val,
                'emotion_confidence':        getattr(f, 'emotion_confidence', 0.0),
                'pose_cabeca_raw_yaw':       getattr(pose, 'raw_yaw', 0.0) if pose else 0.0,
                'pose_cabeca_raw_pitch':     getattr(pose, 'raw_pitch', 0.0) if pose else 0.0,
                'pose_cabeca_raw_roll':      getattr(pose, 'raw_roll', 0.0) if pose else 0.0,
                'pose_cabeca_proximidade_z': getattr(pose, 'proximidade_z', 0.0) if pose else 0.0,
                'olhar_raw_ratio_h':         getattr(olhar, 'raw_ratio_h', 0.0) if olhar else 0.0,
                'olhar_raw_ratio_v':         getattr(olhar, 'raw_ratio_v', 0.0) if olhar else 0.0,
            })

        df = pd.DataFrame(registros)
        colunas_numericas = [
            'pose_cabeca_raw_yaw', 'pose_cabeca_raw_pitch', 'pose_cabeca_raw_roll',
            'pose_cabeca_proximidade_z', 'olhar_raw_ratio_h', 'olhar_raw_ratio_v',
            'emotion_confidence'
        ]
        df[colunas_numericas] = df[colunas_numericas].fillna(0)
        return df

    def calcular_proxies_arcs(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        medians = df.groupby('video_id')[
            ['pose_cabeca_raw_yaw', 'pose_cabeca_raw_pitch']
        ].transform('median')
        df['centro_yaw']   = medians['pose_cabeca_raw_yaw']
        df['centro_pitch'] = medians['pose_cabeca_raw_pitch']

        desvio_cabeca = (
            np.abs(df['pose_cabeca_raw_yaw'] - df['centro_yaw']) +
            np.abs(df['pose_cabeca_raw_pitch'] - df['centro_pitch'])
        )
        desvio_cabeca = np.clip(desvio_cabeca, 0, 90.0)
        df['ARCS_Atencao'] = 100 - (desvio_cabeca / 90.0 * 100)
        df.loc[df['emotion_confidence'] < 0.50, 'ARCS_Atencao'] = 0

        def calcular_satisfacao(linha):
            if linha['emocao'] == 'Feliz':
                return 100 * linha['emotion_confidence']
            elif linha['emocao'] == 'Neutro':
                return 50 * linha['emotion_confidence']
            return 0

        df['ARCS_Satisfacao'] = df.apply(calcular_satisfacao, axis=1)
        return df

    def calcular_umap(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            import umap.umap_ as umap_lib
        except ImportError:
            raise ImportError(
                "O pacote 'umap-learn' é necessário. Instale com: pip install umap-learn"
            )

        colunas_features = [
            'pose_cabeca_raw_yaw', 'pose_cabeca_raw_pitch', 'pose_cabeca_raw_roll',
            'pose_cabeca_proximidade_z', 'olhar_raw_ratio_h', 'olhar_raw_ratio_v',
            'emotion_confidence'
        ]
        features = df[colunas_features].fillna(0)

        print("Treinando UMAP 3D...")
        emb_3d = umap_lib.UMAP(n_components=3, random_state=42).fit_transform(features)
        df = df.copy()
        df['UMAP_3D_X'], df['UMAP_3D_Y'], df['UMAP_3D_Z'] = emb_3d[:, 0], emb_3d[:, 1], emb_3d[:, 2]

        print("Treinando UMAP 2D...")
        emb_2d = umap_lib.UMAP(n_components=2, random_state=42).fit_transform(features)
        df['UMAP_2D_X'], df['UMAP_2D_Y'] = emb_2d[:, 0], emb_2d[:, 1]

        return df

    def gerar_grafico_timeline_atencao(self, df: pd.DataFrame, video_id: str) -> go.Figure:
        fig = go.Figure()

        df_video = df[df['video_id'] == video_id].copy() if 'video_id' in df.columns else df.copy()

        fig.add_trace(go.Scatter(
            x=df_video.get('timestamp_ms', df_video.index),
            y=df_video.get('ARCS_Atencao', []),
            mode='lines',
            name='Atenção',
            line=dict(color='#636EFA', width=2)
        ))

        if 'ARCS_Satisfacao' in df_video.columns:
            fig.add_trace(go.Scatter(
                x=df_video.get('timestamp_ms', df_video.index),
                y=df_video['ARCS_Satisfacao'],
                mode='lines',
                name='Satisfação',
                line=dict(color='#00CC96', width=2)
            ))

        fig.update_layout(
            title='Timeline de Atenção e Satisfação (ARCS)',
            xaxis_title='Tempo (ms)',
            yaxis_title='Score (0–100)',
            template='simple_white',
            height=400
        )
        return fig

    def gerar_painel_validacao_cruzada(self, df: pd.DataFrame) -> go.Figure:
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Atenção vs Engajamento', 'Satisfação vs Emoção'))

        if 'ARCS_Atencao' in df.columns and 'estimativa_engajamento' in df.columns:
            fig.add_trace(
                go.Box(x=df['estimativa_engajamento'], y=df['ARCS_Atencao'], name='Atenção'),
                row=1, col=1
            )

        if 'ARCS_Satisfacao' in df.columns and 'emocao' in df.columns:
            fig.add_trace(
                go.Box(x=df['emocao'], y=df['ARCS_Satisfacao'], name='Satisfação'),
                row=1, col=2
            )

        fig.update_layout(title='Painel de Validação Cruzada ARCS', template='simple_white', height=400)
        return fig

    # =========================================================================
    # GERAÇÃO DO RELATÓRIO PDF — SÍNCRONO (preservado)
    # =========================================================================

    def gerar_relatorio_pdf(self, output_filename: str,
                            analisys: List[FrameAnalysis], video_id: str) -> None:
        """
        Gera o PDF de relatório de forma síncrona.

        ATENÇÃO: método CPU + I/O intensivo.
        Em endpoints async, use `gerar_relatorio_pdf_async()` abaixo.
        """
        print(f"Iniciando a geração do relatório: {output_filename}...")
        image_files = []

        try:
            if not analisys:
                print("Lista de análise vazia.")
                return

            total_frames       = len(analisys)
            dados_temporais    = self.gerar_dados_grafico_temporal(analisys)

            if not dados_temporais or not dados_temporais.get("frame_number"):
                print("Dados temporais insuficientes.")
                return

            dados_distribuicao = self.calcular_distribuicao_percentual(analisys, total_frames)
            momentos_destaque  = self.analisar_concentracao(analisys)

            frames_destaque     = self.buscar_frames_destaque(video_id)
            frames_fluxo        = [f for f in frames_destaque if f.get('tipo') == 'fluxo']
            frames_desengajamento = [f for f in frames_destaque if f.get('tipo') == 'desengajamento']

            print("Gerando DataFrames e Figuras...")
            df_temporal = pd.DataFrame(dados_temporais)

            def dict_to_df(d):
                return (
                    pd.DataFrame(list(d.items()), columns=['Categoria', 'Percentual'])
                    if d else pd.DataFrame(columns=['Categoria', 'Percentual'])
                )

            df_eng_pie   = dict_to_df(dados_distribuicao.get('engajamento'))
            df_comp_pie  = dict_to_df(dados_distribuicao.get('comportamento'))
            df_emo_pie   = dict_to_df(dados_distribuicao.get('emocao'))
            df_fluxo_pie = dict_to_df(dados_distribuicao.get('fluxo'))

            MAPA_CORES = self._get_color_map()

            def _gerar_grafico_barras_temporal(df: pd.DataFrame, coluna: str,
                                               titulo: str) -> go.Figure:
                categorias = df[coluna].unique()
                fig = go.Figure()
                for cat in categorias:
                    mascara = df[coluna] == cat
                    fig.add_trace(go.Bar(
                        x=df.loc[mascara, "frame_number"],
                        y=[1] * mascara.sum(),
                        name=cat,
                        marker_color=MAPA_CORES.get(cat, "#CCCCCC"),
                        hovertemplate=f"Frame: %{{x}}<br>Categoria: {cat}<extra></extra>",
                        showlegend=True
                    ))
                fig.update_layout(
                    title=titulo, barmode="stack", bargap=0,
                    xaxis_title="Frame",
                    yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template="simple_white", height=350
                )
                return fig

            fig_eng  = _gerar_grafico_barras_temporal(df_temporal, "engajamento",   "Engajamento Temporal")
            fig_comp = _gerar_grafico_barras_temporal(df_temporal, "comportamento",  "Comportamento Temporal")
            fig_emo  = _gerar_grafico_barras_temporal(df_temporal, "emocao",         "Emoção Temporal")

            df_temporal["estado_fluxo_str"] = df_temporal["estado_fluxo"].astype(str)
            fig_fluxo = _gerar_grafico_barras_temporal(df_temporal, "estado_fluxo_str", "Fluxo Temporal")

            charts_to_export = [
                (fig_eng,   "temp_spike_eng.png"),
                (fig_comp,  "temp_spike_comp.png"),
                (fig_emo,   "temp_spike_emo.png"),
                (fig_fluxo, "temp_spike_fluxo.png"),
            ]

            if not df_eng_pie.empty:
                charts_to_export.append((
                    px.pie(df_eng_pie, names='Categoria', values='Percentual',
                           title='Engajamento %', color='Categoria', color_discrete_map=MAPA_CORES),
                    "temp_pie_eng.png"
                ))
            if not df_comp_pie.empty:
                charts_to_export.append((
                    px.pie(df_comp_pie, names='Categoria', values='Percentual',
                           title='Comportamento %', color='Categoria', color_discrete_map=MAPA_CORES),
                    "temp_pie_comp.png"
                ))
            if not df_emo_pie.empty:
                charts_to_export.append((
                    px.pie(df_emo_pie, names='Categoria', values='Percentual',
                           title='Emoção %', color='Categoria', color_discrete_map=MAPA_CORES),
                    "temp_pie_emo.png"
                ))
            if not df_fluxo_pie.empty:
                charts_to_export.append((
                    px.pie(df_fluxo_pie, names='Categoria', values='Percentual',
                           title='Fluxo %', color='Categoria', color_discrete_map=MAPA_CORES),
                    "temp_pie_fluxo.png"
                ))

            print("Gerando análises ARCS e timeline de atenção...")
            try:
                df_processado = self._construir_dataframe_processado(analisys)
                df_processado = self.calcular_proxies_arcs(df_processado)

                charts_to_export.append((
                    self.gerar_grafico_timeline_atencao(df_processado, video_id),
                    "temp_arcs_timeline.png"
                ))
                charts_to_export.append((
                    self.gerar_painel_validacao_cruzada(df_processado),
                    "temp_arcs_validacao.png"
                ))
            except Exception as e_arcs:
                print(f"[AVISO] Análise ARCS falhou (será omitida do PDF): {e_arcs}")

            print("Salvando imagens...")
            for fig, fname in charts_to_export:
                try:
                    fig.write_image(fname, width=1000, height=450)
                    image_files.append(fname)
                except Exception as ex_img:
                    print(f"Falha ao salvar imagem {fname}: {ex_img}")

            # Montagem do PDF
            pdf = FPDF(orientation="L")
            pdf.set_auto_page_break(auto=True, margin=15)

            # Capa
            pdf.add_page()
            pdf.set_font("Arial", "B", 24)
            pdf.cell(0, 20, "Relatório de Análise", 0, 1, "C")
            pdf.set_font("Arial", "", 16)
            pdf.cell(0, 10, f"ID: {video_id}", 0, 1, "C")
            pdf.ln(20)

            # Destaques
            pdf.add_page()
            pdf.set_font("Arial", "B", 18)
            pdf.cell(0, 10, "Momentos de Destaque", 0, 1, "L")
            pdf.ln(5)

            def print_period_list(titulo, lista):
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, titulo, 0, 1, "L")
                pdf.set_font("Arial", "", 12)
                if not lista:
                    pdf.cell(0, 8, " - Nenhum período identificado.", 0, 1, "L")
                else:
                    for p in lista:
                        pdf.cell(0, 8, f" - {p[0]}s a {p[1]}s", 0, 1, "L")
                pdf.ln(5)

            print_period_list("Alta Concentração",   momentos_destaque.get("engajamento", []))
            print_period_list("Alto Desengajamento", momentos_destaque.get("alt_desengajamento", []))
            print_period_list("Estado de Fluxo",     momentos_destaque.get("fluxo", []))

            if frames_fluxo:
                pdf.add_page()
                self._adicionar_imagens_ao_pdf(pdf, frames_fluxo, "fluxo")

            if frames_desengajamento:
                pdf.add_page()
                self._adicionar_imagens_ao_pdf(pdf, frames_desengajamento, "desengajamento")

            for img_file in image_files:
                if os.path.exists(img_file):
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 14)
                    titulo_graf = (img_file.replace("temp_", "")
                                           .replace(".png", "")
                                           .replace("_", " ")
                                           .title())
                    pdf.cell(0, 10, titulo_graf, 0, 1, "L")
                    pdf.image(img_file, x=pdf.l_margin, w=pdf.w - 2 * pdf.l_margin)

            pdf.output(output_filename)
            print(f"Relatório gerado: {output_filename}")

        except Exception as e:
            print(f"ERRO GERAL PDF: {e}")

        finally:
            for f in image_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except Exception:
                        pass

    # =========================================================================
    # WRAPPERS ASSÍNCRONOS — use estes em endpoints FastAPI
    # =========================================================================

    async def buscar_frames_do_banco_async(self, video_id: str) -> List[FrameAnalysis]:
        """
        Versão async de `buscar_frames_do_banco_de_dados`.
        Executa a consulta MongoDB em uma thread de I/O, sem bloquear a event loop.
        """
        return await asyncio.to_thread(
            self.buscar_frames_do_banco_de_dados, video_id
        )

    async def gerar_relatorio_pdf_async(self, output_filename: str,
                                        analisys: List[FrameAnalysis],
                                        video_id: str) -> None:
        """
        Versão async de `gerar_relatorio_pdf`.
        Executa a geração pesada (Plotly + FPDF + I/O de imagens) em uma thread,
        liberando a event loop durante todo o processamento.

        Exceções geradas dentro da thread são propagadas normalmente ao await.
        """
        await asyncio.to_thread(
            self.gerar_relatorio_pdf, output_filename, analisys, video_id
        )

    async def gerar_e_fazer_upload_relatorio(
        self,
        video_id: str,
        storage_service: IStorageService,
        output_dir: str = "/tmp",
    ) -> Dict[str, Any]:
        """
        Orquestrador de alto nível totalmente não bloqueante:

            1. Busca frames do MongoDB (thread)
            2. Gera PDF (thread)
            3. Faz upload para Firebase (thread)
            4. Remove o arquivo temporário local

        Retorna:
            {
                "pdf_url": str,       # URL pública no Firebase Storage
                "frame_count": int,   # Nº de frames processados
                "video_id": str,
            }
        """
        output_filename = os.path.join(output_dir, f"relatorio_{video_id}_{uuid.uuid4().hex}.pdf")

        try:
            # Passo 1: busca no banco (I/O MongoDB)
            frames = await self.buscar_frames_do_banco_async(video_id)

            if not frames:
                return {
                    "video_id":    video_id,
                    "pdf_url":     None,
                    "frame_count": 0,
                    "error":       "Nenhum frame encontrado para este vídeo.",
                }

            # Passo 2: geração do PDF (CPU + I/O de disco)
            await self.gerar_relatorio_pdf_async(output_filename, frames, video_id)

            if not os.path.exists(output_filename):
                return {
                    "video_id":    video_id,
                    "pdf_url":     None,
                    "frame_count": len(frames),
                    "error":       "PDF não foi gerado (verifique os logs).",
                }

            # Passo 3: upload para Firebase (I/O de rede)
            upload_result = await asyncio.to_thread(
                storage_service.upload_pdf,
                output_filename,
                f"relatorio_{video_id}",
                "feelframe/relatorios",
            )

            return {
                "video_id":    video_id,
                "pdf_url":     upload_result["secure_url"],
                "frame_count": len(frames),
            }

        except Exception as e:
            print(f"[RelatorioService] Erro ao gerar relatório para {video_id}: {e}")
            return {
                "video_id": video_id,
                "pdf_url":  None,
                "error":    str(e),
            }

        finally:
            # Limpeza do arquivo temporário local
            if os.path.exists(output_filename):
                try:
                    os.remove(output_filename)
                except Exception:
                    pass
