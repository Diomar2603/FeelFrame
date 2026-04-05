import uuid
import os
import math
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

# Importando suas entidades fornecidas
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
        self.video_collection: Collection = self.db["videos"]
        self.frame_analysis_collection: Collection = self.db["frame_analysis"]
        self.frames_destaque_collection: Collection = self.db["frames_destaque"]

    # =========================================================================
    # MÉTODOS ORIGINAIS
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
        """
        Extrai dados protegendo contra Enums nulos ou atributos faltantes.
        """
        dados = {
            "frame_number": [],
            "timestamp_ms": [],
            "engajamento": [],
            "comportamento": [],
            "emocao": [],
            "estado_fluxo": []
        }
        
        for frame in analyses:
            if not frame:
                continue

            f_num = getattr(frame, 'frame_number', None)
            t_ms = getattr(frame, 'timestamp_ms', None)

            if f_num is None or t_ms is None:
                continue

            eng_attr = getattr(frame, 'estimativa_engajamento', None)
            eng_val = eng_attr.value if eng_attr else EstimativaEngajamentoEnum.INDEFINIDO.value
            
            comp_attr = getattr(frame, 'dimensao_comportamental', None)
            comp_val = comp_attr.value if comp_attr else DimensaoComportamentalEnum.INDEFINIDO.value
            
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

    def calcular_distribuicao_percentual(self, analyses: List[FrameAnalysis], total_frames: int) -> Dict[str, Dict[str, float]]:
        """Calcula percentuais ignorando falhas parciais."""
        
        if total_frames == 0 or not analyses:
             return {"engajamento": {}, "comportamento": {}, "emocao": {}, "fluxo": {}}

        eng_vals = []
        com_vals = []
        emo_vals = []
        fluxo_vals = []

        for f in analyses:
            if not f: continue
            
            if getattr(f, 'estimativa_engajamento', None):
                eng_vals.append(f.estimativa_engajamento.value)
            
            if getattr(f, 'dimensao_comportamental', None):
                com_vals.append(f.dimensao_comportamental.value)
                
            if getattr(f, 'emocao', None):
                emo = f.emocao
                emo_vals.append(emo.value if hasattr(emo, 'value') else str(emo))
            
            fluxo_vals.append(getattr(f, 'estado_fluxo', False))

        eng_counter = Counter(eng_vals)
        com_counter = Counter(com_vals)
        emo_counter = Counter(emo_vals)
        fluxo_counter = Counter(fluxo_vals)

        def _counts_to_percent(counter: Counter, total: int) -> Dict[str, float]:
            if total == 0: return {}
            return {
                str(key): round((count / total) * 100.0, 2) 
                for key, count in counter.items()
            }

        return {
            "engajamento": _counts_to_percent(eng_counter, total_frames),
            "comportamento": _counts_to_percent(com_counter, total_frames),
            "emocao": _counts_to_percent(emo_counter, total_frames),
            "fluxo": _counts_to_percent(fluxo_counter, total_frames)
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
        start_period = valid_seconds[0]
        end_period = valid_seconds[0]

        for i in range(1, len(valid_seconds)):
            if valid_seconds[i] == end_period + 1:
                end_period = valid_seconds[i]
            else:
                periods.append((start_period, end_period + 1))
                start_period = valid_seconds[i]
                end_period = valid_seconds[i]
        
        periods.append((start_period, end_period + 1))
        return periods

    def analisar_concentracao(self, analyses: List[FrameAnalysis]) -> Dict[str, List[Tuple[int, int]]]:
        if not analyses:
            return {"engajamento": [], "alt_desengajamento": [], "fluxo": []}

        frames_por_segundo: Dict[int, List[FrameAnalysis]] = {}
        for f in analyses:
            t_ms = getattr(f, 'timestamp_ms', None)
            if t_ms is None: continue
            
            segundo = int(t_ms // 1000)
            if segundo not in frames_por_segundo:
                frames_por_segundo[segundo] = []
            frames_por_segundo[segundo].append(f)
            
        segundos_eng_alto = []
        segundos_alt_des_alto = []
        segundos_fluxo = []

        for segundo, frames_no_segundo in frames_por_segundo.items():
            total_frames = len(frames_no_segundo)
            if total_frames == 0: continue

            count_eng = 0
            count_alt_des = 0
            count_fluxo = 0

            for f in frames_no_segundo:
                eng = getattr(f, 'estimativa_engajamento', None)
                flx = getattr(f, 'estado_fluxo', False)

                if eng in (EstimativaEngajamentoEnum.ENGAJADO, EstimativaEngajamentoEnum.ALTAMENTE_ENGAJADO):
                    count_eng += 1
                
                if eng == EstimativaEngajamentoEnum.ALTAMENTE_DESENGAJADO:
                    count_alt_des += 1

                if flx:
                    count_fluxo += 1

            if (count_eng / total_frames) > 0.75:
                segundos_eng_alto.append(segundo)
            
            if (count_alt_des / total_frames) > 0.75:
                segundos_alt_des_alto.append(segundo)

            if (count_fluxo / total_frames) > 0.5:
                segundos_fluxo.append(segundo)

        return {
            "engajamento": self._merge_periods(segundos_eng_alto),
            "alt_desengajamento": self._merge_periods(segundos_alt_des_alto),
            "fluxo": self._merge_periods(segundos_fluxo)
        }

    def buscar_frames_destaque(self, video_id: str) -> List[Dict[str, Any]]:
        try:
            query = {"video_id": video_id}
            return list(self.frames_destaque_collection.find(query)) or []
        except Exception as e:
            print(f"Erro ao buscar frames de destaque: {e}")
            return []

    def buscar_frames_do_banco_de_dados(self, video_id: str) -> List[FrameAnalysis]:
        """
        Reconstrói os objetos FrameAnalysis a partir do Mongo,
        respeitando rigorosamente a nova estrutura das Dataclasses.
        """
        print(f"Buscando dados para video_id: {video_id}...")
        lista_de_frames = []
        
        try:
            query = {"video_id": video_id}
            resultados_db = self.frame_analysis_collection.find(query)
            
            for doc in resultados_db:
                try:
                    pose_data = doc.get('pose_cabeca', {})
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
                        raw_dim = doc.get('dimensao_comportamental')
                        dim_comp = DimensaoComportamentalEnum(raw_dim)
                    except (ValueError, TypeError):
                        dim_comp = DimensaoComportamentalEnum.INDEFINIDO

                    try:
                        raw_eng = doc.get('estimativa_engajamento')
                        est_eng = EstimativaEngajamentoEnum(raw_eng)
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
                    f_num = doc.get('frame_number', 'desconhecido')
                    print(f" [AVISO] Pular frame {f_num}. Erro de parsing: {inner_e}")

            print(f"Processados {len(lista_de_frames)} frames com sucesso.")
            return lista_de_frames
        
        except Exception as e:
            print(f"ERRO CRÍTICO ao buscar dados no MongoDB: {e}")
            return []

    def _get_color_map(self) -> Dict[str, str]:
        """
        Mapa de cores atualizado para os valores exatos dos Enums novos.
        """
        return {
            # Engajamento
            EstimativaEngajamentoEnum.ALTAMENTE_ENGAJADO.value: "#00C853",
            EstimativaEngajamentoEnum.ENGAJADO.value: "#2979FF",
            EstimativaEngajamentoEnum.DESENGAJADO.value: "#FF9100",
            EstimativaEngajamentoEnum.ALTAMENTE_DESENGAJADO.value: "#FF1744",
            EstimativaEngajamentoEnum.INDEFINIDO.value: "#9E9E9E",
            
            # Comportamento
            DimensaoComportamentalEnum.CONCENTRADO.value: "#00E676",
            DimensaoComportamentalEnum.DISTRAIDO.value: "#FFD600",
            DimensaoComportamentalEnum.INDEFINIDO_CONCENTRADO.value: "#64DD17",
            DimensaoComportamentalEnum.INDEFINIDO_DISTRAIDO.value: "#FF9800",
            DimensaoComportamentalEnum.INDEFINIDO.value: "#BDBDBD",
            
            # Emoções
            EmocaoEnum.FELIZ.value: "#FFEB3B",
            EmocaoEnum.SURPRESO.value: "#00E5FF",
            EmocaoEnum.MEDO.value: "#7B1FA2",
            EmocaoEnum.TRISTE.value: "#2196F3",
            EmocaoEnum.NEUTRO.value: "#9E9E9E",
            EmocaoEnum.INDEFINIDO.value: "#607D8B",
            
            # Estado de Fluxo e Fallbacks
            "True": "#00C853",
            "False": "#9E9E9E",
            "Indefinido": "#607D8B"
        }

    def _adicionar_imagens_ao_pdf(self, pdf: FPDF, frames_destaque: List[Dict[str, Any]], tipo: str):
        if not frames_destaque:
            pdf.cell(0, 8, f" - Nenhuma cena de {tipo} capturada.", 0, 1, "L")
            return

        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Cenas de {tipo.capitalize()} Capturadas:", 0, 1, "L")
        pdf.set_font("Arial", "", 10)

        for i, frame_info in enumerate(frames_destaque[:6]):
            try:
                path = frame_info.get('filepath')
                if path and os.path.exists(path):
                    pdf.ln(5)
                    pdf.cell(0, 6, f"Frame {frame_info.get('frame_number', '?')} - {frame_info.get('timestamp_ms', 0)}ms", 0, 1, "L")
                    
                    emo = frame_info.get('emocao', 'N/A')
                    eng = frame_info.get('engajamento', 'N/A')
                    
                    pdf.cell(0, 6, f"Emoção: {emo}, Engajamento: {eng}", 0, 1, "L")
                    pdf.image(path, x=pdf.l_margin, w=pdf.w - 2*pdf.l_margin, h=60)
                    pdf.ln(2)
                    pdf.cell(0, 1, "_" * 100, 0, 1, "L")
                else:
                    pdf.cell(0, 6, f"[Imagem não encontrada no disco: {path}]", 0, 1, "L")
                    
            except Exception as e:
                pdf.cell(0, 6, f"Erro ao renderizar imagem: {str(e)}", 0, 1, "L")
        
        if len(frames_destaque) > 6:
            pdf.cell(0, 8, f"... e mais {len(frames_destaque) - 6} cenas de {tipo}.", 0, 1, "L")

    # =========================================================================
    # NOVOS MÉTODOS: ANÁLISE ARCS + UMAP (Topológica)
    # =========================================================================

    def _construir_dataframe_processado(self, analyses: List[FrameAnalysis]) -> pd.DataFrame:
        """
        Constrói o DataFrame base a partir da lista de FrameAnalysis,
        desempacotando os objetos aninhados (HeadPose, GazeDirection)
        nas colunas necessárias para as análises ARCS e UMAP.
        """
        registros = []
        for f in analyses:
            if not f:
                continue

            pose = getattr(f, 'pose_cabeca', None)
            olhar = getattr(f, 'olhar', None)

            eng_attr = getattr(f, 'estimativa_engajamento', None)
            eng_val = eng_attr.value if eng_attr else EstimativaEngajamentoEnum.INDEFINIDO.value

            emo_raw = getattr(f, 'emocao', None)
            emo_val = emo_raw.value if hasattr(emo_raw, 'value') else str(emo_raw) if emo_raw else 'Indefinido'

            registros.append({
                'video_id': getattr(f, 'video_id', None),
                'frame_number': getattr(f, 'frame_number', 0),
                'timestamp_ms': getattr(f, 'timestamp_ms', 0),
                'emocao': emo_val,
                'estimativa_engajamento': eng_val,
                'emotion_confidence': getattr(f, 'emotion_confidence', 0.0),
                'pose_cabeca_raw_yaw': getattr(pose, 'raw_yaw', 0.0) if pose else 0.0,
                'pose_cabeca_raw_pitch': getattr(pose, 'raw_pitch', 0.0) if pose else 0.0,
                'pose_cabeca_raw_roll': getattr(pose, 'raw_roll', 0.0) if pose else 0.0,
                'pose_cabeca_proximidade_z': getattr(pose, 'proximidade_z', 0.0) if pose else 0.0,
                'olhar_raw_ratio_h': getattr(olhar, 'raw_ratio_h', 0.0) if olhar else 0.0,
                'olhar_raw_ratio_v': getattr(olhar, 'raw_ratio_v', 0.0) if olhar else 0.0,
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
        """
        Calcula os proxies contínuos ARCS (Atenção e Satisfação) no DataFrame.

        - ARCS_Atencao: baseado no desvio angular da cabeça em relação à
          posição mediana calibrada por vídeo, normalizado entre 0 e 100.
          Frames com emotion_confidence < 0.5 recebem Atenção = 0.
        - ARCS_Satisfacao: mapeamento da emoção ponderado pela confiança
          (Feliz → 100 * conf, Neutro → 50 * conf, demais → 0).
        """
        df = df.copy()

        # --- Atenção (A) ---
        # Calibração geométrica individual por vídeo (mediana como ponto de referência)
        medians = df.groupby('video_id')[['pose_cabeca_raw_yaw', 'pose_cabeca_raw_pitch']].transform('median')
        df['centro_yaw'] = medians['pose_cabeca_raw_yaw']
        df['centro_pitch'] = medians['pose_cabeca_raw_pitch']

        desvio_cabeca = (
            np.abs(df['pose_cabeca_raw_yaw'] - df['centro_yaw']) +
            np.abs(df['pose_cabeca_raw_pitch'] - df['centro_pitch'])
        )

        limite_fisico = 90.0
        desvio_cabeca = np.clip(desvio_cabeca, 0, limite_fisico)
        df['ARCS_Atencao'] = 100 - (desvio_cabeca / limite_fisico * 100)

        # Trava de confiança: frames sem detecção facial confiável → Atenção = 0
        df.loc[df['emotion_confidence'] < 0.50, 'ARCS_Atencao'] = 0

        # --- Satisfação (S) ---
        def calcular_satisfacao(linha):
            if linha['emocao'] == 'Feliz':
                return 100 * linha['emotion_confidence']
            elif linha['emocao'] == 'Neutro':
                return 50 * linha['emotion_confidence']
            else:
                return 0

        df['ARCS_Satisfacao'] = df.apply(calcular_satisfacao, axis=1)

        return df

    def calcular_umap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica redução de dimensionalidade UMAP (3D e 2D) sobre as features
        físicas dos frames.

        Dependência: `pip install umap-learn`

        Retorna o DataFrame original acrescido das colunas:
        UMAP_3D_X, UMAP_3D_Y, UMAP_3D_Z, UMAP_2D_X, UMAP_2D_Y
        """
        try:
            import umap.umap_ as umap_lib
        except ImportError:
            raise ImportError(
                "O pacote 'umap-learn' é necessário para esta análise. "
                "Instale com: pip install umap-learn"
            )

        colunas_features = [
            'pose_cabeca_raw_yaw', 'pose_cabeca_raw_pitch', 'pose_cabeca_raw_roll',
            'pose_cabeca_proximidade_z', 'olhar_raw_ratio_h', 'olhar_raw_ratio_v',
            'emotion_confidence'
        ]

        features = df[colunas_features].fillna(0)

        print("Treinando UMAP 3D...")
        redutor_3d = umap_lib.UMAP(n_components=3, random_state=42)
        emb_3d = redutor_3d.fit_transform(features)
        df = df.copy()
        df['UMAP_3D_X'] = emb_3d[:, 0]
        df['UMAP_3D_Y'] = emb_3d[:, 1]
        df['UMAP_3D_Z'] = emb_3d[:, 2]

        print("Treinando UMAP 2D...")
        redutor_2d = umap_lib.UMAP(n_components=2, n_neighbors=50, min_dist=0.1, random_state=42)
        emb_2d = redutor_2d.fit_transform(features)
        df['UMAP_2D_X'] = emb_2d[:, 0]
        df['UMAP_2D_Y'] = emb_2d[:, 1]

        print("Modelos topológicos computados.")
        return df

    def gerar_grafico_umap_3d(self, df: pd.DataFrame) -> go.Figure:
        """
        Gera o gráfico de dispersão 3D no espaço latente UMAP,
        colorido pelo proxy contínuo de Atenção (ARCS_Atencao).
        Requer que df contenha as colunas UMAP_3D_* e ARCS_Atencao.
        """
        fig = px.scatter_3d(
            df,
            x='UMAP_3D_X', y='UMAP_3D_Y', z='UMAP_3D_Z',
            color='ARCS_Atencao',
            color_continuous_scale='Viridis',
            title="Espaço Latente 3D: Proxy de Atenção (ARCS)"
        )
        fig.update_traces(marker=dict(size=3, opacity=0.8))
        return fig

    def gerar_grafico_densidade_kde(self, df: pd.DataFrame) -> go.Figure:
        """
        Gera o mapa topográfico de densidade (KDE) no espaço latente 2D,
        mostrando onde os comportamentos se concentram.
        Requer as colunas UMAP_2D_X e UMAP_2D_Y.
        """
        fig = px.density_contour(
            df,
            x='UMAP_2D_X', y='UMAP_2D_Y',
            title="Mapa Topográfico do Engajamento (Densidade KDE)",
            color_discrete_sequence=['#2C3E50']
        )
        fig.update_traces(contours_coloring="fill", contours_showlabels=False)
        fig.update_layout(template="simple_white")
        return fig

    def gerar_grafico_trajetoria_individual(self, df: pd.DataFrame, video_id: str) -> go.Figure:
        """
        Gera a trajetória comportamental suavizada de um único estudante
        no espaço latente 2D, com anotações de início e fim do vídeo.
        Requer as colunas UMAP_2D_X, UMAP_2D_Y e ARCS_Atencao.
        """
        df_aluno = df[df['video_id'] == video_id].sort_values('frame_number').copy()
        df_aluno['frame_relativo'] = range(1, len(df_aluno) + 1)

        # Suavização (~1 segundo / 30 frames) para remover ruído micro-motor
        df_aluno['UMAP_2D_X_Smooth'] = df_aluno['UMAP_2D_X'].rolling(window=30, center=True).mean()
        df_aluno['UMAP_2D_Y_Smooth'] = df_aluno['UMAP_2D_Y'].rolling(window=30, center=True).mean()
        df_aluno = df_aluno.dropna(subset=['UMAP_2D_X_Smooth', 'UMAP_2D_Y_Smooth'])

        fig = go.Figure()

        # Linha de conexão cronológica (o Caminho)
        fig.add_trace(go.Scatter(
            x=df_aluno['UMAP_2D_X_Smooth'], y=df_aluno['UMAP_2D_Y_Smooth'],
            mode='lines', line=dict(color='lightgray', width=1),
            showlegend=False, hoverinfo='skip'
        ))

        # Pontos coloridos pelo proxy de Atenção (o Estado)
        fig.add_trace(go.Scatter(
            x=df_aluno['UMAP_2D_X_Smooth'], y=df_aluno['UMAP_2D_Y_Smooth'],
            mode='markers',
            marker=dict(
                size=5,
                color=df_aluno['ARCS_Atencao'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Atenção (0-100)")
            ),
            text=df_aluno['frame_relativo'],
            hovertemplate="Frame Relativo: %{text}<br>Atenção: %{marker.color:.1f}<extra></extra>"
        ))

        # Anotações de início e fim
        inicio = df_aluno.iloc[0]
        fim = df_aluno.iloc[-1]

        fig.add_annotation(
            x=inicio['UMAP_2D_X_Smooth'], y=inicio['UMAP_2D_Y_Smooth'],
            text="INÍCIO DO VÍDEO", showarrow=True, arrowhead=2,
            arrowsize=1, arrowwidth=2, ax=-50, ay=-30,
            font=dict(color="black", size=10)
        )
        fig.add_annotation(
            x=fim['UMAP_2D_X_Smooth'], y=fim['UMAP_2D_Y_Smooth'],
            text="FIM DO VÍDEO", showarrow=True, arrowhead=2,
            arrowsize=1, arrowwidth=2, ax=50, ay=30,
            font=dict(color="black", size=10)
        )

        fig.update_layout(
            title=f"Trajetória Comportamental no Espaço Latente (Vídeo: {video_id[-6:]})",
            template="simple_white",
            xaxis_title="Dimensão Latente 1",
            yaxis_title="Dimensão Latente 2"
        )
        return fig

    def gerar_grafico_timeline_atencao(self, df: pd.DataFrame, video_id: str) -> go.Figure:
        """
        Gera a série temporal suavizada do proxy de Atenção (ARCS_Atencao)
        para um único vídeo, com zonas de referência (Flow e Desatenção).
        """
        df_aluno = df[df['video_id'] == video_id].sort_values('frame_number').copy()
        df_aluno['frame_relativo'] = range(1, len(df_aluno) + 1)
        df_aluno['Atencao_Smooth'] = df_aluno['ARCS_Atencao'].rolling(window=30, center=True).mean()
        df_aluno = df_aluno.dropna(subset=['Atencao_Smooth'])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_aluno['frame_relativo'],
            y=df_aluno['Atencao_Smooth'],
            mode='lines',
            line=dict(color='#2C3E50', width=2.5),
            name='Atenção Suavizada',
            fill='tozeroy',
            fillcolor='rgba(44, 62, 80, 0.1)'
        ))

        # Zona de Flow (>= 80) e limiar de desatenção (<= 20)
        fig.add_hline(
            y=80, line_dash="dash", line_color="#27AE60",
            annotation_text="Zona de Flow (Alta Atenção)",
            annotation_position="top left"
        )
        fig.add_hline(
            y=20, line_dash="dot", line_color="#E74C3C",
            annotation_text="Desatenção / Perda de Rastreamento",
            annotation_position="bottom left"
        )

        fig.update_layout(
            title=f"Série Temporal: Jornada de Atenção (Vídeo: {video_id[-6:]})",
            xaxis_title="Tempo (Frame Relativo)",
            yaxis_title="Proxy Contínuo de Atenção (0 a 100)",
            yaxis_range=[-5, 110],
            template="simple_white",
            hovermode="x unified"
        )
        return fig

    def gerar_painel_trajetorias_todos_estudantes(self, df: pd.DataFrame) -> go.Figure:
        """
        Gera um painel comparativo (grid) com as séries temporais de Atenção
        para todos os vídeos/estudantes, incluindo zonas coloridas de engajamento.
        Salva também um HTML interativo em 'painel_trajetorias_arcs.html'.
        """
        video_ids = df['video_id'].dropna().unique()
        total_videos = len(video_ids)
        colunas = 3
        linhas = math.ceil(total_videos / colunas)

        titulos_subplots = [f"Vídeo: {vid[-6:]}" for vid in video_ids]
        fig_grid = make_subplots(
            rows=linhas, cols=colunas,
            subplot_titles=titulos_subplots,
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )

        for indice, id_alvo in enumerate(video_ids):
            linha_atual = (indice // colunas) + 1
            coluna_atual = (indice % colunas) + 1

            df_aluno = df[df['video_id'] == id_alvo].sort_values('frame_number').copy()
            df_aluno['frame_relativo'] = range(1, len(df_aluno) + 1)
            df_aluno['Atencao_Smooth'] = df_aluno['ARCS_Atencao'].rolling(window=30, center=True).mean()
            df_aluno = df_aluno.dropna(subset=['Atencao_Smooth'])

            # Zonas de cores de fundo
            fig_grid.add_hrect(
                y0=80, y1=100, line_width=0, fillcolor="#27AE60", opacity=0.15,
                row=linha_atual, col=coluna_atual
            )
            fig_grid.add_hrect(
                y0=20, y1=80, line_width=0, fillcolor="#F1C40F", opacity=0.05,
                row=linha_atual, col=coluna_atual
            )
            fig_grid.add_hrect(
                y0=0, y1=20, line_width=0, fillcolor="#E74C3C", opacity=0.15,
                row=linha_atual, col=coluna_atual
            )

            fig_grid.add_trace(go.Scatter(
                x=df_aluno['frame_relativo'],
                y=df_aluno['Atencao_Smooth'],
                mode='lines',
                line=dict(color='#2C3E50', width=2),
                showlegend=False,
                hovertemplate="Frame: %{x}<br>Atenção: %{y:.1f}<extra></extra>"
            ), row=linha_atual, col=coluna_atual)

            fig_grid.add_hline(y=80, line_dash="dash", line_color="#27AE60", line_width=1.5, row=linha_atual, col=coluna_atual)
            fig_grid.add_hline(y=20, line_dash="dot", line_color="#E74C3C", line_width=1.5, row=linha_atual, col=coluna_atual)

        fig_grid.update_yaxes(range=[-5, 105])
        fig_grid.update_layout(
            title_text="Painel de Trajetórias de Atenção por Estudante (Zonas de Engajamento)",
            height=300 * linhas,
            width=1200,
            template="simple_white",
            hovermode="x unified"
        )

        fig_grid.write_html("painel_trajetorias_arcs.html")
        print("Painel salvo em: painel_trajetorias_arcs.html")

        return fig_grid

    def gerar_painel_validacao_cruzada(self, df: pd.DataFrame) -> go.Figure:
        """
        Gera o painel de validação cruzada: curva ARCS de Atenção suavizada
        (linha de base) com pontos coloridos pelas categorias de engajamento
        (estimativa_engajamento), permitindo comparar as duas metodologias.
        Salva também um HTML interativo em 'validacao_cruzada_completa.html'.
        """
        mapa_de_cores = {
            'Altamente Engajado': '#00CC96',
            'Engajado': '#636EFA',
            'Indefinido': '#CCCCCC',
            'Desengajado': '#FFA15A',
            'Altamente Desengajado': '#EF553B'
        }

        video_ids = df['video_id'].dropna().unique()
        colunas = 2
        linhas = math.ceil(len(video_ids) / colunas)

        fig_grid = make_subplots(
            rows=linhas, cols=colunas,
            subplot_titles=[f"Vídeo: {vid[-6:]}" for vid in video_ids],
            vertical_spacing=0.05,
            horizontal_spacing=0.08
        )

        # Traces de legenda (marcadores fantasma apenas para exibir a legenda)
        for nome, cor in mapa_de_cores.items():
            fig_grid.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                marker=dict(size=10, color=cor),
                legendgroup="Engajamento",
                name=nome,
                showlegend=True
            ), row=1, col=1)

        for indice, id_alvo in enumerate(video_ids):
            linha_atual = (indice // colunas) + 1
            coluna_atual = (indice % colunas) + 1

            df_aluno = df[df['video_id'] == id_alvo].sort_values('frame_number').copy()
            df_aluno['frame_relativo'] = range(1, len(df_aluno) + 1)
            df_aluno['Atencao_Smooth'] = df_aluno['ARCS_Atencao'].rolling(window=30, center=True).mean()

            cores_pontos = df_aluno['estimativa_engajamento'].map(mapa_de_cores).fillna('#000000')

            # Zonas de cores de fundo
            fig_grid.add_hrect(y0=80, y1=100, fillcolor="#27AE60", opacity=0.1, line_width=0, row=linha_atual, col=coluna_atual)
            fig_grid.add_hrect(y0=20, y1=80, fillcolor="#F1C40F", opacity=0.05, line_width=0, row=linha_atual, col=coluna_atual)
            fig_grid.add_hrect(y0=0, y1=20, fillcolor="#E74C3C", opacity=0.1, line_width=0, row=linha_atual, col=coluna_atual)

            # Linha de base (Tendência ARCS)
            fig_grid.add_trace(go.Scatter(
                x=df_aluno['frame_relativo'],
                y=df_aluno['Atencao_Smooth'],
                mode='lines',
                line=dict(color='rgba(44, 62, 80, 0.4)', width=1.5),
                showlegend=False,
                hoverinfo='skip'
            ), row=linha_atual, col=coluna_atual)

            # Pontos coloridos pelas categorias de engajamento
            fig_grid.add_trace(go.Scatter(
                x=df_aluno['frame_relativo'],
                y=df_aluno['Atencao_Smooth'],
                mode='markers',
                marker=dict(size=4, color=cores_pontos, opacity=0.8),
                customdata=df_aluno['estimativa_engajamento'],
                hovertemplate="Frame: %{x}<br>Atenção (ARCS): %{y:.1f}<br>Engajamento: %{customdata}<extra></extra>",
                showlegend=False
            ), row=linha_atual, col=coluna_atual)

            fig_grid.add_hline(y=80, line_dash="dash", line_color="#27AE60", line_width=1, row=linha_atual, col=coluna_atual)
            fig_grid.add_hline(y=20, line_dash="dot", line_color="#E74C3C", line_width=1, row=linha_atual, col=coluna_atual)

        fig_grid.update_yaxes(range=[-5, 105])
        fig_grid.update_layout(
            title_text="Validação Cruzada: Curva ARCS vs. Categorização de Engajamento",
            height=400 * linhas,
            width=1300,
            template="simple_white",
            legend=dict(
                title="Categorias de Engajamento (Cores dos Pontos)",
                orientation="v", yanchor="top", y=1, xanchor="left", x=1.02
            )
        )

        fig_grid.write_html("validacao_cruzada_completa.html")
        print("Validação cruzada salva em: validacao_cruzada_completa.html")

        return fig_grid

    # =========================================================================
    # GERAÇÃO DO RELATÓRIO PDF (ORIGINAL + ANÁLISES ARCS/UMAP)
    # =========================================================================

    def gerar_relatorio_pdf(self, output_filename: str, analisys: List[FrameAnalysis], video_id: str):
        print(f"Iniciando a geração do relatório: {output_filename}...")
        image_files = []

        try:
            if not analisys:
                print("Lista de análise vazia.")
                return

            total_frames = len(analisys)
            dados_temporais = self.gerar_dados_grafico_temporal(analisys)
            
            if not dados_temporais or not dados_temporais.get("frame_number"):
                print("Dados temporais insuficientes para gerar gráficos.")
                return

            dados_distribuicao = self.calcular_distribuicao_percentual(analisys, total_frames)
            momentos_destaque = self.analisar_concentracao(analisys)
            
            frames_destaque = self.buscar_frames_destaque(video_id)
            frames_fluxo = [f for f in frames_destaque if f.get('tipo') == 'fluxo']
            frames_desengajamento = [f for f in frames_destaque if f.get('tipo') == 'desengajamento']
            
            # --- Criação dos Gráficos Originais ---
            print("Gerando DataFrames e Figuras...")
            df_temporal = pd.DataFrame(dados_temporais)
            
            def dict_to_df(d):
                return pd.DataFrame(list(d.items()), columns=['Categoria', 'Percentual']) if d else pd.DataFrame(columns=['Categoria', 'Percentual'])

            df_eng_pie = dict_to_df(dados_distribuicao.get('engajamento'))
            df_comp_pie = dict_to_df(dados_distribuicao.get('comportamento'))
            df_emo_pie = dict_to_df(dados_distribuicao.get('emocao'))
            df_fluxo_pie = dict_to_df(dados_distribuicao.get('fluxo'))

            MAPA_CORES = self._get_color_map()
            common_marker = dict(symbol='line-ns', size=12, line=dict(width=2), opacity=0.8)

            fig_eng = px.scatter(df_temporal, x="frame_number", y="engajamento", color="engajamento", color_discrete_map=MAPA_CORES, title="Engajamento Temporal")
            fig_eng.update_traces(marker=common_marker)

            fig_comp = px.scatter(df_temporal, x="frame_number", y="comportamento", color="comportamento", color_discrete_map=MAPA_CORES, title="Comportamento Temporal")
            fig_comp.update_traces(marker=common_marker)

            fig_emo = px.scatter(df_temporal, x="frame_number", y="emocao", color="emocao", color_discrete_map=MAPA_CORES, title="Emoção Temporal")
            fig_emo.update_traces(marker=common_marker)

            df_temporal["estado_fluxo_str"] = df_temporal["estado_fluxo"].astype(str)
            fig_fluxo = px.scatter(df_temporal, x="frame_number", y="estado_fluxo_str", color="estado_fluxo_str", color_discrete_map=MAPA_CORES, title="Fluxo Temporal")
            fig_fluxo.update_traces(marker=common_marker)

            charts_to_export = [
                (fig_eng, "temp_spike_eng.png"),
                (fig_comp, "temp_spike_comp.png"),
                (fig_emo, "temp_spike_emo.png"),
                (fig_fluxo, "temp_spike_fluxo.png")
            ]

            if not df_eng_pie.empty:
                charts_to_export.append((px.pie(df_eng_pie, names='Categoria', values='Percentual', title='Engajamento %', color='Categoria', color_discrete_map=MAPA_CORES), "temp_pie_eng.png"))
            if not df_comp_pie.empty:
                charts_to_export.append((px.pie(df_comp_pie, names='Categoria', values='Percentual', title='Comportamento %', color='Categoria', color_discrete_map=MAPA_CORES), "temp_pie_comp.png"))
            if not df_emo_pie.empty:
                charts_to_export.append((px.pie(df_emo_pie, names='Categoria', values='Percentual', title='Emoção %', color='Categoria', color_discrete_map=MAPA_CORES), "temp_pie_emo.png"))
            if not df_fluxo_pie.empty:
                charts_to_export.append((px.pie(df_fluxo_pie, names='Categoria', values='Percentual', title='Fluxo %', color='Categoria', color_discrete_map=MAPA_CORES), "temp_pie_fluxo.png"))

            # --- Geração dos Gráficos ARCS ---
            print("Gerando análises ARCS e timeline de atenção...")
            try:
                df_processado = self._construir_dataframe_processado(analisys)
                df_processado = self.calcular_proxies_arcs(df_processado)

                fig_timeline = self.gerar_grafico_timeline_atencao(df_processado, video_id)
                charts_to_export.append((fig_timeline, "temp_arcs_timeline.png"))

                fig_validacao = self.gerar_painel_validacao_cruzada(df_processado)
                charts_to_export.append((fig_validacao, "temp_arcs_validacao.png"))

            except Exception as e_arcs:
                print(f"[AVISO] Análise ARCS falhou (será omitida do PDF): {e_arcs}")

            print("Salvando imagens...")
            for fig, fname in charts_to_export:
                try:
                    fig.write_image(fname, width=1000, height=450)
                    image_files.append(fname)
                except Exception as ex_img:
                    print(f"Falha ao salvar imagem {fname}: {ex_img}")

            # --- Montagem do PDF ---
            pdf = FPDF(orientation="L") 
            pdf.set_auto_page_break(auto=True, margin=15)

            # Capa
            pdf.add_page()
            pdf.set_font("Arial", "B", 24)
            pdf.cell(0, 20, "Relatório de Análise", 0, 1, "C")
            pdf.set_font("Arial", "", 16)
            pdf.cell(0, 10, f"ID: {video_id}", 0, 1, "C")
            pdf.ln(20)

            # Seção de Textos (Destaques)
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

            print_period_list("Alta Concentração", momentos_destaque.get("engajamento", []))
            print_period_list("Alto Desengajamento", momentos_destaque.get("alt_desengajamento", []))
            print_period_list("Estado de Fluxo", momentos_destaque.get("fluxo", []))

            # Seção de Imagens Capturadas
            if frames_fluxo:
                pdf.add_page()
                self._adicionar_imagens_ao_pdf(pdf, frames_fluxo, "fluxo")
            
            if frames_desengajamento:
                pdf.add_page()
                self._adicionar_imagens_ao_pdf(pdf, frames_desengajamento, "desengajamento")

            # Seção de Gráficos
            for img_file in image_files:
                if os.path.exists(img_file):
                    pdf.add_page()
                    pdf.set_font("Arial", "B", 14)
                    titulo_graf = img_file.replace("temp_", "").replace(".png", "").replace("_", " ").title()
                    pdf.cell(0, 10, titulo_graf, 0, 1, "L")
                    pdf.image(img_file, x=pdf.l_margin, w=pdf.w - 2*pdf.l_margin)

            pdf.output(output_filename)
            print(f"Relatório gerado: {output_filename}")

        except Exception as e:
            print(f"ERRO GERAL PDF: {e}")

        finally:
            for f in image_files:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass