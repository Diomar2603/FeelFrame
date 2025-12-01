import uuid
import os
import pandas as pd
import plotly.express as px
from fpdf import FPDF
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from enum import Enum
from collections import Counter
from pymongo.collection import Collection
from app.utils.DatabaseConfig import DatabaseConfig
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
        self.fluxo_frames_collection: Collection = self.db["fluxo_frames"]

    def gerar_dados_grafico_temporal(self, analyses: List[FrameAnalysis]) -> Dict[str, List[Any]]:
        """
        Extrai dados de série temporal, usando frame_number para o eixo X.
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
            dados["frame_number"].append(frame.frame_number)
            dados["timestamp_ms"].append(frame.timestamp_ms)
            dados["engajamento"].append(frame.estimativa_engajamento.value)
            dados["comportamento"].append(frame.dimensao_comportamental.value)
            dados["emocao"].append(frame.emocao)
            dados["estado_fluxo"].append(getattr(frame, 'estado_fluxo', False))
        return dados

    def calcular_distribuicao_percentual(self, analyses: List[FrameAnalysis], total_frames: int) -> Dict[str, Dict[str, float]]:
        """Calcula a distribuição percentual de cada categoria."""
        
        if total_frames == 0:
             return {"engajamento": {}, "comportamento": {}, "emocao": {}, "fluxo": {}}

        eng_counter = Counter([f.estimativa_engajamento.value for f in analyses])
        com_counter = Counter([f.dimensao_comportamental.value for f in analyses])
        emo_counter = Counter([f.emocao for f in analyses])
        fluxo_counter = Counter([getattr(f, 'estado_fluxo', False) for f in analyses])

        def _counts_to_percent(counter: Counter, total: int) -> Dict[str, float]:
            if total == 0: return {}
            return {
                key: round((count / total) * 100.0, 2) 
                for key, count in counter.items()
            }

        return {
            "engajamento": _counts_to_percent(eng_counter, total_frames),
            "comportamento": _counts_to_percent(com_counter, total_frames),
            "emocao": _counts_to_percent(emo_counter, total_frames),
            "fluxo": _counts_to_percent(fluxo_counter, total_frames)
        }
    
    def _merge_periods(self, seconds_list: List[int]) -> List[Tuple[int, int]]:
        """Agrupa segundos consecutivos em períodos (início, fim)."""
        if not seconds_list:
            return []
        
        sorted_seconds = sorted(list(set(seconds_list)))
        periods = []
        start_period = sorted_seconds[0]
        end_period = sorted_seconds[0]

        for i in range(1, len(sorted_seconds)):
            if sorted_seconds[i] == end_period + 1:
                end_period = sorted_seconds[i]
            else:
                periods.append((start_period, end_period + 1))
                start_period = sorted_seconds[i]
                end_period = sorted_seconds[i]
        
        periods.append((start_period, end_period + 1))
        return periods

    def analisar_concentracao(self, analyses: List[FrameAnalysis]) -> Dict[str, List[Tuple[int, int]]]:
        """
        Analisa o vídeo em janelas de 1 segundo para encontrar picos
        de engajamento e desengajamento.
        """
        if not analyses:
            return {"engajamento": [], "alt_desengajamento": [], "fluxo": []}

        frames_por_segundo: Dict[int, List[FrameAnalysis]] = {}
        for f in analyses:
            segundo = f.timestamp_ms // 1000
            if segundo not in frames_por_segundo:
                frames_por_segundo[segundo] = []
            frames_por_segundo[segundo].append(f)
            
        segundos_eng_alto = []
        segundos_alt_des_alto = []
        segundos_fluxo = []

        for segundo, frames_no_segundo in frames_por_segundo.items():
            total_frames = len(frames_no_segundo)
            if total_frames == 0:
                continue

            count_eng = len([
                f for f in frames_no_segundo 
                if f.estimativa_engajamento in (
                    EstimativaEngajamentoEnum.ENGAJADO, 
                    EstimativaEngajamentoEnum.ALTAMENTE_ENGAJADO
                )
            ])
            
            count_alt_des = len([
                f for f in frames_no_segundo 
                if f.estimativa_engajamento == EstimativaEngajamentoEnum.ALTAMENTE_DESENGAJADO
            ])

            count_fluxo = len([
                f for f in frames_no_segundo 
                if getattr(f, 'estado_fluxo', False)
            ])

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

    def buscar_frames_fluxo(self, video_id: str) -> List[Dict[str, Any]]:
        """Busca frames de fluxo salvos para o vídeo."""
        try:
            query = {"video_id": video_id}
            return list(self.fluxo_frames_collection.find(query))
        except Exception as e:
            print(f"Erro ao buscar frames de fluxo: {e}")
            return []

    def buscar_frames_do_banco_de_dados(self, video_id: str) -> List[FrameAnalysis]:
        """
        Busca todos os resultados de FrameAnalysis de um video_id
        específico no MongoDB.
        """
        print(f"Buscando dados para video_id: {video_id}...")
        
        lista_de_frames = []
        
        try:
            query = {"video_id": video_id}
            resultados_db = self.frame_analysis_collection.find(query)
            
            for doc in resultados_db:
                try:
                    pose_data = doc.get('pose_cabeca')
                    olhar_data = doc.get('olhar')
                    
                    if not pose_data or not olhar_data:
                        print(f"  [AVISO] Frame {doc.get('frame_number')} pulado: pose_cabeca ou olhar ausente.")
                        continue
                    
                    # CORREÇÃO: Cria HeadPose com todos os campos necessários
                    pose_obj = HeadPose(
                        direcao_horizontal=pose_data.get('direcao_horizontal', 'Indefinido'),
                        raw_yaw=pose_data.get('raw_yaw', 0.0),
                        direcao_vertical=pose_data.get('direcao_vertical', 'Indefinido'),
                        raw_pitch=pose_data.get('raw_pitch', 0.0),
                        proximidade_z=pose_data.get('proximidade_z', 0.0),
                        raw_roll=pose_data.get('raw_roll', 0.0)
                    )
                    
                    olhar_obj = GazeDirection(**olhar_data)
                    
                    # Cria FrameAnalysis com os objetos corrigidos
                    frame_analysis = FrameAnalysis(
                        video_id=doc['video_id'],
                        timestamp_ms=doc['timestamp_ms'],
                        frame_number=doc['frame_number'],
                        emocao=doc['emocao'],
                        pose_cabeca=pose_obj,
                        olhar=olhar_obj,
                        dimensao_comportamental=DimensaoComportamentalEnum(doc['dimensao_comportamental']),
                        estimativa_engajamento=EstimativaEngajamentoEnum(doc['estimativa_engajamento']),
                        emotion_confidence=doc.get('emotion_confidence', 0.0),
                        estado_fluxo=doc.get('estado_fluxo', False)
                    )
                    
                    lista_de_frames.append(frame_analysis)
                    
                except Exception as e:
                    print(f"  [AVISO] Frame {doc.get('frame_number')} pulado: erro ao processar. Detalhe: {e}")

            print(f"Encontrados e processados {len(lista_de_frames)} frames reais no banco.")
            return lista_de_frames
        
        except Exception as e:
            print(f"ERRO FATAL: Não foi possível conectar ou buscar dados no MongoDB: {e}")
            return []

    def _get_color_map(self) -> Dict[str, str]:
        """
        Define um mapa de cores específico para cada categoria.
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
            
            # Estado de Fluxo
            "True": "#00C853",
            "False": "#9E9E9E",
            
            # Valores string diretos (fallback)
            "Feliz": "#FFEB3B",
            "Surpreso": "#00E5FF", 
            "Medo": "#7B1FA2",
            "Triste": "#2196F3",
            "Neutro": "#9E9E9E",
            "Indefinido": "#607D8B"
        }
    
    def gerar_relatorio_pdf(self, output_filename: str, analisys: List[FrameAnalysis], video_id: str):
        """
        Gera relatório PDF completo com seção de cenas de fluxo.
        """
        print(f"Iniciando a geração do relatório: {output_filename}...")

        # --- 1. Obter dados ---
        if not analisys:
            print("A lista de análise está vazia. Abortando geração de PDF.")
            return
            
        total_frames = len(analisys)
        dados_temporais = self.gerar_dados_grafico_temporal(analisys)
        dados_distribuicao = self.calcular_distribuicao_percentual(analisys, total_frames)
        momentos_destaque = self.analisar_concentracao(analisys)
        
        # Buscar frames de fluxo
        frames_fluxo = self.buscar_frames_fluxo(video_id)
        
        if not dados_temporais["frame_number"]:
            print("Dados temporais estão vazios. Abortando geração de gráficos.")
            return

        # --- 2. Criar DataFrames ---
        print("Preparando DataFrames para os gráficos...")
        df_temporal = pd.DataFrame(dados_temporais)
        
        df_eng_pie = pd.DataFrame(dados_distribuicao['engajamento'].items(), columns=['Categoria', 'Percentual'])
        df_comp_pie = pd.DataFrame(dados_distribuicao['comportamento'].items(), columns=['Categoria', 'Percentual'])
        df_emo_pie = pd.DataFrame(dados_distribuicao['emocao'].items(), columns=['Categoria', 'Percentual'])
        df_fluxo_pie = pd.DataFrame(dados_distribuicao['fluxo'].items(), columns=['Categoria', 'Percentual'])

        # --- 3. Obter mapa de cores ---
        MAPA_CORES = self._get_color_map()

        # --- 4. Criar os Gráficos ---
        print("Criando figuras do Plotly...")

        # Gráficos temporais (spike plots)
        fig_eng_temporal = px.scatter(
            df_temporal,
            x="frame_number",
            y="engajamento",
            color="engajamento",
            color_discrete_map=MAPA_CORES,
            title="Distribuição de Engajamento ao Longo do Vídeo",
            hover_data=["frame_number", "timestamp_ms"]
        )
        fig_eng_temporal.update_traces(
            marker_symbol='line-ns', 
            marker_size=12, 
            marker_line_width=2,
            marker_opacity=0.8
        )
        fig_eng_temporal.update_layout(
            xaxis_title="Quantidade de Frames",
            yaxis_title="Nível Engajamento",
            legend_title="Engajamento"
        )

        fig_comp_temporal = px.scatter(
            df_temporal,
            x="frame_number",
            y="comportamento",
            color="comportamento",
            color_discrete_map=MAPA_CORES,
            title="Distribuição de Comportamento ao Longo do Vídeo",
            hover_data=["frame_number", "timestamp_ms"]
        )
        fig_comp_temporal.update_traces(
            marker_symbol='line-ns', 
            marker_size=12, 
            marker_line_width=2,
            marker_opacity=0.8
        )
        fig_comp_temporal.update_layout(
            xaxis_title="Quantidade de Frames",
            yaxis_title="Nível Comportamento",
            legend_title="Comportamento"
        )

        fig_emo_temporal = px.scatter(
            df_temporal,
            x="frame_number",
            y="emocao",
            color="emocao",
            color_discrete_map=MAPA_CORES,
            title="Distribuição de Emoção ao Longo do Vídeo",
            hover_data=["frame_number", "timestamp_ms"]
        )
        fig_emo_temporal.update_traces(
            marker_symbol='line-ns', 
            marker_size=12, 
            marker_line_width=2,
            marker_opacity=0.8
        )
        fig_emo_temporal.update_layout(
            xaxis_title="Quantidade de Frames",
            yaxis_title="Nível Emoção",
            legend_title="Emoção"
        )

        # Gráfico de estado de fluxo
        fig_fluxo_temporal = px.scatter(
            df_temporal,
            x="frame_number",
            y="estado_fluxo",
            color="estado_fluxo",
            color_discrete_map=MAPA_CORES,
            title="Estado de Fluxo ao Longo do Vídeo",
            hover_data=["frame_number", "timestamp_ms"]
        )
        fig_fluxo_temporal.update_traces(
            marker_symbol='line-ns', 
            marker_size=12, 
            marker_line_width=2,
            marker_opacity=0.8
        )
        fig_fluxo_temporal.update_layout(
            xaxis_title="Quantidade de Frames",
            yaxis_title="Estado de Fluxo",
            legend_title="Fluxo"
        )
        
        # Gráficos de Pizza
        fig_pie_eng = px.pie(
            df_eng_pie, 
            names='Categoria', 
            values='Percentual', 
            title='Distribuição Percentual de Engajamento', 
            color='Categoria', 
            color_discrete_map=MAPA_CORES
        )
        
        fig_pie_comp = px.pie(
            df_comp_pie, 
            names='Categoria', 
            values='Percentual', 
            title='Distribuição Percentual de Comportamento', 
            color='Categoria', 
            color_discrete_map=MAPA_CORES
        )
        
        fig_pie_emo = px.pie(
            df_emo_pie, 
            names='Categoria', 
            values='Percentual', 
            title='Distribuição Percentual de Emoção', 
            color='Categoria', 
            color_discrete_map=MAPA_CORES
        )

        fig_pie_fluxo = px.pie(
            df_fluxo_pie, 
            names='Categoria', 
            values='Percentual', 
            title='Distribuição Percentual de Estado de Fluxo', 
            color='Categoria', 
            color_discrete_map=MAPA_CORES
        )

        # Lista de gráficos para exportar
        charts_to_export = [
            (fig_eng_temporal, "temp_grafico_1_spike_eng.png"),
            (fig_comp_temporal, "temp_grafico_2_spike_comp.png"),
            (fig_emo_temporal, "temp_grafico_3_spike_emo.png"),
            (fig_fluxo_temporal, "temp_grafico_4_spike_fluxo.png"),
            (fig_pie_eng, "temp_grafico_5_pizza_eng.png"),
            (fig_pie_comp, "temp_grafico_6_pizza_comp.png"),
            (fig_pie_emo, "temp_grafico_7_pizza_emo.png"),
            (fig_pie_fluxo, "temp_grafico_8_pizza_fluxo.png"),
        ]
        
        image_files = [] 
        
        # --- 5. Salvar imagens e compilar PDF ---
        try:
            print("Exportando gráficos para arquivos PNG...")
            for fig, filename in charts_to_export:
                fig.write_image(filename, width=1000, height=450)
                image_files.append(filename)

            pdf = FPDF(orientation="L") 
            pdf.set_auto_page_break(auto=True, margin=15)

            # Página de Título
            pdf.add_page()
            pdf.set_font("Arial", "B", 24)
            pdf.cell(0, 20, "Relatório de Análise de Engajamento", 0, 1, "C")
            pdf.set_font("Arial", "", 16)
            pdf.cell(0, 10, f"ID do Vídeo: {video_id}", 0, 1, "C")
            pdf.cell(0, 10, f"Total de Frames Analisados: {total_frames}", 0, 1, "C")
            pdf.cell(0, 10, f"Cenas de Fluxo Detectadas: {len(frames_fluxo)}", 0, 1, "C")
            pdf.ln(20)

            # Página de Momentos de Destaque
            pdf.add_page()
            pdf.set_font("Arial", "B", 18)
            pdf.cell(0, 10, "Momentos de Destaque", 0, 1, "L")
            pdf.ln(5)

            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Períodos de Alta Concentração (>75% Engajado ou Alt. Engajado):", 0, 1, "L")
            pdf.set_font("Arial", "", 12)
            if not momentos_destaque["engajamento"]:
                pdf.cell(0, 8, "  - Nenhum período encontrado.", 0, 1, "L")
            else:
                for p in momentos_destaque["engajamento"]:
                    pdf.cell(0, 8, f"  - De {p[0]}s até {p[1]}s", 0, 1, "L")
            
            pdf.ln(5)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Períodos de Alto Desengajamento (>75% Altamente Desengajado):", 0, 1, "L")
            pdf.set_font("Arial", "", 12)
            if not momentos_destaque["alt_desengajamento"]:
                pdf.cell(0, 8, "  - Nenhum período encontrado.", 0, 1, "L")
            else:
                for p in momentos_destaque["alt_desengajamento"]:
                    pdf.cell(0, 8, f"  - De {p[0]}s até {p[1]}s", 0, 1, "L")

            pdf.ln(5)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Períodos de Estado de Fluxo (>50% em Fluxo):", 0, 1, "L")
            pdf.set_font("Arial", "", 12)
            if not momentos_destaque["fluxo"]:
                pdf.cell(0, 8, "  - Nenhum período encontrado.", 0, 1, "L")
            else:
                for p in momentos_destaque["fluxo"]:
                    pdf.cell(0, 8, f"  - De {p[0]}s até {p[1]}s", 0, 1, "L")

            # Página de Cenas de Fluxo
            if frames_fluxo:
                pdf.add_page()
                pdf.set_font("Arial", "B", 18)
                pdf.cell(0, 10, "Cenas de Fluxo Capturadas", 0, 1, "L")
                pdf.ln(5)
                
                pdf.set_font("Arial", "", 12)
                for i, frame_info in enumerate(frames_fluxo[:10]):
                    pdf.cell(0, 8, f"Frame {frame_info['frame_number']} - {frame_info['timestamp_ms']}ms", 0, 1, "L")
                    pdf.cell(0, 8, f"  Emoção: {frame_info['emocao']}, Engajamento: {frame_info['engajamento']}", 0, 1, "L")
                    pdf.cell(0, 8, f"  Comportamento: {frame_info['dimensao']}, Confiança: {frame_info['confidence']:.2f}", 0, 1, "L")
                    pdf.ln(2)
                
                if len(frames_fluxo) > 10:
                    pdf.cell(0, 8, f"... e mais {len(frames_fluxo) - 10} cenas de fluxo.", 0, 1, "L")

            # Páginas dos Gráficos
            print("Compilando o PDF...")
            for i, img_file in enumerate(image_files):
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                title = charts_to_export[i][1].replace("temp_grafico_", "").replace(".png", "").replace("_", " ").title()
                pdf.cell(0, 10, f"Gráfico: {title}", 0, 1, "L")
                pdf.ln(10)
                
                pdf_width = pdf.w - pdf.l_margin - pdf.r_margin
                pdf.image(img_file, x=pdf.l_margin, w=pdf_width) 

            pdf.output(output_filename)
            print(f"Relatório salvo com sucesso em: {output_filename}")

        except Exception as e:
            print(f"ERRO ao gerar o PDF: {e}")
            
        finally:
            print("Limpando arquivos de imagem temporários...")
            for img_file in image_files:
                if os.path.exists(img_file):
                    os.remove(img_file)