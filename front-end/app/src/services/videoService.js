import { API_BASE_URL } from '../config/api';

class VideoService {
  async getVideoData(videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/files/video-data/${videoId}`);
      if (!response.ok) throw new Error(`Erro na requisição: ${response.status}`);
      const data = await response.json();
      if (data.status !== 'success' || !data.analysis) {
         throw new Error("Payload da API inválido ou análise falhou.");
      }
      return data;
    } catch (error) {
      console.error("VideoService [getVideoData] Error:", error);
      throw error;
    }
  }

  // NOVA ROTA: Upload de Vídeo
  async uploadVideo(file) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      // Substitua '/upload' pela rota real do seu backend
      const response = await fetch(`${API_BASE_URL}/files/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error("Erro ao fazer upload do vídeo");
      
      // Espera-se que o backend retorne { video_id: "...", name: "..." }
      return await response.json();
    } catch (error) {
      console.error("VideoService [uploadVideo] Error:", error);
      throw error;
    }
  }

  // NOVA ROTA: Listar Projetos
  async getProjects() {
    try {
      // Substitua '/projects' pela rota real do seu backend
      const response = await fetch(`${API_BASE_URL}/files/videos`);
      if (!response.ok) throw new Error("Erro ao buscar projetos");
      
      // Espera-se que o backend retorne [{ id: "...", name: "..." }, ...]
      return await response.json();
    } catch (error) {
      console.error("VideoService [getProjects] Error:", error);
      throw error;
    }
  }

  // NOVA ROTA: Gerar Relatório
  async generateReport(videoId) {
    try {
      // Substitua '/report' pela rota real do seu backend
      const response = await fetch(`${API_BASE_URL}/relatorios/${videoId}`, { method: 'POST' });
      if (!response.ok) throw new Error("Erro ao gerar relatório");
      
      return await response.json();
    } catch (error) {
      console.error("VideoService [generateReport] Error:", error);
      throw error;
    }
  }
}

export const videoService = new VideoService();