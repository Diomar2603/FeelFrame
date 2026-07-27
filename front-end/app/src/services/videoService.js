import { apiFetch } from './apiClient';

async function _get(path) {
  const res = await apiFetch(path);
  if (!res.ok) throw new Error(`Erro na requisição: ${res.status}`);
  return res.json();
}

async function _json(path, method, body) {
  const res = await apiFetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || 'Erro na requisição');
  }
  return res.json();
}

class VideoService {
  async getVideoData(videoId) {
    try {
      const data = await _get(`/arquivos/dados/${videoId}`);
      if (data.status !== 'success' || !data.analysis) {
        throw new Error('Payload da API inválido ou análise ainda em andamento.');
      }
      return data;
    } catch (error) {
      console.error('VideoService [getVideoData] Error:', error);
      throw error;
    }
  }

  async uploadVideo(file) {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await apiFetch('/arquivos/enviar/', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Erro ao fazer upload do vídeo');
      }
      return res.json();
    } catch (error) {
      console.error('VideoService [uploadVideo] Error:', error);
      throw error;
    }
  }

  async getProjects() {
    try {
      return await _get('/arquivos/videos/');
    } catch (error) {
      console.error('VideoService [getProjects] Error:', error);
      throw error;
    }
  }

  async deleteProject(videoId) {
    try {
      const res = await apiFetch(`/arquivos/videos/${videoId}`, { method: 'DELETE' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Erro ao excluir projeto');
      }
      return res.json();
    } catch (error) {
      console.error('VideoService [deleteProject] Error:', error);
      throw error;
    }
  }

  async getMarkers(videoId) {
    try {
      return await _get(`/arquivos/videos/${videoId}/marcadores`);
    } catch (error) {
      console.error('VideoService [getMarkers] Error:', error);
      throw error;
    }
  }

  async addMarker(videoId, time, label = '') {
    try {
      return await _json(`/arquivos/videos/${videoId}/marcadores`, 'POST', { time, label });
    } catch (error) {
      console.error('VideoService [addMarker] Error:', error);
      throw error;
    }
  }

  async updateMarker(markerId, updates) {
    try {
      return await _json(`/arquivos/marcadores/${markerId}`, 'PATCH', updates);
    } catch (error) {
      console.error('VideoService [updateMarker] Error:', error);
      throw error;
    }
  }

  async bulkReplaceEmotions(videoId, startTime, endTime, newEmotion) {
    try {
      return await _json(`/arquivos/videos/${videoId}/substituir-emocoes`, 'PATCH', {
        start_time: startTime,
        end_time: endTime,
        new_emotion: newEmotion,
      });
    } catch (error) {
      console.error('VideoService [bulkReplaceEmotions] Error:', error);
      throw error;
    }
  }

  async generateReport(videoId) {
    try {
      const res = await apiFetch(`/relatorios/${videoId}`, { method: 'GET' });
      if (!res.ok) throw new Error('Erro ao gerar relatório');

      const blob = await res.blob();
      const disposition = res.headers.get('Content-Disposition') || '';
      const match = disposition.match(/filename="?([^"]+)"?/);
      const filename = match ? match[1] : `Relatorio_${videoId}.pdf`;

      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('VideoService [generateReport] Error:', error);
      throw error;
    }
  }
}

export const videoService = new VideoService();
