import { API_BASE_URL } from '../config/api';

function authHeader() {
  const token = localStorage.getItem('feelframe_token');
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function _get(path) {
  const res = await fetch(`${API_BASE_URL}${path}`, { headers: authHeader() });
  if (!res.ok) throw new Error(`Erro na requisição: ${res.status}`);
  return res.json();
}

class VideoService {
  async getVideoData(videoId) {
    try {
      const data = await _get(`/files/video-data/${videoId}`);
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

      const res = await fetch(`${API_BASE_URL}/files/upload/`, {
        method: 'POST',
        headers: authHeader(),
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
      return await _get('/files/videos/');
    } catch (error) {
      console.error('VideoService [getProjects] Error:', error);
      throw error;
    }
  }

  async deleteProject(videoId) {
    try {
      const res = await fetch(`${API_BASE_URL}/files/videos/${videoId}`, {
        method: 'DELETE',
        headers: authHeader(),
      });
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
      return await _get(`/files/videos/${videoId}/markers`);
    } catch (error) {
      console.error('VideoService [getMarkers] Error:', error);
      throw error;
    }
  }

  async addMarker(videoId, time, label = '') {
    try {
      const res = await fetch(`${API_BASE_URL}/files/videos/${videoId}/markers`, {
        method: 'POST',
        headers: { ...authHeader(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ time, label }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Erro ao adicionar marcador');
      }
      return res.json();
    } catch (error) {
      console.error('VideoService [addMarker] Error:', error);
      throw error;
    }
  }

  async updateMarker(markerId, updates) {
    try {
      const res = await fetch(`${API_BASE_URL}/files/markers/${markerId}`, {
        method: 'PATCH',
        headers: { ...authHeader(), 'Content-Type': 'application/json' },
        body: JSON.stringify(updates),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Erro ao atualizar marcador');
      }
      return res.json();
    } catch (error) {
      console.error('VideoService [updateMarker] Error:', error);
      throw error;
    }
  }

  async bulkReplaceEmotions(videoId, startTime, endTime, newEmotion) {
    try {
      const res = await fetch(`${API_BASE_URL}/files/videos/${videoId}/bulk-replace-emotions`, {
        method: 'PATCH',
        headers: { ...authHeader(), 'Content-Type': 'application/json' },
        body: JSON.stringify({ start_time: startTime, end_time: endTime, new_emotion: newEmotion }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Erro na substituição em lote');
      }
      return res.json();
    } catch (error) {
      console.error('VideoService [bulkReplaceEmotions] Error:', error);
      throw error;
    }
  }

  async generateReport(videoId) {
    try {
      const res = await fetch(`${API_BASE_URL}/relatorios/${videoId}`, {
        method: 'POST',
        headers: authHeader(),
      });
      if (!res.ok) throw new Error('Erro ao gerar relatório');
      return res.json();
    } catch (error) {
      console.error('VideoService [generateReport] Error:', error);
      throw error;
    }
  }
}

export const videoService = new VideoService();
