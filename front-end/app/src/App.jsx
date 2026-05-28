import React, { useRef, useState, useEffect, useCallback, memo } from 'react';
import { Routes, Route, useNavigate } from 'react-router-dom';
import './Editor.css';
import { EMOTIONS_CONFIG } from './constants/emotions';
import { videoService } from './services/videoService';
import { useAuth } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import LoginPage from './components/auth/LoginPage';

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// ---------------------------------------------------------------------------
// Utilitários de nível de módulo (estáveis, sem re-criação)
// ---------------------------------------------------------------------------
function formatTime(t) {
  return new Date(t * 1000).toISOString().substr(11, 8);
}

// ---------------------------------------------------------------------------
// MarkerPin — componente memoizado para evitar re-renders a 60fps
// ---------------------------------------------------------------------------
const MarkerPin = memo(function MarkerPin({ marker, duration, onColorChange }) {
  const leftPct = duration > 0 ? (marker.time / duration) * 100 : 0;
  const color   = marker.color || '#f0a500';
  return (
    <div
      className="marker-pin"
      style={{ left: `${leftPct}%`, '--mc': color }}
      onMouseDown={e => e.stopPropagation()} // não aciona timeline seek
    >
      <div className="marker-pin-arrow" style={{ borderTopColor: color }} />
      <div className="marker-tooltip" onClick={e => e.stopPropagation()}>
        <div className="marker-tooltip-info">
          <span className="marker-tooltip-label">{marker.label || '(sem rótulo)'}</span>
          <span className="marker-tooltip-time">{formatTime(marker.time)}</span>
        </div>
        <div className="marker-tooltip-color-row">
          <span className="marker-tooltip-color-label">Cor</span>
          <input
            type="color"
            value={color}
            onChange={e => onColorChange(marker.marker_id, e.target.value)}
            className="marker-color-input"
            title="Alterar cor do marcador"
          />
        </div>
      </div>
    </div>
  );
});

// ---------------------------------------------------------------------------
// VideoEditor
// ---------------------------------------------------------------------------
function VideoEditor() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();

  const mainVideoRef       = useRef(null);
  const faceVideoRef       = useRef(null);
  const requestRef         = useRef();
  const sseRef             = useRef(null);
  const timelineTracksRef  = useRef(null);
  const durationRef        = useRef(1);       // espelho de `duration` para closures estáveis
  const colorChangeTimerRef = useRef({});      // debounce timers por marker_id

  const [videoData, setVideoData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const [projects, setProjects] = useState([]);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);
  const [toastType, setToastType] = useState('info'); // 'info' | 'success' | 'error'

  const [markers, setMarkers] = useState([]);
  const [markerLabel, setMarkerLabel] = useState('');

  const [dragSel, setDragSel]       = useState(null); // { startPct, endPct, startTime, endTime }
  const [bulkEmotion, setBulkEmotion] = useState('');

  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(1);
  const [isSeeking, setIsSeeking] = useState(false);

  const handleLogout = () => { logout(); navigate('/login', { replace: true }); };

  const showToast = useCallback((msg, type = 'info') => {
    setToastMessage(msg);
    setToastType(type);
  }, []);
  const closeToast = () => setToastMessage(null);

  // Mantém durationRef sincronizado para closures estáveis (drag, seek)
  useEffect(() => { durationRef.current = duration; }, [duration]);

  const refreshProjects = useCallback(async () => {
    try {
      const data = await videoService.getProjects();
      setProjects(data.videos || []);
    } catch {
      // silencioso
    }
  }, []);

  // =========================================
  // UPLOAD COM SSE DE PROGRESSO
  // =========================================
  const handleImportVideo = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Fecha SSE anterior se existir
    if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }

    try {
      setError(null);
      showToast(`Enviando "${file.name}"...`);

      const response = await videoService.uploadVideo(file);
      const videoId = response.video_id;

      showToast(`"${file.name}" em processamento (0%)...`);

      const source = new EventSource(`${API_BASE}/files/processing-progress/${videoId}`);
      sseRef.current = source;

      source.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.status === 'success') {
            showToast(`"${file.name}" processado com sucesso!`, 'success');
            source.close();
            sseRef.current = null;
            refreshProjects();
          } else if (data.status === 'failed') {
            setToastMessage(null);
            setError(`Falha no processamento de "${file.name}".`);
            source.close();
            sseRef.current = null;
          } else {
            const pct = data.progress_percent || 0;
            showToast(`"${file.name}" em processamento (${pct}%)...`);
          }
        } catch {
          source.close();
          sseRef.current = null;
        }
      };

      source.onerror = () => {
        source.close();
        sseRef.current = null;
      };

    } catch {
      setToastMessage(null);
      setError('Falha ao enviar vídeo. Verifique sua conexão ou backend.');
    }
    e.target.value = null;
  };

  // Cleanup SSE ao desmontar
  useEffect(() => () => { if (sseRef.current) sseRef.current.close(); }, []);

  // =========================================
  // DROPDOWN DE PROJETOS
  // =========================================
  const toggleProjectsDropdown = async () => {
    const nextState = !isDropdownOpen;
    setIsDropdownOpen(nextState);
    if (nextState) await refreshProjects();
  };

  const loadProject = async (projectId) => {
    setIsDropdownOpen(false);
    try {
      setIsLoading(true);
      setError(null);
      const [responseData, markersData] = await Promise.all([
        videoService.getVideoData(projectId),
        videoService.getMarkers(projectId),
      ]);
      setVideoData(responseData);
      setMarkers(markersData.markers || []);
      setCurrentTime(0);
      setIsPlaying(false);
    } catch {
      setError('Falha ao carregar os dados do projeto.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDeleteProject = async (projectId, projectName) => {
    if (!window.confirm(`Excluir "${projectName}"?\n\nEsta ação é irreversível e removerá todos os dados relacionados.`)) return;
    setIsDropdownOpen(false);
    try {
      await videoService.deleteProject(projectId);
      if (videoData?.video_id === projectId) {
        setVideoData(null);
        setMarkers([]);
        setCurrentTime(0);
        setIsPlaying(false);
      }
      await refreshProjects();
      showToast(`"${projectName}" excluído com sucesso.`, 'success');
    } catch {
      showToast('Falha ao excluir o projeto.', 'error');
    }
  };

  const handleAddMarker = async () => {
    if (!videoData) return;
    try {
      const result = await videoService.addMarker(videoData.video_id, currentTime, markerLabel.trim());
      setMarkers(prev => [...prev, result].sort((a, b) => a.time - b.time));
      setMarkerLabel('');
      showToast('Marcador adicionado.', 'success');
    } catch {
      showToast('Falha ao adicionar marcador.', 'error');
    }
  };

  // =========================================
  // ATUALIZAÇÃO DE COR DO MARCADOR (debounce 500ms)
  // =========================================
  const handleMarkerColorChange = useCallback((markerId, color) => {
    // Atualização visual imediata
    setMarkers(prev => prev.map(m => m.marker_id === markerId ? { ...m, color } : m));
    // Salva no backend após pausa de digitação
    clearTimeout(colorChangeTimerRef.current[markerId]);
    colorChangeTimerRef.current[markerId] = setTimeout(() => {
      videoService.updateMarker(markerId, { color })
        .catch(() => setToastMessage('Falha ao salvar cor do marcador.'));
    }, 500);
  }, []); // todas as deps são refs/setters estáveis

  // =========================================
  // CLICK-TO-SEEK + DRAG-TO-SELECT NA TIMELINE
  // =========================================
  const handleTimelineMouseDown = useCallback((e) => {
    if (e.button !== 0) return;
    const startX = e.clientX;
    const rect   = timelineTracksRef.current?.getBoundingClientRect();
    if (!rect) return;
    const w = rect.width;
    const L = rect.left;

    const calcSel = (clientX) => {
      const sRaw = Math.max(0, Math.min(startX, clientX) - L);
      const eRaw = Math.max(0, Math.min(Math.max(startX, clientX) - L, w));
      return {
        startPct:  (sRaw / w) * 100,
        endPct:    (eRaw / w) * 100,
        startTime: (sRaw / w) * durationRef.current,
        endTime:   (eRaw / w) * durationRef.current,
      };
    };

    const onMove = (mv) => {
      if (Math.abs(mv.clientX - startX) >= 5) setDragSel(calcSel(mv.clientX));
    };

    const onUp = (up) => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      const dx = up.clientX - startX;
      if (Math.abs(dx) < 5) {
        // Click simples → seek
        const x    = Math.max(0, Math.min(up.clientX - L, w));
        const time = Math.max(0, Math.min((x / w) * durationRef.current, durationRef.current));
        if (mainVideoRef.current) mainVideoRef.current.currentTime = time;
        if (faceVideoRef.current) faceVideoRef.current.currentTime = time;
        setCurrentTime(time);
        setDragSel(null);
      }
      // Drag: mantém dragSel para exibir barra de bulk replace
    };

    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  }, []); // stable: usa apenas refs e setters estáveis do React

  // ESC limpa seleção
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') { setDragSel(null); setBulkEmotion(''); } };
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, []);

  // =========================================
  // SUBSTITUIÇÃO EM LOTE
  // =========================================
  const handleBulkReplace = async () => {
    if (!dragSel || !bulkEmotion || !videoData) return;
    try {
      const result = await videoService.bulkReplaceEmotions(
        videoData.video_id, dragSel.startTime, dragSel.endTime, bulkEmotion
      );
      setVideoData(prev => ({
        ...prev,
        analysis: { ...prev.analysis, emocao: result.emocao },
      }));
      setDragSel(null);
      setBulkEmotion('');
      showToast(`${result.updated_count} frame(s) → "${bulkEmotion}".`, 'success');
    } catch {
      showToast('Falha na substituição em lote.', 'error');
    }
  };

  const handleReport = async () => {
    if (!videoData) return;
    try {
      showToast('Gerando relatório em background...');
      await videoService.generateReport(videoData.video_id);
    } catch {
      setToastMessage(null);
      setError('Falha ao solicitar o relatório.');
    }
  };

  // =========================================
  // MOTOR DE 60FPS E CONTROLES
  // =========================================
  const updateProgress = () => {
    if (!isSeeking && mainVideoRef.current) {
      setCurrentTime(mainVideoRef.current.currentTime);
      if (faceVideoRef.current &&
          Math.abs(mainVideoRef.current.currentTime - faceVideoRef.current.currentTime) > 0.3) {
        faceVideoRef.current.currentTime = mainVideoRef.current.currentTime;
      }
    }
    if (mainVideoRef.current && !mainVideoRef.current.paused) {
      requestRef.current = requestAnimationFrame(updateProgress);
    }
  };

  useEffect(() => {
    if (isPlaying) requestRef.current = requestAnimationFrame(updateProgress);
    else cancelAnimationFrame(requestRef.current);
    return () => cancelAnimationFrame(requestRef.current);
  }, [isPlaying, isSeeking]);

  const togglePlay = () => {
    if (mainVideoRef.current.paused) {
      mainVideoRef.current.play();
      faceVideoRef.current.play();
      setIsPlaying(true);
    } else {
      mainVideoRef.current.pause();
      faceVideoRef.current.pause();
      setIsPlaying(false);
    }
  };

  const handleSeekMouseDown = () => {
    setIsSeeking(true);
    if (isPlaying) { mainVideoRef.current.pause(); faceVideoRef.current.pause(); }
  };

  const handleSeekChange = (e) => {
    const newTime = (parseFloat(e.target.value) / 100) * duration;
    setCurrentTime(newTime);
    mainVideoRef.current.currentTime = newTime;
    faceVideoRef.current.currentTime = newTime;
  };

  const handleSeekMouseUp = () => {
    setIsSeeking(false);
    if (isPlaying) {
      mainVideoRef.current.play();
      faceVideoRef.current.play();
      requestRef.current = requestAnimationFrame(updateProgress);
    }
  };

  const handleLoadedMetadata = () => {
    if (mainVideoRef.current.duration) setDuration(mainVideoRef.current.duration);
  };

  // =========================================
  // RENDERIZAÇÃO
  // =========================================
  const progressPercent = videoData ? (currentTime / duration) * 100 : 0;
  const emocoesArray = videoData?.analysis?.emocao || [];
  const activeEmotionData = emocoesArray.find(d => currentTime >= d.start && currentTime < d.end) || emocoesArray[0];
  const currentEmotionLabel = activeEmotionData ? activeEmotionData.emocao : 'Indefinido';

  return (
    <div className="editor-container">

      {toastMessage && (
        <div className={`toast-notification ${toastType === 'success' ? 'toast-success' : toastType === 'error' ? 'toast-error' : ''}`}>
          <span>{toastMessage}</span>
          <button onClick={closeToast} className="toast-close-btn">✖</button>
        </div>
      )}

      <header className="top-bar">
        <div className="header-left">
          <label className="btn-primary" style={{ cursor: 'pointer' }}>
            Import
            <input
              type="file"
              accept="video/mp4,video/webm,video/avi,video/mov"
              onChange={handleImportVideo}
              style={{ display: 'none' }}
            />
          </label>
          <button className="btn-secondary" onClick={handleReport} disabled={!videoData}>
            Relatório
          </button>
        </div>

        <div className="header-center">
          <div className="dropdown-container">
            <button className="btn-menu" onClick={toggleProjectsDropdown}>
              Projetos ▼
            </button>
            {isDropdownOpen && (
              <ul className="dropdown-menu">
                {projects.length === 0 ? (
                  <li className="dropdown-empty">Nenhum projeto encontrado</li>
                ) : (
                  projects.map(proj => (
                    <li key={proj.video_id} className="dropdown-item">
                      <span
                        className="dropdown-item-name"
                        onClick={() => loadProject(proj.video_id)}
                      >
                        {proj.original_filename}
                      </span>
                      <button
                        className="dropdown-item-delete"
                        onClick={(e) => { e.stopPropagation(); handleDeleteProject(proj.video_id, proj.original_filename); }}
                        title="Excluir projeto"
                      >✕</button>
                    </li>
                  ))
                )}
              </ul>
            )}
          </div>
        </div>

        <div className="header-right">
          <div className="user-menu">
            <span className="user-name">{user?.name}</span>
            <button className="btn-logout" onClick={handleLogout} title="Sair">Sair</button>
          </div>
        </div>
      </header>

      <main className="workspace">
        {isLoading && <div className="status-overlay">Carregando projeto...</div>}
        {error && <div className="status-overlay error">{error}</div>}

        {!videoData ? (
          <div className="empty-workspace-msg">
            Selecione um Projeto no menu superior para começar.
          </div>
        ) : (
          <>
            <section className="monitors-area">
              <aside className="left-monitor">
                <div className="cam-box">
                  <video ref={faceVideoRef} src={videoData.processed_url} muted playsInline />
                </div>
                <div className="avatar-box">
                  <div className="avatar-circle">
                    <svg viewBox="0 0 24 24" fill="white">
                      <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                    </svg>
                  </div>
                  <div className="emotion-label">{currentEmotionLabel}</div>
                </div>
              </aside>

              <article className="right-monitor">
                <div className="main-video-box">
                  <video
                    ref={mainVideoRef}
                    src={videoData.original_url}
                    onLoadedMetadata={handleLoadedMetadata}
                    playsInline
                  />
                </div>
              </article>
            </section>

            <section className="timeline-area">
              <div className="timeline-controls-bar">
                <button className="control-play-btn" onClick={togglePlay}>
                  {isPlaying ? '⏸' : '▶'}
                </button>
                <span className="timecode">{formatTime(currentTime)}</span>
                <input
                  type="range"
                  className="seek-slider"
                  min="0" max="100" step="0.01"
                  value={progressPercent || 0}
                  onMouseDown={handleSeekMouseDown}
                  onChange={handleSeekChange}
                  onMouseUp={handleSeekMouseUp}
                />
                <div className="marker-add-group">
                  <input
                    type="text"
                    className="marker-label-input"
                    value={markerLabel}
                    onChange={e => setMarkerLabel(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') handleAddMarker(); }}
                    placeholder="Rótulo do marcador"
                    maxLength={50}
                  />
                  <button className="btn-marker-add" onClick={handleAddMarker} title="Adicionar marcador no tempo atual">
                    + Marcador
                  </button>
                </div>
              </div>

              <div className="timeline-body">
                <div className="timeline-headers">
                  <div className="header-cell avatar-icon">
                    <svg viewBox="0 0 24 24" fill="white">
                      <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
                    </svg>
                  </div>
                </div>

                <div
                  className="timeline-tracks"
                  ref={timelineTracksRef}
                  onMouseDown={handleTimelineMouseDown}
                >
                  <div className="playhead" style={{ left: `${progressPercent}%` }}>
                    <div className="playhead-knob"></div>
                    <div className="playhead-line"></div>
                  </div>

                  {markers.map(m => (
                    <MarkerPin
                      key={m.marker_id}
                      marker={m}
                      duration={duration}
                      onColorChange={handleMarkerColorChange}
                    />
                  ))}

                  {dragSel && (
                    <div
                      className="drag-selection"
                      style={{
                        left:  `${dragSel.startPct}%`,
                        width: `${dragSel.endPct - dragSel.startPct}%`,
                      }}
                      onMouseDown={e => e.stopPropagation()}
                    />
                  )}

                  <div className="emotions-multitrack">
                    {EMOTIONS_CONFIG.map((emotionObj) => (
                      <div className="emotion-lane" key={emotionObj.id}>
                        <div className="lane-content">
                          {emocoesArray
                            .filter(data => data.emocao === emotionObj.id)
                            .map((block, idx) => {
                              const left  = (block.start / duration) * 100;
                              const width = ((block.end - block.start) / duration) * 100;
                              return (
                                <div
                                  key={idx}
                                  className="emotion-block"
                                  style={{ left: `${left}%`, width: `${width}%`, backgroundColor: emotionObj.color }}
                                />
                              );
                            })}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
              {dragSel && (
                <div className="bulk-replace-bar">
                  <span className="bulk-replace-info">
                    {formatTime(dragSel.startTime)} → {formatTime(dragSel.endTime)}
                  </span>
                  <select
                    className="bulk-replace-select"
                    value={bulkEmotion}
                    onChange={e => setBulkEmotion(e.target.value)}
                  >
                    <option value="">Substituir emoção por...</option>
                    {EMOTIONS_CONFIG.map(em => (
                      <option key={em.id} value={em.id}>{em.id}</option>
                    ))}
                  </select>
                  <button
                    className="bulk-replace-apply-btn"
                    onClick={handleBulkReplace}
                    disabled={!bulkEmotion}
                  >
                    Aplicar
                  </button>
                  <button
                    className="bulk-replace-cancel-btn"
                    onClick={() => { setDragSel(null); setBulkEmotion(''); }}
                  >
                    ✕
                  </button>
                </div>
              )}
            </section>
          </>
        )}
      </main>
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route
        path="/*"
        element={
          <ProtectedRoute>
            <VideoEditor />
          </ProtectedRoute>
        }
      />
    </Routes>
  );
}
