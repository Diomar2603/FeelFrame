import React, { useRef, useState, useEffect } from 'react';
import './Editor.css';
import { EMOTIONS_CONFIG } from './constants/emotions';
import { videoService } from './services/videoService';

export default function VideoEditor() {
  const mainVideoRef = useRef(null);
  const faceVideoRef = useRef(null);
  const requestRef = useRef(); 
  
  // Estados da API / Sistema
  const [videoData, setVideoData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Novos Estados (Projetos e Notificações)
  const [projects, setProjects] = useState([]);
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [toastMessage, setToastMessage] = useState(null);

  // Estados do Player
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(1); 
  const [isSeeking, setIsSeeking] = useState(false); 

  // =========================================
  // FUNÇÕES DOS BOTÕES SUPERIORES
  // =========================================

  const handleImportVideo = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    try {
      setError(null);
      setToastMessage(`O vídeo "${file.name}" está sendo processado em background...`);
      await videoService.uploadVideo(file);
    } catch (err) {
      setToastMessage(null);
      setError("Falha ao enviar vídeo. Verifique sua conexão ou backend.");
    }
    e.target.value = null; 
  };

  const toggleProjectsDropdown = async () => {
    const nextState = !isDropdownOpen;
    setIsDropdownOpen(nextState);
    
    if (nextState) {
      try {
        const data = await videoService.getProjects();
        setProjects(data.videos || []); 
      } catch (err) {
        console.error("Erro ao carregar lista de projetos");
      }
    }
  };

  const loadProject = async (projectId) => {
    setIsDropdownOpen(false);
    try {
      setIsLoading(true);
      setError(null);
      const responseData = await videoService.getVideoData(projectId); 
      setVideoData(responseData);
      
      setCurrentTime(0);
      setIsPlaying(false);
    } catch (err) {
      setError('Falha ao carregar os dados do projeto.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReport = async () => {
    if (!videoData) return;
    try {
      setToastMessage("Gerando relatório em background...");
      await videoService.generateReport(videoData.video_id);
    } catch (err) {
      setToastMessage(null);
      setError("Falha ao solicitar o relatório.");
    }
  };

  const closeToast = () => setToastMessage(null);

  // =========================================
  // MOTOR DE 60FPS E CONTROLES
  // =========================================
  const updateProgress = () => {
    if (!isSeeking && mainVideoRef.current) {
      setCurrentTime(mainVideoRef.current.currentTime);
      if (faceVideoRef.current && Math.abs(mainVideoRef.current.currentTime - faceVideoRef.current.currentTime) > 0.3) {
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
    if (isPlaying) {
      mainVideoRef.current.pause();
      faceVideoRef.current.pause();
    }
  };

  const handleSeekChange = (e) => {
    const newPercent = parseFloat(e.target.value);
    const newTime = (newPercent / 100) * duration;
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

  const formatTime = (timeInSeconds) => new Date(timeInSeconds * 1000).toISOString().substr(11, 8);

  // =========================================
  // RENDERIZAÇÃO SEGURA
  // =========================================
  
  // Extrai com segurança as propriedades (Opcional Chaining para evitar erros se a API mudar)
  const progressPercent = videoData ? (currentTime / duration) * 100 : 0;
  const emocoesArray = videoData?.analysis?.emocao || [];
  
  const activeEmotionData = emocoesArray.find(d => currentTime >= d.start && currentTime < d.end) || emocoesArray[0];
  const currentEmotionLabel = activeEmotionData ? activeEmotionData.emocao : "Indefinido";

  return (
    <div className="editor-container">
      
      {toastMessage && (
        <div className="toast-notification">
          <span>{toastMessage}</span>
          <button onClick={closeToast} className="toast-close-btn">✖</button>
        </div>
      )}

      <header className="top-bar">
        <label className="btn-primary" style={{cursor: 'pointer'}}>
          Import
          <input 
            type="file" 
            accept="video/mp4,video/webm" 
            onChange={handleImportVideo} 
            style={{display: 'none'}} 
          />
        </label>
        
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
                  <li key={proj.video_id} onClick={() => loadProject(proj.video_id)}>
                    {proj.original_filename}
                  </li>
                ))
              )}
            </ul>
          )}
        </div>

        <button 
          className="btn-secondary" 
          onClick={handleReport}
          disabled={!videoData} // Desabilita se não tiver vídeo selecionado
        >
          Relatório
        </button>
      </header>

      <main className="workspace">
        {isLoading && <div className="status-overlay">Carregando projeto...</div>}
        {error && <div className="status-overlay error">{error}</div>}
        
        {/* A CORREÇÃO PRINCIPAL ESTÁ AQUI: Se não há videoData, desenha a tela inicial. */}
        {!videoData ? (
           <div className="empty-workspace-msg">
              Selecione um Projeto no menu superior para começar.
           </div>
        ) : (
          <>
            <section className="monitors-area">
              <aside className="left-monitor">
                <div className="cam-box">
                  {/* Agora é seguro ler processed_url porque videoData não é nulo */}
                  <video ref={faceVideoRef} src={videoData.processed_url} muted playsInline />
                </div>
                <div className="avatar-box">
                  <div className="avatar-circle">
                    <svg viewBox="0 0 24 24" fill="white"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>
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
                    min="0" max="100" 
                    step="0.01" 
                    value={progressPercent || 0} 
                    onMouseDown={handleSeekMouseDown}
                    onChange={handleSeekChange} 
                    onMouseUp={handleSeekMouseUp}
                />
              </div>

              <div className="timeline-body">
                <div className="timeline-headers">
                  <div className="header-cell avatar-icon">
                      <svg viewBox="0 0 24 24" fill="white"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>
                  </div>
                </div>

                <div className="timeline-tracks">
                  <div className="playhead" style={{ left: `${progressPercent}%` }}>
                    <div className="playhead-knob"></div>
                    <div className="playhead-line"></div>
                  </div>

                  <div className="emotions-multitrack">
                      {EMOTIONS_CONFIG.map((emotionObj) => (
                        <div className="emotion-lane" key={emotionObj.id}>
                          <div className="lane-content">
                            {emocoesArray
                              .filter(data => data.emocao === emotionObj.id)
                              .map((block, idx) => {
                                const left = (block.start / duration) * 100;
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
            </section>
          </>
        )}
      </main>
    </div>
  );
}