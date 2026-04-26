export const EMOTIONS_CONFIG = [
  { id: 'TRISTE', color: '#4a90e2' },     // Azul
  { id: 'SURPRESO', color: '#f5a623' },   // Amarelo
  { id: 'FELIZ', color: '#7ed321' },      // Verde
  { id: 'MEDO', color: '#9013fe' },       // Roxo
  { id: 'NEUTRO', color: '#9b9b9b' },     // Cinza
  { id: 'INDEFINIDO', color: '#000000' }  // Preto
];

export const videoUrls = {
  main: "aluno10Recorte.mp4",
  face: "quadro_fixo_aluno10Recorte.mp4"
};

export const MOCK_DURATION = 9000; 

function generateContinuousFrameData(duration) {
  const data = [];
  let currentTime = 0;
  
  while (currentTime < duration) {
    // Blocos curtos para simular a troca rápida de emoções (0.1s a 0.4s)
    const frameDuration = Math.random() * 0.3 + 0.1; 
    const endTime = Math.min(currentTime + frameDuration, duration);
    
    const randomEmotion = EMOTIONS_CONFIG[Math.floor(Math.random() * EMOTIONS_CONFIG.length)].id;
    
    data.push({
      start: currentTime,
      end: endTime,
      emotion: randomEmotion
    });
    
    // O próximo bloco começa EXATAMENTE onde o atual terminou
    currentTime = endTime; 
  }
  return data;
}

export const analysisData = generateContinuousFrameData(MOCK_DURATION);