import { memo } from 'react';
import { EMOTIONS_CONFIG } from '../constants/emotions';

// Coloque os arquivos PNG em: front-end/app/constans/images/emotions/
// Nomes esperados: feliz.png, triste.png, surpreso.png, medo.png, neutro.png, indefinido.png
import felizImg    from '../../constans/images/emotions/feliz.png';
import tristeImg   from '../../constans/images/emotions/triste.png';
import surpresoImg from '../../constans/images/emotions/surpreso.png';
import medoImg     from '../../constans/images/emotions/medo.png';
import neutroImg   from '../../constans/images/emotions/neutro.png';
import indefinidoImg from '../../constans/images/emotions/indefinido.png';

const EMOTION_IMAGES = {
  Feliz:     felizImg,
  Triste:    tristeImg,
  Surpreso:  surpresoImg,
  Medo:      medoImg,
  Neutro:    neutroImg,
  Indefinido: indefinidoImg,
};

const EMOTION_COLOR = Object.fromEntries(
  EMOTIONS_CONFIG.map(e => [e.id, e.color])
);

const EmotionDisplay = memo(function EmotionDisplay({ emotion }) {
  const img   = EMOTION_IMAGES[emotion];
  const color = EMOTION_COLOR[emotion] ?? '#444444';

  return (
    <div className="emotion-display" style={{ '--emotion-color': color }}>
      {img && (
        <img
          className="emotion-display-img"
          src={img}
          alt={emotion}
        />
      )}
      <span className="emotion-display-label">{emotion}</span>
    </div>
  );
});

export default EmotionDisplay;
