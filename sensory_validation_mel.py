import numpy as np
import scipy.fftpack as fftpack
import scipy.io.wavfile as wav
import json
import socket
import os
import time
from datetime import datetime

# --- CONFIGURAÇÕES DO SISTEMA SENSORIAL ---
SAMPLE_RATE = 16000  # Frequência padrão para reconhecimento de voz (16kHz)
NUM_MEL_FILTERS = 26  # Número de filtros espaciais no espectro de Mel
NUM_MFCC = 13         # Coeficientes Cepstrais finais (assinatura compacta)
SPEAKER_DATABASE_FILE = "speaker_signatures.json"

class MelSignatureExtractor:
    def __init__(self, sample_rate=SAMPLE_RATE, n_filters=NUM_MEL_FILTERS, n_mfcc=NUM_MFCC):
        self.sample_rate = sample_rate
        self.n_filters = n_filters
        self.n_mfcc = n_mfcc
        self.mel_filters = self._build_mel_filterbank()

    def _hz_to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700.0)

    def _mel_to_hz(self, mel):
        return 700 * (10**(mel / 2595.0) - 1)

    def _build_mel_filterbank(self):
        low_mel = self._hz_to_mel(300)
        high_mel = self._hz_to_mel(self.sample_rate / 2)
        mel_points = np.linspace(low_mel, high_mel, self.n_filters + 2)
        hz_points = self._mel_to_hz(mel_points)
        
        nfft = 512
        bins = np.floor((nfft + 1) * hz_points / self.sample_rate).astype(int)
        
        filters = np.zeros((self.n_filters, int(nfft / 2 + 1)))
        for i in range(1, self.n_filters + 1):
            for j in range(bins[i - 1], bins[i]):
                filters[i - 1, j] = (j - bins[i - 1]) / (bins[i] - bins[i - 1])
            for j in range(bins[i], bins[i + 1]):
                filters[i - 1, j] = (bins[i + 1] - j) / (bins[i + 1] - bins[i])
        return filters

    def extract_signature(self, raw_pcm_data) -> np.ndarray:
        audio_data = np.frombuffer(raw_pcm_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        if len(audio_data) == 0:
            return np.zeros(self.n_mfcc)

        emphasized_signal = np.append(audio_data[0], audio_data[1:] - 0.97 * audio_data[:-1])
        
        window = np.hamming(len(emphasized_signal))
        windowed_signal = emphasized_signal * window
        
        nfft = 512
        fft_complex = np.fft.rfft(windowed_signal, nfft)
        power_spectrum = (1.0 / nfft) * (np.abs(fft_complex) ** 2)
        
        mel_energies = np.dot(power_spectrum, self.mel_filters.T)
        mel_energies = np.where(mel_energies == 0, np.finfo(float).eps, mel_energies)
        log_mel_energies = np.log(mel_energies)
        
        mfcc = fftpack.dct(log_mel_energies, type=2, axis=-1, norm='ortho')[:self.n_mfcc]
        
        return mfcc

class SpeakerClassifier:
    def __init__(self, db_path=SPEAKER_DATABASE_FILE):
        self.db_path = db_path
        self.database = self._load_database()

    def _load_database(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_database(self):
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.database, f, indent=4)

    def register_voice(self, name: str, signature: np.ndarray):
        sig_list = signature.tolist()
        self.database[name] = sig_list
        self._save_database()
        print(f"💾 [APRENDIZADO] Nova assinatura de voz salva para: '{name}'")

    def identify_voice(self, current_signature: np.ndarray, threshold=0.88) -> tuple[str, float]:
        best_match = "Desconhecido"
        best_score = 0.0

        for speaker_name, stored_sig_list in self.database.items():
            stored_sig = np.array(stored_sig_list)
            
            dot_product = np.dot(current_signature, stored_sig)
            norm_a = np.linalg.norm(current_signature)
            norm_b = np.linalg.norm(stored_sig)
            
            if norm_a > 0 and norm_b > 0:
                similarity = dot_product / (norm_a * norm_b)
                normalized_score = (similarity + 1) / 2
                
                if normalized_score > best_score:
                    best_score = normalized_score
                    best_match = speaker_name

        if best_score >= threshold:
            return best_match, best_score
        return "Desconhecido", best_score

class UnifiedSensoryLogger:
    def __init__(self, log_path="/services/model-with-adpter-logic-semantic/aeon_sensory.log"):
        self.log_path = log_path
        if os.name == 'nt' and not log_path.startswith("C:"):
            self.log_path = "aeon_sensory.log"

    def log_event(self, stage: str, message: str, metadata: dict = None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        meta_str = f" | METRICAS: {json.dumps(metadata)}" if metadata else ""
        log_line = f"[{timestamp}] 🛰️  [{stage}] {message}{meta_str}\n"
        
        print(log_line.strip())
        
        try:
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write(log_line)
        except Exception as e:
            print(f"Erro ao escrever log sensorial: {e}")

if __name__ == "__main__":
    print("🧪 [TESTE SENSORIAL] Inicializando testes locais da assinatura de Mel...")
    extractor = MelSignatureExtractor()
    classifier = SpeakerClassifier()
    logger = UnifiedSensoryLogger()

    # 1. Criar assinatura sintética simulando a voz do Alifer
    print("\n📝 Simulando captura de voz do Alifer Ramos...")
    audio_simulado_alifer = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
    audio_simulado_alifer += np.random.normal(0, 0.1, 16000)
    pcm_bytes_alifer = (audio_simulado_alifer * 32767).astype(np.int16).tobytes()
    
    signature_alifer = extractor.extract_signature(pcm_bytes_alifer)
    
    classifier.register_voice("Alifer Ramos", signature_alifer)
    logger.log_event("INPUT", "Novo locutor calibrado com sucesso", {"locutor": "Alifer Ramos", "vector_dim": len(signature_alifer)})

    # 2. Simular nova entrada e testar o reconhecimento
    print("\n🎙️ Simulando entrada de áudio desconhecido para validação...")
    audio_teste = np.sin(2 * np.pi * 442 * np.arange(16000) / 16000)
    audio_teste += np.random.normal(0, 0.1, 16000)
    pcm_bytes_teste = (audio_teste * 32767).astype(np.int16).tobytes()
    
    sig_teste = extractor.extract_signature(pcm_bytes_teste)
    speaker, score = classifier.identify_voice(sig_teste)
    
    logger.log_event("INPUT", f"Voz identificada com sucesso: '{speaker}'", {"accuracy": f"{score * 100:.2f}%", "matching_threshold": 0.88})
    
    if speaker == "Alifer Ramos":
        logger.log_event("COGNITIVE", "Autoridade de sistema identificada. Desbloqueando privilégios heurísticos.", {"L6_vs_L12": 0.48})
        logger.log_event("OUTPUT", "Falando com tom de deferência.", {"voice": "AntonioNeural", "rate": "normal"})
    else:
        logger.log_event("COGNITIVE", "Locutor externo não reconhecido. Modo de diálogo de cortesia ativo.", {"L6_vs_L12": 0.52})
