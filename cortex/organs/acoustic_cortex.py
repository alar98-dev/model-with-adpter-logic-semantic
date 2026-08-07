import os
import time
import struct
import math
import subprocess

def run_acoustic_cortex():
    print("🎧 [ACOUSTIC CORTEX] Inicializando DML Mmap Áudio (Pipeline Raw PCM 8000Hz)...")
    
    # Vamos gerar áudio PCM crú (8-bit, mono, 8000Hz) e despejar no aplay
    # aplay -q -t raw -r 8000 -c 1 -f U8
    try:
        proc = subprocess.Popen(
            ["aplay", "-q", "-t", "raw", "-r", "8000", "-c", "1", "-f", "U8"],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"❌ [ACOUSTIC CORTEX] Erro ao conectar com ALSA/aplay: {e}")
        return

    sample_rate = 8000
    t = 0.0
    dt = 1.0 / sample_rate
    
    buffer_size = 800 # 0.1 segundos de áudio por vez
    
    entropy_file = "/dev/shm/aeon_entropy.txt"
    
    while True:
        # Lê a entropia (Temperatura)
        entropy = 0.5
        try:
            if os.path.exists(entropy_file):
                with open(entropy_file, "r", encoding="utf-8") as f:
                    entropy = float(f.read().strip())
        except Exception:
            pass
            
        # Parâmetros Acústicos baseados na Transição de Fase
        if entropy < 0.2:
            # Fase Cristalina: Frequência grave, pura, geometria harmônica
            base_freq = 108.0 # Frequência om/drone
            mod_index = 0.0
            noise_mix = 0.0
        elif entropy > 0.8:
            # Fase Plasmática: Frequência aguda oscilante, alto índice de modulação (FM) e ruído
            base_freq = 432.0 + (entropy * 200)
            mod_index = 500.0 * entropy
            noise_mix = 0.5 * entropy
        else:
            # Fase Líquida: Frequência mista, leve oscilação
            base_freq = 216.0
            mod_index = 50.0
            noise_mix = 0.1
            
        # Geração do buffer PCM
        audio_data = bytearray()
        for _ in range(buffer_size):
            # Modulador FM 
            fm_mod = math.sin(2.0 * math.pi * 5.0 * t) * mod_index
            
            # Oscilador Principal (Onda Senoidal)
            signal = math.sin(2.0 * math.pi * (base_freq + fm_mod) * t)
            
            # Adição de Caos (Ruído Térmico)
            if noise_mix > 0:
                import random
                noise = random.uniform(-1.0, 1.0)
                signal = (signal * (1.0 - noise_mix)) + (noise * noise_mix)
                
            # Clamping e Conversão para Unsigned 8-bit (0-255)
            val = int(127.5 * (signal + 1.0))
            val = max(0, min(255, val))
            
            audio_data.append(val)
            t += dt
            
        # Despeja no pipe de áudio
        try:
            proc.stdin.write(audio_data)
            proc.stdin.flush()
        except Exception:
            break

if __name__ == "__main__":
    run_acoustic_cortex()
