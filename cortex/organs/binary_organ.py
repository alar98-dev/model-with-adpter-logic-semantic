import sys
import time
import os
import numpy as np

def devour_file(filepath):
    if not os.path.exists(filepath):
        print(f"Erro: Arquivo não encontrado: {filepath}")
        return

    print(f"[{filepath}] Iniciando o Órgão Binário. Estilhaçando bits...")
    
    chunk_size = 512 # 512 bytes * 8 bits = 4096 bits = 64x64
    
    try:
        with open(filepath, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                    
                # Se for menor que 512, preenche com zeros no final (Padding)
                if len(chunk) < chunk_size:
                    chunk = chunk + b'\x00' * (chunk_size - len(chunk))
                
                # Desempacota Bytes em Bits (Estilhaçamento)
                byte_array = np.frombuffer(chunk, dtype=np.uint8)
                bit_array = np.unpackbits(byte_array) # Array de 4096 elementos (0 e 1)
                
                # Mapeia 0 -> -2.0 e 1 -> +2.0 (Energia Bruta)
                tensor_injection = np.where(bit_array == 0, -2.0, 2.0).astype(np.float32)
                
                # Injeta na veia da memória compartilhada
                with open("/dev/shm/aeon_binary_injection.dat", "wb") as out_f:
                    out_f.write(tensor_injection.tobytes())
                
                # Velocidade de Devorar (Aproximadamente 10 frames por segundo)
                time.sleep(0.1)
                sys.stdout.write(".")
                sys.stdout.flush()
                
        print(f"\n[{filepath}] Arquivo completamente devorado.")
    except Exception as e:
        print(f"Falha ao devorar arquivo: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python binary_organ.py <caminho_do_arquivo>")
    else:
        devour_file(sys.argv[1])
