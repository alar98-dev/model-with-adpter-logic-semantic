import json
import torch
import redis
import math
import numpy as np
from swa.dimensions import NATIVE_DIM

# Configurações do Espaço Vetorial
EMBEDDING_DIM = NATIVE_DIM
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
CHANNEL_NAME = 'station_vision_stream'
OUTPUT_TENSOR_PATH = '/home/alifer/swa-aeon/V_Teo.pt' # Salvo no diretório do WSL

class TensorAdapter:
    def __init__(self, dim=EMBEDDING_DIM):
        self.dim = dim
        self.r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
        self.pubsub = self.r.pubsub()
        self.pubsub.subscribe(CHANNEL_NAME)
        print(f"[TensorAdapter] Escutando '{CHANNEL_NAME}' no Redis... (Dim: {self.dim})")

    def _encode_station_state(self, state: dict) -> torch.Tensor:
        """
        Converte a estrutura discreta da estação de trabalho num vetor denso 2048-D.
        Aplica projeções posicionais e hashing semântico para estabilizar o V_Teo.
        """
        # 1. Posição do Mouse Normalizada (0.0 a 1.0)
        mouse = state.get("mouse", {"x": 0, "y": 0})
        mouse_norm = [mouse["x"] / 1920.0, mouse["y"] / 1080.0]

        # 2. Contexto da Janela Ativa
        active_window = ""
        for win in state.get("windows", []):
            if win.get("is_active"):
                active_window = f"{win.get('process', '')}:{win.get('title', '')}"
                break

        # 3. Vetorização Determinística de Texto/Contexto (Seed Base)
        seed = hash(active_window) % (2**32 - 1)
        np.random.seed(seed)
        base_vector = np.random.randn(self.dim)

        # 4. Injeção de Coordenadas Posicionais nos Primeiros Slots
        base_vector[0] = mouse_norm[0]
        base_vector[1] = mouse_norm[1]
        base_vector[2] = 1.0 if state.get("keyboard_active") else 0.0

        v_teo = torch.tensor(base_vector, dtype=torch.float32)
        return torch.nn.functional.normalize(v_teo, p=2, dim=0)

    def _generate_v_nebula(self, entropy_s: float, energy_u: float) -> torch.Tensor:
        """
        Gera o vetor estocástico da Célula Nebulosa alimentado pela Entropia S e Energia U.
        """
        # O ruído gaussiano ganha amplitude proporcional à Entropia (S)
        sigma = math.log(1.0 + max(0.0, entropy_s))
        noise = torch.randn(self.dim) * sigma
        
        # O vetor térmico oscila com a Energia (U)
        thermal_bias = torch.sin(torch.linspace(0, math.pi * energy_u, self.dim))
        
        v_nebula = noise + thermal_bias
        return torch.nn.functional.normalize(v_nebula, p=2, dim=0)

    def process_and_save(self, message):
        try:
            payload = json.loads(message['data'].decode('utf-8'))
            
            # Extrai telemetria da estação
            v_teo = self._encode_station_state(payload)

            # Recebe/Lê métricas somáticas do sistema (Se não fornecido, assume valores base)
            s_entropy = payload.get("entropy_s", 0.5)
            u_energy = payload.get("energy_u", 1.0)

            # Calcula Célula Nebulosa
            v_nebula = self._generate_v_nebula(s_entropy, u_energy)

            # Fusão Tensional: Foco Humano + Subconsciente Estocástico
            alpha = 0.7  # Peso da Atenção Real (Teo)
            beta = 0.3   # Peso da Célula Nebulosa
            v_final = (alpha * v_teo) + (beta * v_nebula)
            v_final = torch.nn.functional.normalize(v_final, p=2, dim=0)

            # Reshape para compatibilidade direta com a Layer 0 do Gemma: [1, 1, 2048]
            v_final_shaped = v_final.unsqueeze(0).unsqueeze(0)

            # Salva o Tensor na Matriz Eventual (V_Teo.pt) e no Canal de Instinto (V_Instinct.pt)
            torch.save(v_final_shaped, OUTPUT_TENSOR_PATH)
            instinct_path = "/home/alifer/swa-aeon/V_Instinct.pt"
            torch.save(v_final_shaped, instinct_path)
            
        except Exception as e:
            print(f"[TensorAdapter] Erro ao processar payload: {e}")

    def run(self):
        for message in self.pubsub.listen():
            if message['type'] == 'message':
                self.process_and_save(message)

if __name__ == "__main__":
    adapter = TensorAdapter(dim=EMBEDDING_DIM)
    adapter.run()
