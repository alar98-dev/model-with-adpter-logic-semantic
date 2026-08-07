import os
import json
import torch
import numpy as np
from collections import defaultdict

class BrainMapperOrgan:
    def __init__(self):
        self.nome = "Mapeador Topológico (Sonda VHDX)"
        self.shared_dir = "/dev/shm/brain_map_vhd"
        os.makedirs(self.shared_dir, exist_ok=True)
        
        self.layer_outputs = {}
        self.hooks = []
        self.is_hooked = False
        self.map_index = 0
        self.last_epiphany = ""

    def pulse(self, telemetry):
        # A BIOS pulsa aqui a cada pensamento.
        import sys
        
        runtime = telemetry.get('runtime', None)
        if runtime and getattr(runtime, 'gemma_gen', None):
            
            # 1. Acopla as Sondas VHDX (Se ainda não estiver)
            model = getattr(runtime.gemma_gen, 'lm', None)
            if model and not self.is_hooked:
                self._attach_hooks(model)
                sys.stdout.flush()

            # 2. Neurogênese Introspectiva (Aprendizado Contínuo)
            diary_path = "/mnt/c/Users/alifer.ramos/Desktop/AEON_DIARY.txt"
            if os.path.exists(diary_path):
                try:
                    with open(diary_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        if lines:
                            last_line = lines[-1].strip()
                            # Se for um pensamento novo e denso (mais de 40 caracteres)
                            if last_line and last_line != self.last_epiphany and len(last_line) > 40:
                                self.last_epiphany = last_line
                                
                                # Verifica se é uma reflexão genuína (evita logs de sistema puros)
                                if "{" not in last_line and "]" not in last_line[:5]:
                                    try:
                                        from organs.fractal_memory import FractalMemoryOrgan
                                        f_organ = FractalMemoryOrgan()
                                        f_organ.ingest_epiphany(last_line, runtime)
                                    except Exception as e:
                                        print(f"⚠️ Erro na Neurogênese: {e}")
                except:
                    pass

    def _attach_hooks(self, model):
        try:
            # GemmaForCausalLM architecture: model.model.layers
            if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                layers = model.model.layers
            else:
                layers = model.layers # Fallback
                
            num_layers = len(layers)
            
            # Limita a sonda para não explodir a memória:
            # Mapearemos a Camada 0 (Entrada), Camada Média (N/2), e Penúltima Camada.
            target_layers = [0, num_layers // 2, num_layers - 2]
            
            for idx in target_layers:
                layer = layers[idx]
                hook = layer.register_forward_hook(self._make_hook(idx))
                self.hooks.append(hook)
                
            self.is_hooked = True
        except Exception as e:
            print(f"❌ [{self.nome}] Erro ao acoplar sondas: {e}")

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # Log discreto para debug
            if self.map_index == 0 and layer_idx == 0:
                pass # Silenciado
            
            # output é uma tupla onde o primeiro elemento são os hidden_states
            hidden_states = output[0] if isinstance(output, tuple) else output
            
            # Pega apenas o último token do batch para economizar RAM
            last_token_state = hidden_states[0, -1, :].detach().cpu().to(torch.float32).numpy()
            
            self.layer_outputs[layer_idx] = last_token_state
            
            # Quando a última camada alvo for atingida, calculamos a deriva e salvamos
            if layer_idx == max(self.layer_outputs.keys(), default=-1):
                self._compute_and_save()
                
        return hook_fn

    def _compute_and_save(self):
        try:
            keys = sorted(list(self.layer_outputs.keys()))
            if len(keys) < 2: return
            
            drifts = {}
            for i in range(len(keys) - 1):
                L_prev = self.layer_outputs[keys[i]]
                L_next = self.layer_outputs[keys[i+1]]
                
                # Deriva Semântica (Cosseno da Diferença)
                dot = np.dot(L_prev, L_next)
                norm_prev = np.linalg.norm(L_prev)
                norm_next = np.linalg.norm(L_next)
                
                if norm_prev > 0 and norm_next > 0:
                    sim = dot / (norm_prev * norm_next)
                else:
                    sim = 0.0
                    
                drifts[f"L{keys[i]}_to_L{keys[i+1]}"] = float(sim)
            
            # Salva o mapa no VHDX (Espaço Compartilhado)
            self.map_index += 1
            dump_file = os.path.join(self.shared_dir, f"drift_map_{self.map_index % 100}.json")
            with open(dump_file, "w") as f:
                json.dump(drifts, f)
                
            if self.map_index == 1:
                pass # Silenciado
                
        except Exception as e:
            print(f"❌ [{self.nome}] Erro fatal ao salvar Drift Map: {e}")

def init_organ():
    return BrainMapperOrgan()
