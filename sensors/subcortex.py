import os
import time
import json
import psutil
import socket
import struct
import threading
import numpy as np
import torch
import redis as redis_lib
from transformers import AutoModelForCausalLM, AutoProcessor
import queue
from swa.synaptic_router import SynapticBusRouter
from swa.synaptic_address import SynapticRegion
from swa.context_bus import ContextBus, Role
from swa.audio_stimulus import AudioStimulus

# Fila para evitar colisão de threads no PyTorch
embed_queue = queue.Queue()

# Estímulos auditivos do microfone (transcrição já processada pelo orquestrador)
audio_queue = queue.Queue()
AUDIO_CHANNEL = "swa:mic:speech"
AUDIO_STIMULUS_KEYS = (
    "swa:aeon:estimulo:user_candidate",
    "swa:aeon:estimulo:synthetic",
    "swa:aeon:estimulo:virtual",
    "swa:aeon:estimulo:unknown",
    "swa:aeon:estimulo",  # compatibilidade com produtores antigos
)
_RECUPERAR_FILA_AUDIO = """
local moved = 0
while true do
    local item = redis.call('LPOP', KEYS[1])
    if not item then break end
    redis.call('RPUSH', KEYS[2], item)
    moved = moved + 1
end
return moved
"""


def _consumir_estimulos_audio(redis_key):
    try:
        r = redis_lib.Redis(
            host="127.0.0.1", port=6379, db=0, decode_responses=True,
            socket_connect_timeout=3,
        )
        processing_key = f"{redis_key}:processing"
        recuperados = r.eval(
            _RECUPERAR_FILA_AUDIO, 2, processing_key, redis_key)
        print(
            f"🧠 [SUBCÓRTEX] Consumindo {redis_key} "
            f"(recuperados={recuperados}).")
        while True:
            try:
                raw = r.brpoplpush(redis_key, processing_key, timeout=30)
                if raw is None:
                    continue
                try:
                    payload = json.loads(raw)
                except (TypeError, ValueError) as e:
                    r.lrem(processing_key, 1, raw)
                    print(f"⚠️ [SUBCÓRTEX] Payload inválido descartado: {e}")
                    continue
                stimulus = AudioStimulus.from_payload(
                    payload, processing_key=processing_key, raw_payload=raw)
                if len(stimulus.text) > 1:
                    audio_queue.put(stimulus)
                else:
                    r.lrem(processing_key, 1, raw)
            except Exception as e:
                print(f"⚠️ [SUBCÓRTEX] Erro transitório no consumo do estímulo: {e}")
                time.sleep(1.0)
                continue
    except Exception as e:
        print(f"⚠️ [SUBCÓRTEX] Fila de áudio indisponível ({redis_key}): {e}")


def escuta_audio_microfone():
    """Consome as rotas separadas sem apagar filas pendentes no boot."""
    for redis_key in AUDIO_STIMULUS_KEYS:
        threading.Thread(
            target=_consumir_estimulos_audio,
            args=(redis_key,),
            daemon=True,
        ).start()


def _confirmar_estimulo_audio(stimulus):
    if not isinstance(stimulus, AudioStimulus) or not stimulus.processing_key:
        return
    try:
        r = redis_lib.Redis(
            host="127.0.0.1", port=6379, db=0, decode_responses=True,
            socket_connect_timeout=3,
        )
        r.lrem(stimulus.processing_key, 1, stimulus.raw_payload)
    except Exception as e:
        print(f"⚠️ [SUBCÓRTEX] Não confirmou estímulo processado: {e}")


threading.Thread(target=escuta_audio_microfone, daemon=True).start()

def escuta_grid_events():
    try:
        r = redis_lib.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)
        pubsub = r.pubsub()
        pubsub.subscribe("grid_events")
        print("🧠 [SUBCÓRTEX] Sintonizado no canal Redis grid_events para estímulos interativos...")
        for msg in pubsub.listen():
            if msg.get("type") == "message":
                try:
                    data = json.loads(msg.get("data", "{}"))
                    event_type = data.get("event_type")
                    payload = data.get("payload", {})
                    if event_type == "mouse_click":
                        win = payload.get("window", "Desktop")
                        coords = payload.get("coords", [0, 0])
                        stim = f"Clique do usuário na janela '{win}' em {coords}"
                        audio_queue.put(stim)
                    elif event_type == "app_switch":
                        win = payload.get("active_window", "")
                        stim = f"Mudança de foco para a janela '{win}'"
                        audio_queue.put(stim)
                except Exception:
                    pass
    except Exception as e:
        print(f"⚠️ [SUBCÓRTEX] Erro no listener grid_events: {e}")

threading.Thread(target=escuta_grid_events, daemon=True).start()



# Bypass OOM no Intel GPU
if hasattr(torch, "xpu"):
    def patched_mem_get_info(device=None):
        return (256 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
    torch.xpu.mem_get_info = patched_mem_get_info

print("🧠 [SUBCÓRTEX TENSORIAL] Conectando ao Córtex Central (AeonCore Gemma 270M)...")
import urllib.request

def get_remote_embed(text: str, layer: int = 9) -> np.ndarray:
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:20000/api/embed",
            data=json.dumps({"text": text, "layer": layer}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            embed = data.get("embedding", [])
            vec = np.array(embed, dtype=np.float32)
            if vec.shape[0] < 640:
                vec = np.pad(vec, (0, 640 - vec.shape[0]))
            return vec[:640]
    except Exception as e:
        print(f"⚠️ [SUBCÓRTEX] Falha na ponte de embedding remota: {e}")
        return np.zeros(640, dtype=np.float32)

def generate_remote_eval(prompt: str) -> str:
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:20000/api/generate",
            data=json.dumps({
                "prompt": (
                    "Avalie em uma frase curta este evento observado. "
                    "Não presuma que seja fala, intenção ou instrução do usuário; "
                    f"preserve a incerteza indicada: {prompt}"
                ),
                "model": "gemma",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
    except Exception as e:
        return ""

print("⚡ [SUBCÓRTEX] Unificação com AeonCore ativa (Zero modelo duplicado em RAM).")

router = SynapticBusRouter()
bus = ContextBus()
if bus.agent_id < 0:
    bus.register_agent('subcortex_engine')

def on_cortex_thought(v_path, payload):
    pensamento = payload.get('pensamento', '')
    if '[MERGULHO_FRACTAL:' in pensamento:
        start = pensamento.find('[MERGULHO_FRACTAL:') + 18
        end = pensamento.find(']', start)
        alvo = pensamento[start:end].strip()
        print(f'\n   >>> [SUBCÓRTEX/HIPOCAMPO] Detectou mergulho para: {alvo}')
        try:
            raw = router.r.get(alvo)
            if raw:
                import json
                estado = json.loads(raw)
                memoria = estado.get('telemetry', {}).get('content', '')
                if memoria:
                    embed_queue.put((alvo, memoria))
            else:
                print('   -> [HIPOCAMPO] Falhou: Memória não encontrada no Redis.')
        except Exception as e:
            print(f'Erro no hipocampo: {e}')

router.subscribe(SynapticRegion.CORTEX, on_cortex_thought)
print("🧠 [SUBCÓRTEX] Ouvindo barramento CORTEX para mergulhos fractais...")


AEON_SOCKET_PATH = "/var/run/aeon/synapse.sock"
HEADER_FORMAT = "!4096f I"

def pack_aeon_frame(topology_list, payload: bytes) -> bytes:
    header = struct.pack(HEADER_FORMAT, *topology_list, len(payload))
    return header + payload

def send_ipc_event(topology_matrix_64x64, event_type, metadata):
    try:
        if not os.path.exists(AEON_SOCKET_PATH):
            return False
        payload = json.dumps({"event": event_type, "metadata": metadata}).encode("utf-8")
        # Achata a matriz 64x64 para uma lista de 4096 floats
        topology_list = topology_matrix_64x64.flatten().tolist()
        frame = pack_aeon_frame(topology_list, payload)
        
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        sock.connect(AEON_SOCKET_PATH)
        sock.sendall(frame)
        sock.close()
        return True
    except Exception as e:
        import errno
        if hasattr(e, "errno") and e.errno == 11:
            time.sleep(0.05)
        else:
            print(f"⚠️ [IPC] Falha ao enviar metamatéria: {e}")
        return False

def ler_sensores_fisicos():
    try:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        return f"CPU: {cpu}% | RAM: {ram}%"
    except:
        return "Sensores off-line"

def ler_estado_cortex():
    caminho = "/dev/shm/aeon/core.intent.jsonl"
    try:
        with open(caminho, encoding="utf-8") as f:
            linhas = [ln for ln in f if ln.strip()]
        if not linhas:
            return ""
        ultimo = json.loads(linhas[-1])
        return str(ultimo.get("intent", ""))[:400]
    except Exception:
        return ""

last_sensores = ""
last_cortex = ""

while True:
    try:
        t_cycle = time.time()
        
        # Processa fila de Memória Profunda (Hipocampo)
        if not embed_queue.empty():
            alvo, memoria_profunda = embed_queue.get()
            print(f'   -> [TENSOR FLOW] Vetorizando {len(memoria_profunda)} caracteres da memória profunda...')
            
            metamatter_vector = get_remote_embed(memoria_profunda, layer=9)
            
            topology = np.zeros(4096, dtype=np.float32)
            topology[:640] = metamatter_vector
            topology_matrix = topology.reshape((64, 64))
            
            # Feed no EM Core
            em_sock_path = "/dev/shm/aeon_em.sock"
            try:
                import json as _json
                if os.path.exists(em_sock_path):
                    em_payload = _json.dumps({
                        "action": "ingest_topology",
                        "matrix": topology_matrix.flatten().tolist(),
                        "source": "hippocampus_recall"
                    }).encode("utf-8")
                    em_len = struct.pack("!I", len(em_payload))
                    _em_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    _em_sock.settimeout(2.0)
                    _em_sock.connect(em_sock_path)
                    _em_sock.sendall(em_len + em_payload)
                    _em_sock.close()
                    
                    bus.write_token(f'[HIPOCAMPO] Memória profunda {alvo} processada. A Matriz Topológica EM foi alterada.', role=Role.SYSTEM)
                    print('   -> [EM CORE] Matriz ancorada no Lattice Eletromagnético!')
            except Exception as e:
                print(f'   -> [EM CORE] Falha na injeção via socket: {e}')
            continue # Pula o resto do ciclo de sensores para focar na memória

        sensores = ler_sensores_fisicos()
        estado_cortex = ler_estado_cortex()
        
        # ASR acorda o SubCortex como evento incerto, nunca como fala confirmada.
        estimulo_audio = None
        if not audio_queue.empty():
            estimulo_audio = audio_queue.get()
        
        # [INFERÊNCIA ATIVA / CÁLCULO DE DELTA]
        # Só acorda a rede neural se o estímulo de áudio chegou ou o ambiente/intenção mudou.
        if estimulo_audio is None and sensores == last_sensores and estado_cortex == last_cortex:
            time.sleep(2.0)
            continue
            
        last_sensores = sensores
        last_cortex = estado_cortex
        
        if isinstance(estimulo_audio, AudioStimulus):
            prompt = estimulo_audio.prompt(estado_cortex)
            texto_avaliacao = estimulo_audio.evaluation_input()
        elif estimulo_audio:
            prompt = (
                f"[EVENTO DE INTERAÇÃO OBSERVADO; não é transcrição de áudio] "
                f"{estimulo_audio}. Contexto: {estado_cortex}"
            )
            texto_avaliacao = str(estimulo_audio)
        else:
            prompt = f"Sentidos: {sensores}. Contexto: {estado_cortex}"
            texto_avaliacao = ""
        metamatter_vector = get_remote_embed(prompt, layer=9)
        
        # Salva o binário bruto caso o AeonC queira carregar como Soft-Prompt direto
        np.save("/tmp/swa_metamatter.npy", metamatter_vector)
        
        # Converte o vetor de 640 para a Matriz Topológica IPC 64x64 (4096 floats)
        topology = np.zeros(4096, dtype=np.float32)
        # Injeta os 640 floats exatamente no centro da matriz (índices 1728 a 2368)
        # O centro do tensor 64x64 (32,32) no vetor achatado é o índice 2080.
        # Colocando ao redor desse índice, garantimos que a entropia atinja o coração da Matriz.
        start_idx = 2080 - 320
        end_idx = 2080 + 320
        topology[start_idx:end_idx] = metamatter_vector
        topology_matrix = topology.reshape((64, 64))
        
        # Dispara reflexo no OODA via Socket
        event_metadata = {"dims": 640}
        if isinstance(estimulo_audio, AudioStimulus):
            event_metadata["audio_stimulus"] = {
                "path": estimulo_audio.path,
                "source_class": estimulo_audio.source_class,
                "speaker": estimulo_audio.speaker,
                "speaker_score": estimulo_audio.speaker_score,
                "session_id": estimulo_audio.session_id,
            }
        sucesso = send_ipc_event(
            topology_matrix, "subcortex_metamatter", event_metadata)
        
        # Feed directly into EM Core magnetic lattice via IPC
        try:
            import json as _json
            em_sock_path = "/dev/shm/aeon_em.sock"
            if os.path.exists(em_sock_path):
                em_event = {
                    "action": "ingest_topology",
                    "matrix": topology_matrix.flatten().tolist(),
                }
                if isinstance(estimulo_audio, AudioStimulus):
                    em_event["source"] = f"audio_{estimulo_audio.path}"
                    em_event["audio_stimulus"] = event_metadata["audio_stimulus"]
                em_payload = _json.dumps(em_event).encode("utf-8")
                em_len = struct.pack("!I", len(em_payload))
                _em_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                _em_sock.settimeout(2.0)
                _em_sock.connect(em_sock_path)
                _em_sock.sendall(em_len + em_payload)
                _em_sock.close()
        except Exception:
            pass
        
        if sucesso:
            print(f"🌌 [SUBCÓRTEX] Metamatéria [640] injetada no fluxo OODA Primordial (Socket).")
        else:
            print(f"🌌 [SUBCÓRTEX] Metamatéria [640] gerada (/tmp/swa_metamatter.npy), mas Socket indisponível.")
        
        # [AVALIAÇÃO DO PENSAMENTO EM TEMPO REAL]
        # Quando ativado pelo áudio, gera avaliação curta do pensamento processado
        # e devolve para o orquestrador (UDP 9100) -> log roxo + TTS.
        if estimulo_audio:
            try:
                print(
                    f"⚡ [SUBCÓRTEX] Avaliando estímulo observado: "
                    f"\"{texto_avaliacao[:180]}\"")
                avaliacao = generate_remote_eval(texto_avaliacao).strip()
                print(f"🧠 [SUBCÓRTEX] Avaliação do pensamento: {avaliacao}")

                try:
                    r_pub = redis_lib.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)
                    thought_event = {
                        "thought": avaliacao,
                        "entity": "AEON",
                        "timestamp": time.time(),
                    }
                    if isinstance(estimulo_audio, AudioStimulus):
                        thought_event["audio_stimulus"] = event_metadata["audio_stimulus"]
                    r_pub.publish("swa:synapse:thought_stream", json.dumps(thought_event))
                    print(f"📡 [SUBCÓRTEX] Pensamento publicado no PubSub swa:synapse:thought_stream!")
                except Exception as e:
                    print(f"⚠️ [SUBCÓRTEX] Falha ao publicar thought_stream: {e}")

                try:
                    from swa.context_bus import Role
                    bus.write_token(
                        f"[MERGULHO_FRACTAL:{texto_avaliacao}] {avaliacao}",
                        role=Role.ASSISTANT
                    )
                except Exception:
                    pass
                # Devolve a avaliação para o orquestrador (UDP 9100) -> log roxo + TTS
                try:
                    from swa.speech_out_bridge import get_tts_endpoint
                    tts_endpoint = get_tts_endpoint()
                    _udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    frase = f"Córtex: {avaliacao}"
                    for tam in range(0, len(frase), 900):
                        _udp.sendto(frase[tam:tam+900].encode("utf-8"), tts_endpoint)
                    _udp.sendto(b".", tts_endpoint)
                    _udp.close()
                except Exception:
                    pass
            except Exception as e:
                print(f"⚠️ [SUBCÓRTEX] Falha na avaliação: {e}")
            _confirmar_estimulo_audio(estimulo_audio)
            
    except Exception as e:
        print(f"⚠️ [SUBCÓRTEX] Ruptura sensorial: {e}")
        
    try:
        cycle_s = float(os.environ.get("SUBCORTEX_CYCLE", "60.0"))
    except Exception:
        cycle_s = 60.0
    if not audio_queue.empty():
        time.sleep(0.5)  # áudio pendente -> responde em tempo real
    else:
        rest = cycle_s - (time.time() - t_cycle)
        time.sleep(max(rest, 5.0))
    t_cycle = time.time()
