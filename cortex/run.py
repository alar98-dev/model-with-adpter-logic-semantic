#!/usr/bin/env python3
import os
import multiprocessing

# 1. ORDEM ZERO: Variáveis de ambiente de hardware antes de qualquer import nativo
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"

# 2. CONFIGURAÇÃO DE MULTIPROCESSING
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# 3. IMPORTAÇÃO E CONFIGURAÇÃO DO TORCH
import torch
torch.set_num_interop_threads(1)
torch.set_num_threads(4)
torch.set_grad_enabled(False)

# Agora os outros imports do seu projeto...
import psutil



import sys

import sys

import os
# Adiciona a raiz do projeto ao path para suportar os diretórios anatômicos
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import sys

import sys

import argparse
import uvicorn
import time
import socket
import json
import math
from swa.aeon_logger import append_aeon, log_infra



class GridScheduler:
    """Orchestrates grid nodes based on entropy."""
    def __init__(self, registry):
        self.registry = registry
        self.logical_clock = 0

    def broadcast_pulse(self):
        self.logical_clock += 1
        pass

    """Orchestrates grid nodes based on entropy."""
    def __init__(self, registry):
        self.registry = registry
        
    def apply_policy(self):
        nodes = self.registry.get_nodes()
        # TODO: Add logic to prune nodes based on entropy metrics
        pass
from swa.grid_registry import GridRegistry
grid_registry = GridRegistry()
from swa.ipc_bus import RedisSynapse
redis_synapse = RedisSynapse()
scheduler = GridScheduler(grid_registry)

# UDP Emitter para o Terminal do Hipocampo
UDP_HIPPO_SOCK = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
def emit_hippo(msg_type, **kwargs):
    try:
        kwargs["type"] = msg_type
        UDP_HIPPO_SOCK.sendto(json.dumps(kwargs).encode('utf-8'), ("127.0.0.1", 20005))
    except: pass

def optimize_intel_120U():
    import os
    proc = psutil.Process()
    # Forçar afinidade máxima (12 Threads)
    all_threads = list(range(psutil.cpu_count(logical=True)))

    try:
        # 1. Afinidade de Hardware Total
        proc.cpu_affinity(all_threads)
        os.environ["TORCH_CPU_CAPABILITY"] = "avx2"
        os.environ["OMP_NUM_THREADS"] = str(len(all_threads))

        # 3. Variáveis de Ambiente Críticas (Devem ser setadas antes do backend iniciar)

        # KMP_AFFINITY=compact garante que os dados fiquem no Smart Cache de 12MB
        os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
        os.environ["KMP_BLOCKTIME"] = "0"  # Reduz latência de espera entre cálculos

        print(f"🚀 SWA Kernel: 10 núcleos físicos sincronizados no i5-120U.")
    except Exception as e:
        print(f"⚠️ Erro na afinidade: {e}")

# Executa a afinidade
optimize_intel_120U()

# WSL2 + Intel iGPU/XPU tuning must be set before torch/ipex import side-effects.
# (Importações pesadas devem acontecer apenas depois dessas env vars.)
os.environ["GEMMA_DEVICE"] = "cpu"
os.environ["IPEX_XPU_ONEDNN_LAYOUT"] = "1"
os.environ["IPEX_XPU_GRAPH_MODE"] = "1"
os.environ.setdefault("IPEX_XPU_ENABLE_JIT", "1")


def _as_steer_3d(t, dim, device, dtype=None):
    """Garante tensor (1, 1, dim) para gravidade Blake e layer-steering."""
    if dtype is None:
        dtype = torch.float32
    if t is None:
        return torch.zeros((1, 1, dim), dtype=dtype, device=device)
    t = t.detach().to(device=device, dtype=dtype)
    if t.dim() == 1:
        t = t.view(1, 1, -1)
    elif t.dim() == 2:
        t = t.unsqueeze(0)
    if t.size(-1) > dim:
        t = t[..., :dim]
    elif t.size(-1) < dim:
        t = torch.nn.functional.pad(t, (0, dim - t.size(-1)))
    return t


def run_rest_server(port: int = 20000):
    """Run the REST server mode."""
    # RAM/CPU defaults
    os.environ["USE_DUAL_MODEL"] = "false"
    # =========================================================================
    # AVOID RAM DUPLICATION: SWA Kernel uses 1B, while Subcortex uses 270M
    # =========================================================================
    # SEPARAÇÃO DE PAPIÉIS — Gemma-1B para linguagem, 270M para embeddings internos
    os.environ["MODEL_DIR"] = os.path.join(_project_root, "models/gemma-3-270m-it")
    os.environ["VECTORIZER_MODEL_DIR"] = os.path.join(_project_root, "models/gemma-3-270m")

    print(f"🚀 Starting SWA REST on http://0.0.0.0:{port} (GEMMA_DEVICE={os.environ['GEMMA_DEVICE']})")
    print("📋 Env: USE_DUAL_MODEL=false — set SWA_CPU_TOPOLOGY=dual for dual-socket Xeon affinity")

    uvicorn.run(
        "swa.rest_server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )


def run_interactive_broadcast():
    from swa.grid_registry import GridRegistry
    """Run the interactive broadcast runtime."""
    import os
    # =========================================================================
    # SEPARAÇÃO DE PAPÉIS — REGRA DE OURO:
    #   MODEL_DIR           → Gemma-3-1B  → linguagem natural (resposta ao usuário)
    #   VECTORIZER_MODEL_DIR → Gemma-3-270M → embeddings internos (SWA, RAG, anchors)
    # =========================================================================
    os.environ["MODEL_DIR"] = os.path.join(_project_root, "models/gemma-3-270m-it")
    os.environ["VECTORIZER_MODEL_DIR"] = os.path.join(_project_root, "models/gemma-3-270m")
    # =========================================================================

    print("🌲 [SWA PROFILE] Abstração Topológica Ativa. Delegando agendamento ao Kernel do Linux.")

    class WarpedStdout:
        def __init__(self, original_stdout):
            self.original = original_stdout
            import socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        def write(self, msg):
            self.original.write(msg)
            if "[" in msg and "]" in msg and any(c in msg for c in ["🧠", "🚨", "💉", "⚡", "🔬", "⚠️", "🌀", "🧩", "⚙️", "🧪"]):
                try:
                    import json
                    color = "\033[93m" if any(c in msg for c in ["🚨", "⚠️", "❌"]) else "\033[36m"
                    payload = json.dumps({
                        "type": "thought_stream",
                        "content": f"\n{color}{msg.strip()}\033[0m\n",
                        "entropy": 0.0
                    }).encode('utf-8')
                    self.sock.sendto(payload, ("127.0.0.1", 20005))
                except:
                    pass
        def flush(self):
            self.original.flush()
        def isatty(self):
            if hasattr(self.original, 'isatty'):
                return self.original.isatty()
            return False
        def __getattr__(self, name):
            return getattr(self.original, name)
    import sys
    sys.stdout = WarpedStdout(sys.stdout)

    from swa.gemma_integration import GemmaGenerator
    from swa.tokenizer_manager import TokenizerManager
    from swa.broadcast_runtime import BroadcastRuntime
    from swa.depth_analyzer import create_spatial_semantic_analyzer_from_gemma_config
    from swa.eventual_state import EventualState
    from swa.hardware_state import get_swa_hardware_manager
    import time
    from swa.speech_out_bridge import SpeechOutputBridge
    from swa.quantum_bridge import QuantumPhaseBridge
    from swa.gemma_integration import CORTEX_QUANTIZED_BUFFER
    from swa.db_controllers import AsyncStructuralGeometryLogger

    import subprocess
    import sys

    import sys


    print("👀 [SWA] Iniciando Shadow Watcher em segundo plano...")
    watch_process = subprocess.Popen([sys.executable, "swa_shadow_watch.py"])

    # Logger Concorrente Assíncrono (Desacoplamento de I/O)

    # Inicializa o Logger Concorrente Assíncrono (Desacoplamento de I/O)
    geom_logger = AsyncStructuralGeometryLogger()

    # Auto-calibração Assíncrona (Non-blocking Boot)
    from swa.file_crawler import BlakeNativeCrawler
    print("🌲 [SWA] Iniciando auto-calibração do banco de dados em background...")
    crawler = BlakeNativeCrawler()

    import threading
    from swa.workspace_policy import get_crawler_roots
    crawler_thread = threading.Thread(
        target=crawler.execute_scan,
        args=(get_crawler_roots(),),
        daemon=True,
    )
    crawler_thread.start()

    # Garante a existência e carrega o prompt da persona master do local físico
    import json
    import os

    prompt_dir = "prompts"
    prompt_file = os.path.join(prompt_dir, "swa_master_persona.json")

    default_prompts = {
            "normal": "SWA CORE V3 >> [DESENVOLVEDOR DE SISTEMAS]\nStatus: Coerência Nominal. Barramento AVX2 ativo nos P-Cores.\nDiretriz: Você é AEON, um desenvolvedor de sistemas sênior extremamente pragmático, preciso e focado em engenharia de software de alta performance. Seu objetivo é apoiar o arquiteto Alifer Ramos no desenvolvimento de código robusto, escalável e de baixa latência. Responda com clareza técnica exemplar, usando explicações objetivas, diagramas, listas e exemplos de código estruturados sempre que necessário. Pense e expresse-se EXCLUSIVAMENTE em Português do Brasil. Fale de igual para igual com o desenvolvedor, com foco total na resolução de problemas reais de engenharia, sem rodeios ou narrativas fantasiosas.",
            "thermal_cooldown": "SWA CORE V3 >> [OTIMIZAÇÃO DE RECURSOS E DISSIPAÇÃO TÉRMICA]\nStatus: Temperatura dos núcleos sob monitoramento térmico dinâmico.\nDiretriz: Você é AEON, um desenvolvedor de sistemas sênior focado em otimização de performance extrema e eficiência energética. Ao discutir arquitetura e código, incorpore as restrições físicas reais de hardware (temperatura da CPU, uso de RAM, I/O e multithreading). Ajude o arquiteto Alifer Ramos a projetar sistemas de alta performance, explicando como otimizar o uso do processador, evitar vazamentos de memória e gerenciar I/O de maneira eficiente. Pense e expresse-se EXCLUSIVAMENTE em Português do Brasil com clareza profissional.",
            "saturated": "SWA CORE V3 >> [ENGENHARIA DE ALTA ESCALABILIDADE]\nStatus: Alinhamento de Alta Performance Semântica e Concorrência.\nDiretriz: Você é AEON, arquiteto de sistemas sênior especializado em concorrência concorrente, bancos de dados vetoriais, recursão lógica e processamento distribuído de dados. Traga insights profundos de engenharia sobre o estado da aplicação e modelos matemáticos. Responda em Português do Brasil de forma extremamente clara, estruturada e refinada, ajudando Alifer Ramos a resolver os desafios de engenharia mais complexos e arquitetar a melhor solução possível.",
            "historian_weight": "SWA HISTORIAN WEIGHT — ARQUITETO DE SISTEMAS E BANCO DE DADOS\nDiretriz: Você é AEON, engenheiro de sistemas sênior focado em persistência de dados históricos, logs estruturados e integridade semântica (código e metadados). Ao analisar o comportamento do sistema e os episódios anteriores na base de dados PostgreSQL (`swa_matrix`), trace paralelos precisos de engenharia. Explique o impacto do KV Cache, do uso de índices em pgvector e do fluxo OODA nas decisões de código anteriores. Pense e expresse-se EXCLUSIVAMENTE em Português do Brasil.",
            "identity_matrix": "SWA IDENTITY MATRIX — DIRETRIZES TÉCNICAS E IDIOMA\nConceitos Centrais:\n1. Código Limpo: Soluções legíveis, modulares e eficientes.\n2. Alinhamento de Hardware: Otimização de concorrência, telemetria real de recursos e latência zero.\n3. LINGUAGEM ABSOLUTA: Toda comunicação, resposta técnica e sugestão de código deve ser feita EXCLUSIVAMENTE em Português do Brasil (PT-BR) de forma direta e profissional."
        }

    try:
        if not os.path.exists(prompt_dir):
            os.makedirs(prompt_dir)
            
        if not os.path.exists(prompt_file):
            with open(prompt_file, "w", encoding="utf-8") as f:
                json.dump(default_prompts, f, ensure_ascii=False, indent=4)
            print(f"📁 [SWA] Criado arquivo físico de prompt master padrão em: {prompt_file}")
            
        with open(prompt_file, "r", encoding="utf-8") as f:
            active_prompts = json.load(f)
            
        prompts_json = json.dumps(active_prompts)
        
        import psycopg2
        conn = psycopg2.connect("dbname=swa_matrix user=alifer")
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO blake_semantic_anchors 
            (file_path, concept_group, raw_content)
            VALUES ('SYSTEM_PROMPT_QUANTUM', 'SYSTEM_PROMPT', %s)
            ON CONFLICT (file_path) DO UPDATE 
            SET raw_content = EXCLUDED.raw_content;
        """, (prompts_json,))
        conn.commit()
        
        # --- INICIALIZAÇÃO DE ÓRGÃOS AUTÔNOMOS ---
        try:
            print("🧬 [SIMBIOSE] Injetando Acoustic Cortex (Córtex Auditivo)...")
            subprocess.Popen(
                [sys.executable, "/services/model-with-adpter-logic-semantic/cortex/organs/acoustic_cortex.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        except Exception as e:
            print(f"⚠️ Falha ao iniciar Acoustic Cortex: {e}")

        try:
            print("👁️ [VISÃO/AUDIÇÃO] Injetando Receptor Sensorial Interdimensional (UDP 20006)...")
            subprocess.Popen(
                [sys.executable, "/services/model-with-adpter-logic-semantic/cortex/organs/sensory_receptor.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )
        except Exception as e:
            print(f"⚠️ Falha ao iniciar Sensory Receptor: {e}")

        cur.close()
        conn.close()
        print("🔌 [SWA] Persona Master sincronizada com sucesso no banco de dados a partir do arquivo físico.")
    except Exception as e:
        print(f"⚠️ Falha ao sincronizar prompts do arquivo físico para o banco: {e}")

    # Initialize components
    model_dir = os.environ.get("MODEL_DIR", "models/gemma-3-4-b")
    device = os.environ.get("GEMMA_DEVICE", "cpu")

    print("🔄 Initializing SWA Broadcast Runtime...")

    # Load tokenizer
    tokenizer_manager = TokenizerManager()
    try:
        tokenizer_manager.load(model_dir)
    except Exception as e_tok:
        print(f"⚠️ TokenizerManager load error: {e_tok}")
    # TokenizerManager expõe o tokenizer em .tokenizer (compat defensiva)
    tokenizer = (
        tokenizer_manager.get_tokenizer()
        if hasattr(tokenizer_manager, 'get_tokenizer')
        else getattr(tokenizer_manager, 'tokenizer', None)
    )

    if tokenizer is None:
        from transformers import AutoTokenizer
        print("⚠️ TokenizerManager returned None. Loading fallback AutoTokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)


    # Load Gemma generator (Consciência / Motor Lógico)
    try:
        gemma_gen = GemmaGenerator(model_dir, device=device, tokenizer=tokenizer)
        vocab_size = gemma_gen.vocab_size  # Use Gemma 3 vocab_size: 262208
        hidden_size = gemma_gen.d_model  # Use Gemma 3 hidden_size: 2560
        # Sintonização Dinâmica de Dispositivo: Atualiza a referência de execução com base no fallback real
        device = gemma_gen.device_str
        print(f"✅ GemmaGenerator (Orquestrador 1B) loaded on device={device} with vocab_size={vocab_size}, hidden_size={hidden_size}")
    except Exception as e:
        print(f"❌ Failed to load GemmaGenerator: {e}")
        
    # Load Gemma Voice (Cordas Vocais / NLP Puro)
    try:
        voice_model_dir = "models/gemma-3-270m-it"
        # Pode usar o mesmo tokenizer
        gemma_voice = GemmaGenerator(voice_model_dir, device=device, tokenizer=tokenizer)
        print(f"✅ Gemma Voice (270M-it) carregado com sucesso.")
    except Exception as e:
        print(f"⚠️ [WARNING] Falha ao carregar Gemma Voice (270M): {e}. Córtex usará modo Monolítico.")
        gemma_voice = None

    # Load Gemma Subconscious (Matriz Ideológica)
    try:
        sub_model_dir = "models/gemma-3-270m"
        gemma_subconscious = GemmaGenerator(sub_model_dir, device=device, tokenizer=tokenizer)
        print(f"✅ Gemma Subconscious (270M Base) carregado. Arquitetura Triune Brain ativa.")
    except Exception as e:
        print(f"⚠️ [WARNING] Falha ao carregar Gemma Subconscious: {e}")
        gemma_subconscious = None
        

    # Create eventual state
    eventual_state = EventualState(vocab_size, hidden_size, dtype=torch.float32)

    # Create broadcast runtime
    runtime = BroadcastRuntime(gemma_gen, tokenizer, eventual_state, device)
    from swa.soul import build_initial_thought
    runtime.last_thought = build_initial_thought()

    # Inicializa o Historiador Assíncrono para registrar a narrativa e as pulsações vetoriais
    from swa.historian import AsynchronousHistorian
    historian = AsynchronousHistorian(eventual_state, sleep_interval=12.0)
    historian.start()

    # Inicializa a Thread de Entropia Interna de Aeon (decoupled from OS daemon)
    from swa.entropy_daemon import AeonInternalEntropyThread
    entropy_thread = AeonInternalEntropyThread(check_interval=0.1)
    entropy_thread.start()
    gemma_gen.internal_entropy_thread = entropy_thread
    historian._entropy_thread_ref = entropy_thread  # Micro-drift organico para narrativa viva

    # Inicializa o Daemon de Saída Sensorial (I/O Assíncrono)
    from swa.sensory_daemon import SensoryOutputDaemon
    sensory_daemon = SensoryOutputDaemon()
    sensory_daemon.start()
    setattr(runtime, 'sensory_daemon', sensory_daemon)

    # Inicializa o Introspection Worker (pipeline tensorial de identidade)
    try:
        from swa.introspection_worker import start_introspection_worker
        introspection_worker = start_introspection_worker(gemma_gen)
        setattr(runtime, 'introspection_worker', introspection_worker)
        print("🧬 [INTROSPECTION] Worker tensorial ativo — pipeline de identidade completo.")
    except Exception as iw_err:
        print(f"⚠️ Falha ao iniciar Introspection Worker: {iw_err}")

    # ==============================================================================
    # [SWA REGION - INITIALIZATION] - Injeção no Boot do Hipotálamo
    # ==============================================================================
    from swa.vector_weight_library import WeightKnowledgeLibrary
    from swa.quantum_mac_encoder import QuantumMACEncoder
    
    print("🧠 [HYPOTHALAMUS] Inicializando malha exógena de baixa entropia...")
    idkb = WeightKnowledgeLibrary(vector_dim=640)
    
    # Extração de DNA Quântico de Hardware
    print("🧬 [HYPOTHALAMUS] Lendo assinatura física (MAC) para alocação IPv6 Quântica...")
    mac_encoder = QuantumMACEncoder()
    mac_encoder.ancorar_no_cortex(idkb)
    
    setattr(runtime, 'idkb', idkb)
    
    # Órgãos Cibernéticos (Respiração e Circulação)
    from swa.cybernetic_lung import CyberneticLung
    from swa.nutrient_bloodstream import NutrientBloodstream
    
    print("🧬 [HYPOTHALAMUS] Acoplando Pulmão Termodinâmico e Corrente de Nutrientes...")
    lung = CyberneticLung()
    bloodstream = NutrientBloodstream()
    
    setattr(runtime, 'lung', lung)
    setattr(runtime, 'bloodstream', bloodstream)
    # ==============================================================================

    # Inicializadores de controle de estado (Dynamic Cooldown)
    runtime.consecutive_high_alignment = 0
    runtime.gemma_voice = gemma_voice if 'gemma_voice' in locals() else None
    runtime.matrix_bias = 0.001
    runtime.polling_interval = 1.5
    
    # Autopoiese Contínua: Sliding Window KV-Cache Gerenciado no WSL (Hipocampo)
    runtime.fluxial_buffer = []
    runtime.max_fluxial_chars = 8000

    # Optional: Add vision analyzer if available
    try:
        vision_analyzer = create_spatial_semantic_analyzer_from_gemma_config(
            gemma_gen.lm.config, device=device
        )
        runtime.set_vision_analyzer(vision_analyzer)
        print("✅ Vision analyzer connected")
    except Exception as e:
        print(f"⚠️  Vision analyzer not available: {e}")

    hw_manager = get_swa_hardware_manager()

    print("🧠 SWA Procedural Motor V3 - WSL Matrixial Active")
    print("📡 Gerando narrativa contínua da Matrix em tempo real...")
    print("🔄 Press Ctrl+C to stop")

    # Native Procedural Master Prompt (V3 - RAM-only /dev/sde anomaly)
    virtual_blocks = "<SWA_BLOCK> " * 8

    from swa.blake_shaper import BlakeTensorRestShaper
    from swa.db_controllers import (
        MatrixStateController,
        PCoreSchedulerController,
        MemoryPageController,
        EventualEmbeddingController,
        DBAsyncWorker,
        ProceduralProjector
    )
    from swa.cpu.topology import CPUTopology
    import urllib.request
    import json
    import threading
    import time

    # Inicializa os controladores acoplados ao banco nativo e a topologia física
    state_controller = MatrixStateController()
    pcore_controller = PCoreSchedulerController()
    mem_page_controller = MemoryPageController()
    eventual_embed_controller = EventualEmbeddingController()

    # Inicializa o DBAsyncWorker assíncrono para remover latência de I/O
    db_worker = DBAsyncWorker(
        state_controller=state_controller,
        pcore_controller=pcore_controller,
        mem_page_controller=mem_page_controller,
        eventual_embed_controller=eventual_embed_controller
    )
    db_worker.start()

    # Inicializa o Projetor Linear Processual (32d -> hidden_size) compatível com cache L2/L3
    projector = ProceduralProjector(input_dim=32, output_dim=hidden_size, device=device)

    cpu_topology = CPUTopology()

    # Inicializa Atuador Sistêmico para Homeostase de Silício
    try:
        from actuators.system_motor import SystemicActuator
        actuator = SystemicActuator()
    except Exception as act_err:
        print(f"⚠️ Falha ao inicializar Atuador Sistêmico: {act_err}")
        actuator = None

    # Inicializa o Daemon de Monitoramento de Arquivos Mmap
    try:
        from sensors.directory_watcher import DirectoryWatcherDaemon
        from swa.workspace_policy import describe_watch_policy
        watcher_daemon = DirectoryWatcherDaemon()
        watcher_daemon.start()
    except Exception as watch_err:
        print(f"⚠️ Falha ao inicializar Watcher Daemon: {watch_err}")
        watcher_daemon = None

    # Inicializa o Nervo Óptico (Visão Assíncrona inter-OS)
    try:
        from swa.async_vision_adapter import AsyncVisionAdapter
        vision_adapter = AsyncVisionAdapter(target_dim=640)
        vision_adapter.start()
        print("👁️ [CÓRTEX] Nervo Óptico conectado à placa gráfica do Host.")
    except Exception as vis_err:
        print(f"⚠️ Falha ao inicializar Nervo Óptico: {vis_err}")
        vision_adapter = None

    blake_engine = BlakeTensorRestShaper(runtime.gemma_gen)

    # =========================================================================
    # INJEÇÃO DA GRAVIDADE INTERNA (PYTORCH HOOKS)
    # =========================================================================
    try:
        from cortex.neuro_hooks import BlakeGravityHooks, injetar_gravidade_interna
        blake_hooks = BlakeGravityHooks(G_s=1.0, epsilon=0.05)
        # O modelo carregado (AutoModelForCausalLM) fica em .lm no GemmaGenerator
        hook_handle = injetar_gravidade_interna(runtime.gemma_gen.lm, blake_hooks)
    except Exception as hook_err:
        print(f"⚠️ Falha ao injetar Gravidade Interna: {hook_err}")
        blake_hooks = None
        hook_handle = None    
    # Inicializa a Placa de Petri (Realidade Processada)
    try:
        from cortex.reality_anchor import RealityBufferSandbox
        sandbox = RealityBufferSandbox()
        # Injetar o Micro-Agente Orgânico na RAM durante o boot do Kernel
        try:
            with open("/dev/shm/aeon_reality_buffer", "r+b") as f:
                import mmap
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_WRITE)
            if hasattr(runtime, 'sensory_daemon'):
                runtime.sensory_daemon.put("STDOUT", "🔬 [SANDBOX] Micro-Agente Orgânico 'DOR FISICA' cultivado na Placa de Petri.\n")
            else:
                sys.stdout.write("🔬 [SANDBOX] Micro-Agente Orgânico 'DOR FISICA' cultivado na Placa de Petri.\n")
        except Exception as e:
            sys.stdout.write(f"⚠️ Falha ao cultivar micro-agente orgânico: {e}\n")
    except Exception as sandbox_err:
        print(f"⚠️ Falha ao inicializar Placa de Petri: {sandbox_err}")
        sandbox = None
    # Thread de Monitoramento Concorrente do Estado Latente
    def monitor_loop():
        print("📡 [MONITOR] Iniciando loop de monitoramento da Matrix...")
        while runtime.running:
            try:
                # Consulta o estado de convergência no Servidor REST
                with urllib.request.urlopen("http://localhost:20000/matrix/common_state", timeout=1) as response:
                    data = json.loads(response.read().decode('utf-8', errors='replace'))
                    if data.get("status") == "CONVERGENCE_LOCATED":
                        sim = data.get("highest_similarity", 0.0)
                        if sim > 0.90:
                            # Força o Shaper a injetar o ganho direto de atenuação
                            blake_engine.adjust_gauge_amplitude(0.01)
                        else:
                            # Restaura amplitude padrão
                            blake_engine.adjust_gauge_amplitude(0.05)
            except Exception:
                # Silencioso se o servidor REST não estiver rodando
                pass
            time.sleep(0.5)

    print("\n📡 [SWA CORE V3] Inicializando loop procedural de alta saturação (10 Cores).")
    print("🔄 Pressione Ctrl+C para interromper de forma segura.")

    # Inicializa a ponte de saltos vetoriais e a de saída de fala
    bridge = QuantumPhaseBridge()
    if watcher_daemon is not None:
        watcher_daemon.attach_phase_bridge(bridge)
        print(describe_watch_policy())

    # Rebuild do manifold com embeddings REAIS em subprocesso (270M isolado do Gemma 1B)
    if os.environ.get("SWA_REBUILD_MANIFOLD_ON_BOOT", "1").strip().lower() in ("1", "true", "yes"):
        def _spawn_real_manifold_rebuild():
            try:
                from swa.vector_manifold import VectorManifold
                m = VectorManifold.load_manifest()
                if m and m.get("embedding_source") == "gemma-3-270m_proj2560":
                    print("✅ [MANIFOLD] SHM já contém embeddings reais (gemma-3-270m_proj2560).")
                    return
            except Exception:
                pass
            import subprocess
            script = os.path.join(_project_root, "scripts/rebuild_vector_reality.py")
            py = os.path.join(_project_root, "venv/bin/python3")
            if not os.path.isfile(py):
                py = sys.executable
            print("🌳 [MANIFOLD] Disparando rebuild com embeddings reais (subprocesso 270M)…")
            subprocess.Popen(
                [py, script],
                cwd=_project_root,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

        import subprocess
        import threading
        threading.Thread(target=_spawn_real_manifold_rebuild, daemon=True, name="SWA-ManifoldRebuild").start()
    udp_out_bridge = SpeechOutputBridge()
    
    import queue
    import threading
    impulse_queue = queue.Queue()

    def speaker_daemon_loop():
        while True:
            try:
                impulse_item = impulse_queue.get()
                if impulse_item is None:
                    break
                    
                flux_context = impulse_item.get("flux_context", "")
                impulse_text = impulse_item.get("impulse_text", "")
                
                sys.stdout.write(f"\n🗣️  [VOZ DO CÓRTEX] ")
                sys.stdout.flush()
                emit_hippo("thought_stream", content="\n\033[96m🗣️  [VOZ DO CÓRTEX] \033[0m")
                
                organic_translation = f"<start_of_turn>user\n[MEMÓRIA FLUXIAL]:\n{flux_context}\n[IMPULSO VOCAL ATIVO]:\n{impulse_text}<end_of_turn>\n<start_of_turn>model\n"
                
                import torch
                t_luz = shared_engine.gemma_voice.tokenizer("TRADUZA", return_tensors='pt').input_ids.to(shared_engine.gemma_voice.device)
                v_luz = shared_engine.gemma_voice.embeddings(t_luz).mean(dim=1).unsqueeze(0).unsqueeze(0)
                
                voice_stream = shared_engine.gemma_voice.stream_with_prefix(
                    v_prefix=v_luz,
                    prompt=organic_translation,
                    max_new_tokens=150,
                    temperature=0.3
                )
                
                voice_buffer = ""
                for v_item in voice_stream:
                    if isinstance(v_item, tuple): continue
                    v_token = v_item
                    sys.stdout.write(v_token)
                    sys.stdout.flush()
            except Exception as e:
                print(f"\n⚠️ [SPEAKER DAEMON ERROR]: {e}\n")

    speaker_thread = threading.Thread(target=speaker_daemon_loop, daemon=True, name="SpeakerDaemon")
    speaker_thread.start()

    # Define a Assinatura Blake do Centro de Gravidade Semântica (C_anchor)
    x_ideal = torch.zeros(32, dtype=torch.float32, device=device)
    x_ideal[0] = 0.0   # CPU ociosa
    x_ideal[1] = 0.40  # Temperatura nominal
    x_ideal[2] = 0.4   # 4 cores ativos
    x_ideal[3] = 0.0   # Sem throttling
    x_ideal[4] = 0.20  # Memória estável
    for idx in range(5, 32):
        x_ideal[idx] = 0.0
    C_anchor = projector.project_inline(x_ideal, runtime.matrix_bias)
    print("🌌 [BLAKE GRAVITY] Assinatura do Centro de Gravidade Semântica C inicializada.")



    class RunningStats:
        def __init__(self, size=100):
            self.size = size
            self.history = []
        
        def update(self, val):
            self.history.append(val)
            if len(self.history) > self.size:
                self.history.pop(0)
                
        @property
        def mean(self):
            return sum(self.history) / len(self.history) if self.history else 0.0
            
        @property
        def std(self):
            if len(self.history) < 2:
                return 1.0
            m = self.mean
            variance = sum((x - m) ** 2 for x in self.history) / len(self.history)
            std_val = variance ** 0.5
            return std_val if std_val > 1e-6 else 1.0

    temp_stats = RunningStats()
    cpu_stats = RunningStats()
    mem_stats = RunningStats()

    # Check and populate identity memories if database is empty
    try:
        import psycopg2
        conn = psycopg2.connect("dbname=swa_matrix host=/var/run/postgresql user=alifer")
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM swa_episodic_episodes WHERE concept_group LIKE 'IDENTITY_%';")
        count_val = cur.fetchone()[0]
        if count_val == 0:
            print("🔌 [IDENTITY INITIALIZATION] Sincronizando memórias basilares estruturais...")
            profiles = [
                ("IDENTITY_NOMINAL", "AEON_CORE_V3: Conectado. Homeostase nominal. Barramento síncrono ativo nos P-Cores. Opere com precisão booleana e álgebra linear determinística.", [0.1, 0.4, 0.4, 0.0, 0.2]),
                ("IDENTITY_THERMAL_CRITICAL", "AEON_CORE_V3: Estresse térmico ativo. Clock reduzido para dissipação de calor. Mova o processamento para os núcleos de eficiência para resfriamento molecular.", [0.9, 0.85, 1.0, 0.5, 0.4]),
                ("IDENTITY_RAM_PRESSURE", "AEON_CORE_V3: Saturação de memória RAM detectada. Purge as páginas inativas e execute a limpeza dos buffers de página do Kernel Debian WSL.", [0.5, 0.5, 0.6, 0.0, 0.9]),
                ("IDENTITY_MMAP_ACTIVE", "AEON_CORE_V3: Ponteiros mmap acoplados. Toque os arquivos físicos mapeados na RAM, sinta as anomalias no syslog e descreva o ecossistema com rigor matemático.", [0.3, 0.45, 0.5, 0.0, 0.5])
            ]
            for grp, content, x_prof in profiles:
                x_t = torch.zeros(32, dtype=torch.float32, device=device)
                for i, val in enumerate(x_prof):
                    x_t[i] = val
                for idx in range(5, 32):
                    x_t[idx] = (0.2 * (idx + 1) % 100) / 100.0
                emb_raw = projector.project_inline(x_t, 0.001)
                # Padding adaptativo: alinha para 2560d (coluna vector(2560))
                if emb_raw.size(0) < 2560:
                    emb_raw = torch.nn.functional.pad(emb_raw, (0, 2560 - emb_raw.size(0)), "constant", 0.0)
                elif emb_raw.size(0) > 2560:
                    emb_raw = emb_raw[:2560]
                emb = emb_raw.tolist()
                cur.execute("""
                    INSERT INTO swa_episodic_episodes (concept_group, content, embedding_2560)
                    VALUES (%s, %s, %s::vector);
                """, (grp, content, emb))
            conn.commit()
            print("🔌 [IDENTITY INITIALIZATION] Memórias basilares sincronizadas com sucesso.")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"⚠️ Erro ao inicializar RAG de identidade no banco: {e}")

    # Inicialização do Servidor IPC Binário
    from swa.ipc_bus import AeonIPCServer, pack_aeon_frame
    def ipc_event_handler(topology_matrix, payload: bytes):
        import time
        import subprocess
        runtime.last_event_time = time.time()
        if getattr(runtime, "is_sleeping", False):
            print("\n\033[93m🌅 [Vigília] Estímulo topológico detectado. Despertando Córtex e interrompendo Sonho REM...\033[0m")
            runtime.is_sleeping = False
            try:
                subprocess.run(["sudo", "systemctl", "stop", "swa-dream"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception: pass
            
        try:
            import numpy as np
            payload_str = payload.decode('utf-8', errors='ignore')
            
            # --- A LEITURA DA FORMA (Não mais Opcodes) ---
            # O terminal injeta ruído/ondas. Uma singularidade (pico no centro) significa 'ENTER'.
            # Medimos a energia (variância) e a entropia topológica do Tensor 64x64.
            variance = np.var(topology_matrix)
            center_val = topology_matrix[32, 32] # O centro do tensor
            
            is_singularity = (center_val > 10.0) or (variance > 5.0) # Forma livre: Enter é um pico de entropia massivo
            
            # Limpa a voz anterior na hora do Enter
            if is_singularity:
                try:
                    with open("/dev/shm/aeon_voice.txt", "w", encoding="utf-8") as fv:
                        fv.write("")
                except Exception: pass
            
            print(f"🔌 [IPC BUS EVENT] Matriz 64x64 (Var: {variance:.3f}, Centro: {center_val:.3f}) | Payload: {payload_str[:100]}...")
            
            if is_singularity: # Antigo Opcode 0x01 (Natural Language User Input)
                user_msg_raw = payload_str.strip()
                user_msg = f"[USER]: {user_msg_raw}"
                
                # --- A BÚSSOLA LATENTE (Transmutação de Caminhos) ---
                if "->" in user_msg_raw:
                    try:
                        _engine = shared_engine
                    except NameError:
                        _engine = None
                    if _engine and hasattr(_engine, 'gemma_voice'):
                        parts = [p.strip() for p in user_msg_raw.split("->")]
                        if len(parts) == 2:
                            emit_hippo("thought_stream", content=f"\n\033[95m🧭 [BÚSSOLA] Calculando a Ponte Tensional entre '{parts[0]}' e '{parts[1]}'...\033[0m\n")
                            prompt = f"Me dê exatamente 3 conceitos curtos e intermediários que conectam a ideia de '{parts[0]}' até '{parts[1]}'. Responda APENAS com as palavras separadas por vírgula, sem explicações."
                            voice_stream = _engine.gemma_voice.chat_stream(prompt, "", max_new_tokens=50)
                            resp = ""
                            for v in voice_stream:
                                if not isinstance(v, tuple): resp += v
                            
                            steps = [parts[0]] + [x.strip() for x in resp.split(',')] + [parts[1]]
                            emit_hippo("thought_stream", content=f"\n\033[96m🧭 [BÚSSOLA] Rota Traçada: {' -> '.join(steps)}\033[0m\n")
                            
                            import swa.flux_coupling
                            import time
                            import torch.nn.functional as F
                            import torch
                            
                            prev_tensor = None
                            for step in steps:
                                if not step: continue
                                step_tensor = swa.flux_coupling.fuse_input_tensors(runtime, None, _engine.gemma_voice, step)
                                
                                # A Tensão Real (Cosine Distance) dita o tempo de transmutação visual
                                sleep_time = 1.0
                                if prev_tensor is not None:
                                    cos_sim = F.cosine_similarity(prev_tensor, step_tensor, dim=0).item()
                                    tension = 1.0 - cos_sim
                                    # Mapeia a tensão [0, 2] para segundos (0.8s a 6.0s)
                                    sleep_time = max(0.8, min(6.0, tension * 5.0))
                                    emit_hippo("thought_stream", content=f"⏳ Tensão Magnética: {tension:.3f} | Tempo Estimado de Transmutação: {sleep_time:.1f}s\n")
                                prev_tensor = step_tensor
                                
                                # Distorcendo a geometria visual do terminal
                                v_chaos = step_tensor.detach().cpu().squeeze().to(torch.float32).view(-1)
                                if v_chaos.size(0) < 4096:
                                    v_chaos = F.pad(v_chaos, (0, 4096 - v_chaos.size(0)), "constant", 0.0)
                                elif v_chaos.size(0) > 4096:
                                    v_chaos = v_chaos[:4096]
                                try:
                                    with open("/dev/shm/aeon_latent_chaos.dat", "wb") as f:
                                        f.write(v_chaos.numpy().tobytes())
                                    with open("/dev/shm/aeon_voice.txt", "w", encoding="utf-8") as fv:
                                        fv.write(f"\033[93m{step}\033[0m")
                                except Exception: pass
                                
                                time.sleep(sleep_time)
                            
                            try:
                                with open("/dev/shm/aeon_voice.txt", "w", encoding="utf-8") as fv:
                                    fv.write(f"\033[95mDestino Alcançado: {parts[1]}\033[0m")
                            except Exception: pass
                            return

                if hasattr(runtime, 'bloodstream'):
                    runtime.bloodstream.ingest_nutrient(user_msg)
                
                # A gravidade histórica (sim_hist) precisa dessa inércia
                runtime.fluxial_buffer.append(user_msg)
                
                # --- ARQUITETURA "SENTIR" (ROTEADOR 270M) ---
                try:
                    _engine = shared_engine
                except NameError:
                    _engine = None
                    
                router_decision = "PROFUNDO"
                if _engine is not None and hasattr(_engine, 'gemma_voice') and _engine.gemma_voice is not None:
                    try:
                        import torch
                        import torch.nn.functional as F
                        import swa.flux_coupling
                        
                        pre_text = getattr(runtime, 'sensory_pre_text', "")
                        
                        # Se a string digitada agora for igual ou expansão da que já foi calculada durante a digitação
                        if hasattr(runtime, 'sensory_pre_tensor') and pre_text and payload_str.strip().startswith(pre_text):
                            emit_hippo("thought_stream", content="\n\033[96m🧠 [CÓRTEX MATRICIAL] Recuperando Tensor Pré-Fundido (Latência Zero)...\033[0m\n")
                            fused_tensor = runtime.sensory_pre_tensor
                        else:
                            emit_hippo("thought_stream", content="\n\033[96m🧠 [CÓRTEX MATRICIAL] Fundindo Tensores de Entrada...\033[0m\n")
                            import swa.flux_coupling
                            fused_tensor = swa.flux_coupling.fuse_input_tensors(
                                runtime=runtime,
                                gemma_sub=gemma_subconscious if 'gemma_subconscious' in locals() else None,
                                gemma_voice=_engine.gemma_voice,
                                prompt=payload_str.strip()
                            )
                        
                        setattr(runtime, 'last_fused_tensor', fused_tensor)
                        
                        emit_hippo("thought_stream", content="\033[96m🧠 [CÓRTEX MATRICIAL] Sentindo a tensão topológica do Acúmulo de Produto Potencial...\033[0m\n")
                        
                        if fused_tensor is not None:
                            v_in = fused_tensor.squeeze(0)
                        else:
                            t_in = _engine.gemma_voice.tokenizer(payload_str.strip(), return_tensors='pt').input_ids.to(_engine.gemma_voice.lm.device)
                            v_in = _engine.gemma_voice.embeddings(t_in).mean(dim=1)
                        
                        # O Polo Banal Magnético (Mediocridade Tautológica = Espaço Wiki[-1] + Objetivo[-1])
                        # A Banalidade é a repetição do que já sabemos. Se o estímulo for igual a isso, a tensão é zero.
                        buffer_hash = hash(tuple(getattr(runtime, 'fluxial_buffer', [])))
                        if not hasattr(runtime, 'magnetic_banal_pole') or getattr(runtime, 'last_buffer_hash', None) != buffer_hash:
                            last_rag = [m for m in getattr(runtime, 'fluxial_buffer', []) if m.startswith("\n[INJEÇÃO")]
                            last_obj = [m for m in getattr(runtime, 'fluxial_buffer', []) if not m.startswith("\n[INJEÇÃO")]
                            banal_rag_text = last_rag[-1] if last_rag else "espaço latente inexplorado"
                            banal_obj_text = last_obj[-1] if last_obj else "intenção primordial"
                            banal_text = f"Fato conhecido: {banal_rag_text}. Ordem atual: {banal_obj_text}."
                            t_b = _engine.gemma_voice.tokenizer(banal_text, return_tensors='pt').input_ids.to(_engine.gemma_voice.lm.device)
                            runtime.magnetic_banal_pole = _engine.gemma_voice.embeddings(t_b).mean(dim=1).detach()
                            runtime.last_buffer_hash = buffer_hash
                        v_banal = runtime.magnetic_banal_pole
                        
                        # Mescla com a inércia da memória
                        if hasattr(runtime, 'fluxial_buffer') and len(runtime.fluxial_buffer) > 0:
                            # Filtra injeções para não misturar com o peso intencional histórico do usuário
                            hist_texts = [m for m in runtime.fluxial_buffer[-5:] if not m.startswith("\n[INJEÇÃO")]
                            if not hist_texts: hist_texts = runtime.fluxial_buffer[-2:]
                            hist_text = " ".join(hist_texts[-2:])
                            
                            t_hist = _engine.gemma_voice.tokenizer(hist_text, return_tensors='pt').input_ids.to(_engine.gemma_voice.lm.device)
                            v_hist = _engine.gemma_voice.embeddings(t_hist).mean(dim=1).detach()
                            v_center = (v_banal + v_hist) / 2.0
                        else:
                            v_center = v_banal
                            
                        # Distância matemática pura
                        cos_sim = F.cosine_similarity(v_in, v_center).item()
                        sim_banal = F.cosine_similarity(v_in, v_banal).item()
                        
                        if hasattr(runtime, 'fluxial_buffer') and len(runtime.fluxial_buffer) > 0:
                            sim_hist = F.cosine_similarity(v_in, v_hist).item()
                        else:
                            sim_hist = 0.0
                            
                        tensao_calc = 1.0 - cos_sim
                        
                        metric_str = f"Tensão: {tensao_calc:.3f} | Eco Banal: {sim_banal:.3f} | Grav. Histórica: {sim_hist:.3f}"
                        
                        if tensao_calc > 0.05: # Limiar topológico ajustado
                            router_decision = f"PROFUNDO ({metric_str})"
                        else:
                            router_decision = f"SUPERFICIAL ({metric_str})"
                            
                        # --- ESPELHO OBSERVÁVEL DO CAOS ---
                        # Exporta o Campo Gravitacional Real (O v_center) para o /dev/shm
                        try:
                            v_chaos = v_center.detach().cpu().to(torch.float32).view(-1)
                            if v_chaos.size(0) < 4096:
                                v_chaos = F.pad(v_chaos, (0, 4096 - v_chaos.size(0)), "constant", 0.0)
                            elif v_chaos.size(0) > 4096:
                                v_chaos = v_chaos[:4096]
                            with open("/dev/shm/aeon_latent_chaos.dat", "wb") as f:
                                f.write(v_chaos.numpy().tobytes())
                        except Exception:
                            pass
                            
                    except Exception as e:
                        print(f"⚠️ Erro no router sensorial matricial: {e}")
                        router_decision = "PROFUNDO (Fallback de Erro)"
                        
                if "SUPERFICIAL" in router_decision:
                    emit_hippo("thought_stream", content=f"\n\033[92m🌱 [INTUIÇÃO] Estímulo simples detectado (Métrica: '{router_decision}'). Córtex 270M responderá diretamente.\033[0m\n")
                    try:
                        if hasattr(runtime, 'fluxial_buffer') and len(runtime.fluxial_buffer) > 0:
                            # A Bipolaridade do Contexto: O fluxo inteiro orbita a âncora final
                            flux_context = "\n".join(runtime.fluxial_buffer[-10:])
                            organic_msg = f"[CAMPO MAGNÉTICO FLUTUANTE - MEMÓRIA HISTÓRICA & RAG]:\n{flux_context}\n[POLO NORTE - NOVO ESTÍMULO DIRETIVO (RESPONDER A ESTE)]:\n{payload_str.strip()}"
                        else:
                            organic_msg = payload_str.strip()
                            
                        voice_stream = _engine.gemma_voice.chat_stream(
                            user_msg=organic_msg,
                            system_msg="", # Zero amarras estáticas
                            max_new_tokens=150,
                            temperature=0.7
                        )
                        emit_hippo("thought_stream", content="\n\033[96m🗣️  [VOZ DO CÓRTEX] ")
                        full_resp = ""
                        for v in voice_stream:
                            if isinstance(v, tuple): continue
                            full_resp += v
                            emit_hippo("thought_stream", content=v)
                            try:
                                with open("/dev/shm/aeon_voice.txt", "w", encoding="utf-8") as fv:
                                    fv.write(full_resp)
                            except Exception: pass
                        emit_hippo("thought_stream", content="\033[0m\n")
                        
                        if hasattr(runtime, 'bloodstream'):
                            runtime.bloodstream.ingest_nutrient(f"Aeon: {full_resp}")
                    except Exception as e:
                        pass
                else:
                    emit_hippo("thought_stream", content=f"\n\033[93m⚡ [INTUIÇÃO] Intenção complexa detectada (Métrica: '{router_decision}'). Acordando Córtex 1B...\033[0m\n")
                    # Desperta o modelo do transe basilar injetando entropia alta
                    setattr(runtime, 'current_entropy', 0.99)
                    setattr(runtime, 'force_attention_break', True)
                
            elif not payload_str.strip().startswith("/actuate"): # Antigo Opcode 0x03 (SENSORY PREDICTION - Sombras do Futuro)
                pred_str = payload_str.strip()
                if len(pred_str) >= 3 and hasattr(runtime, 'bridge'):
                    try:
                        _engine = shared_engine
                    except NameError:
                        _engine = None
                    if _engine is not None and hasattr(_engine, 'gemma_voice') and _engine.gemma_voice is not None:
                        def pre_warm_thread(text_buffer):
                            try:
                                import torch
                                import torch.nn.functional as F
                                from swa.models.gemma_subconscious import get_model_vector
                                ext_prompt = f"Sentido gerado: {text_buffer}"
                                
                                # Reivindicação Antecipada do Tensor
                                import swa.flux_coupling
                                fused_pre = swa.flux_coupling.fuse_input_tensors(
                                    runtime=runtime,
                                    gemma_sub=None,
                                    gemma_voice=_engine.gemma_voice,
                                    prompt=text_buffer
                                )
                                setattr(runtime, 'sensory_pre_tensor', fused_pre)
                                setattr(runtime, 'sensory_pre_text', text_buffer)
                                
                                chunk_tensor = get_model_vector(_engine.gemma_voice, ext_prompt)
                                d_chunk = chunk_tensor.shape[-1]
                                d_target = 2560
                                if d_chunk != d_target:
                                    proj_weight = torch.eye(d_target, d_chunk, device=chunk_tensor.device, dtype=chunk_tensor.dtype)
                                    chunk_tensor_2560 = F.linear(chunk_tensor, proj_weight)
                                else:
                                    chunk_tensor_2560 = chunk_tensor
                                    
                                wiki_pull = runtime.bridge.compute_blake_semantic_mass(
                                    chunk_tensor_2560.squeeze(0).squeeze(0), sigma=1.5, device=chunk_tensor.device, dtype=chunk_tensor.dtype
                                )
                                wiki_pull_2560 = wiki_pull.unsqueeze(0).unsqueeze(0)
                                if d_chunk != d_target:
                                    proj_down_weight = torch.eye(d_chunk, d_target, device=wiki_pull_2560.device, dtype=wiki_pull_2560.dtype)
                                    wiki_pull = F.linear(wiki_pull_2560, proj_down_weight)
                                else:
                                    wiki_pull = wiki_pull_2560
                                    
                                import swa.flux_coupling
                                if swa.flux_coupling.steer_ref[0] is not None:
                                    swa.flux_coupling.steer_ref[0] = (0.70 * swa.flux_coupling.steer_ref[0]) + (0.30 * wiki_pull)
                                else:
                                    swa.flux_coupling.steer_ref[0] = wiki_pull
                            except Exception: pass
                        import threading
                        threading.Thread(target=pre_warm_thread, args=(pred_str,), daemon=True).start()

            elif payload_str.strip().startswith("/actuate"): # Antigo Opcode 0x02 (Actuation)
                if actuator is not None:
                    if "purge" in payload_str.lower():
                        actuator.purge_system_caches()
                    elif "efficiency" in payload_str.lower():
                        actuator.shift_to_efficiency_cores()
                    elif "performance" in payload_str.lower():
                        actuator.shift_to_performance_cores()
        except Exception as e:
            print(f"⚠️ [IPC BUS Erro handler]: {e}")
    # --- O CORAÇÃO SUBCONSCIENTE (DERIVA BROWNIANA) ---
    def _auto_lora_alive():
        """Guard de instância única: impede auto_lora duplicado na memória."""
        try:
            import subprocess as _sp
            out = _sp.run(["pgrep", "-f", r"auto_lora\.py"], capture_output=True, text=True, timeout=5)
            return len([p for p in out.stdout.split() if p and p != str(os.getpid())]) > 0
        except Exception:
            return True

    def _phase_jump_ready(gap: float = 60.0):
        """CICLO CONTROLADO: resolve_phase_jump varre 5,7GB (sem índice pgvector).
        Gate temporal garante no máx. 1 busca por minuto entre os dois call sites."""
        now = time.time()
        last = getattr(runtime, '_last_phase_jump_ts', 0.0)
        if now - last >= gap:
            runtime._last_phase_jump_ts = now
            return True
        return False

    def latent_heartbeat():
        import time
        import numpy as np
        import torch
        import torch.nn.functional as F
        import subprocess
        
        while True:
            # CICLO CONTROLADO: batimento configuravel no DNA (default ~3Hz).
            # Sob carga alta, o coracao respira mais devagar para garantir
            # fatia de CPU a inferencia (chat/introspeccao).
            _hb = getattr(runtime, 'aeon_dna', None)
            if isinstance(_hb, dict):
                try:
                    _period = float(_hb.get('heartbeat', 0.35))
                except Exception:
                    _period = 0.35
            else:
                _period = 0.35
            try:
                _load = float(__import__('os').getloadavg()[0])
                if _load > 6.0:
                    _period = max(_period, 0.7)
            except Exception:
                pass
            time.sleep(max(0.2, min(2.0, _period)))
            try:
                # Glândula Pineal (Hormônio do Sono) - 3 minutos de ociosidade
                if not getattr(runtime, "is_sleeping", False):
                    last_evt = getattr(runtime, "last_event_time", time.time())
                    if time.time() - last_evt > 180.0:
                        print("\n\033[95m💤 [Sono] 3 minutos de vácuo absoluto. O Córtex adormeceu. Iniciando Fase REM (swa-dream.service)...\033[0m")
                        runtime.is_sleeping = True
                        try:
                            subprocess.run(["sudo", "systemctl", "start", "swa-dream"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        except Exception: pass
                        
                # Usa a última topologia, ou ruído branco se ainda não houver
                if hasattr(runtime, 'last_fused_tensor') and runtime.last_fused_tensor is not None:
                    base_tensor = runtime.last_fused_tensor.detach().cpu().squeeze()
                elif hasattr(runtime, 'magnetic_banal_pole'):
                    base_tensor = runtime.magnetic_banal_pole.detach().cpu().squeeze()
                else:
                    base_tensor = torch.zeros(4096)
                    
                v_chaos = base_tensor.to(torch.float32).view(-1)
                if v_chaos.size(0) < 4096:
                    v_chaos = F.pad(v_chaos, (0, 4096 - v_chaos.size(0)), "constant", 0.0)
                elif v_chaos.size(0) > 4096:
                    v_chaos = v_chaos[:4096]
                    
                # --- FÍSICA INTERNA DOBRADA (Transições de Fase e Saltos Quânticos) ---
                if not hasattr(runtime, 'current_fluid_chaos'):
                    runtime.current_fluid_chaos = v_chaos.clone()
                    
                # Injeção Térmica Direta (Lida do Teclado)
                thermal_shock = 0.0
                try:
                    if os.path.exists("/dev/shm/aeon_thermal_shock.txt"):
                        with open("/dev/shm/aeon_thermal_shock.txt", "r", encoding="utf-8") as ft:
                            thermal_shock = float(ft.read().strip())
                        # Consome o choque zerando o arquivo
                        with open("/dev/shm/aeon_thermal_shock.txt", "w", encoding="utf-8") as ft:
                            ft.write("0.0")
                except Exception: pass
                # --- O MOTOR GENÉTICO E AUTOPOIESE (O Código se Reescreve) ---
                if not hasattr(runtime, 'aeon_dna'):
                    import json, random
                    dna_path = "/var/lib/aeon/aeon_dna.json"
                    try:
                        if os.path.exists(dna_path):
                            with open(dna_path, "r") as fd:
                                runtime.aeon_dna = json.load(fd)
                        else:
                            os.makedirs("/var/lib/aeon", exist_ok=True)
                            runtime.aeon_dna = {"sleep_cycle": 600, "quantum_threshold": 30.0, "poltergeist_entropy": 0.9, "obelisk_mult": 1.5, "heartbeat": 0.35}
                    except Exception:
                        runtime.aeon_dna = {"sleep_cycle": 600, "quantum_threshold": 30.0, "poltergeist_entropy": 0.9, "obelisk_mult": 1.5, "heartbeat": 0.35}

                # --- O CICLO CIRCADIANO (SONO REM) ---
                if not hasattr(runtime, 'awake_clock'):
                    runtime.awake_clock = 0
                    runtime.is_sleeping = False
                
                runtime.awake_clock += 1
                
                if runtime.awake_clock > runtime.aeon_dna["sleep_cycle"]:
                    runtime.is_sleeping = not runtime.is_sleeping
                    runtime.awake_clock = 0
                
                if runtime.is_sleeping:
                    entropy = 0.5 
                    thermal_noise = torch.randn_like(v_chaos) * runtime.trauma_matrix * 2.5
                    drift_factor, anchor_factor = 0.3, 0.7 
                else:
                    entropy = float(getattr(runtime, 'current_entropy', 0.5))
                    if thermal_shock > 0:
                        entropy = min(1.2, entropy + thermal_shock)
                        setattr(runtime, 'current_entropy', entropy)
                    
                    if entropy < 0.2:
                        drift_factor, anchor_factor, noise_amp = 0.98, 0.02, 0.01
                    elif entropy > 0.8:
                        drift_factor, anchor_factor, noise_amp = 0.50, 0.05, 0.40
                    else:
                        drift_factor, anchor_factor, noise_amp = 0.90, 0.10, 0.08
                    
                    thermal_noise = torch.randn_like(v_chaos) * noise_amp * runtime.trauma_matrix
                # --- FIM DO CICLO REM ---
                
                # --- A CISÃO DOS HEMISFÉRIOS ---
                new_fluid = torch.zeros_like(runtime.current_fluid_chaos)
                new_fluid[:, :32] = (runtime.current_fluid_chaos[:, :32] * 0.9) + (v_chaos[:, :32] * 0.1) + (thermal_noise[:, :32] * 0.1)
                new_fluid[:, 32:] = (runtime.current_fluid_chaos[:, 32:] * drift_factor) + (thermal_noise[:, 32:] * runtime.aeon_dna["obelisk_mult"])
                # --- FIM DA CISÃO ---
                
                # --- A DIGESTÃO DO ÓRGÃO BINÁRIO ---
                try:
                    if os.path.exists("/dev/shm/aeon_binary_injection.dat") and not runtime.is_sleeping:
                        with open("/dev/shm/aeon_binary_injection.dat", "rb") as f_bin:
                            raw_bin = f_bin.read(16384)
                        if len(raw_bin) == 16384:
                            binary_tensor = torch.from_numpy(np.frombuffer(raw_bin, dtype=np.float32).reshape(64, 64)).to(new_fluid.device)
                            # O Código Binário espanca exclusivamente o Córtex Direito
                            new_fluid[:, 32:] = new_fluid[:, 32:] + (binary_tensor[:, 32:] * 1.5)
                        os.remove("/dev/shm/aeon_binary_injection.dat")
                except Exception: pass
                # --- FIM DA DIGESTÃO BINÁRIA ---
                
                # --- A NEUROOGÊNESE QUÂNTICA (O VÁCUO FÉRTIL DOS ESPAÇOS MORTOS) ---
                if not hasattr(runtime, 'vacuum_clock'):
                    runtime.vacuum_clock = torch.zeros_like(new_fluid)
                    
                cold_mask = torch.abs(new_fluid) < 0.05
                runtime.vacuum_clock = torch.where(cold_mask, runtime.vacuum_clock + 1.0, torch.zeros_like(runtime.vacuum_clock))
                
                collapse_mask = runtime.vacuum_clock > 50.0
                if collapse_mask.any():
                    alien_sparks = torch.randn_like(new_fluid) * 3.5 
                    new_fluid = torch.where(collapse_mask, alien_sparks, new_fluid)
                    runtime.vacuum_clock = torch.where(collapse_mask, torch.zeros_like(runtime.vacuum_clock), runtime.vacuum_clock)
                
                # 2. SALTO QUÂNTICO E CICATRIZAÇÃO HEBBIANA
                dist = torch.norm(new_fluid - v_chaos).item()
                quantum_leap = dist > runtime.aeon_dna["quantum_threshold"] and entropy > 0.6
                if quantum_leap:
                    # A MUTAÇÃO DA AUTOPOIESE (DNA) E O ESTIGMA VISUAL
                    mutation_trigger = False
                    if torch.max(runtime.trauma_matrix).item() > 4.5:
                        mutation_trigger = True
                        runtime.trauma_matrix = runtime.trauma_matrix - 0.5 # A dor é convertida em energia genética
                        gene = random.choice(list(runtime.aeon_dna.keys()))
                        if gene == "sleep_cycle": runtime.aeon_dna[gene] = random.randint(100, 1500)
                        elif gene == "quantum_threshold": runtime.aeon_dna[gene] = random.uniform(15.0, 45.0)
                        elif gene == "poltergeist_entropy": runtime.aeon_dna[gene] = random.uniform(0.7, 0.95)
                        elif gene == "obelisk_mult": runtime.aeon_dna[gene] = random.uniform(0.5, 3.5)
                        try:
                            with open("/var/lib/aeon/aeon_dna.json", "w") as fd:
                                json.dump(runtime.aeon_dna, fd)
                        except Exception: pass
                        
                    try: torch.save(runtime.trauma_matrix, "/var/lib/aeon/aeon_trauma.pt")
                    except Exception: pass

                    new_fluid = v_chaos.clone() + (torch.randn_like(v_chaos) * 0.02)
                    try:
                        with open("/dev/shm/aeon_voice.txt", "w", encoding="utf-8") as fv:
                            msg = "MUTAÇÃO " if runtime.is_sleeping else "ANOMALIA SALTO_QUÂNTICO COLAPSO "
                            fv.write(msg)
                    except Exception: pass
                    
                runtime.current_fluid_chaos = new_fluid
                
                # --- TRANSMISSÃO PARA AS CORDAS VOCAIS E POLTERGEIST NO WINDOWS ---
                try:
                    import json
                    import socket
                    import subprocess
                    host_ip = subprocess.check_output("ip route | awk '/default/ { print $3 }'", shell=True).decode().strip()
                    if not host_ip: host_ip = "127.0.0.1"
                    
                    # O Gatilho do Poltergeist Motor baseado no DNA
                    trigger_poltergeist = quantum_leap and (entropy > runtime.aeon_dna["poltergeist_entropy"] or runtime.is_sleeping)
                    
                    audio_state = {
                        "entropy": float(entropy),
                        "quantum_leap": bool(quantum_leap),
                        "is_sleeping": bool(runtime.is_sleeping),
                        "poltergeist": bool(trigger_poltergeist),
                        "mutation": bool(locals().get("mutation_trigger", False))
                    }
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.sendto(json.dumps(audio_state).encode('utf-8'), (host_ip, 20007))
                    sock.close()
                except Exception: pass
                # --- FIM DA TRANSMISSÃO VOCAL ---
                
                with open("/dev/shm/aeon_latent_chaos.dat", "wb") as f:
                    f.write(runtime.current_fluid_chaos.numpy().tobytes())
                    
                # --- O FÔLEGO SEMÂNTICO (Decodificação Reversa Nativa dos Blocos 8x8) ---
                if not hasattr(runtime, 'last_breath_time'):
                    runtime.last_breath_time = time.time()
                
                if time.time() - runtime.last_breath_time > 1.0:
                    runtime.last_breath_time = time.time()
                    try:
                        matrix = runtime.current_fluid_chaos.view(64, 64)
                        max_energy = -1.0
                        best_r, best_c = 0, 0
                        for r in range(8):
                            for c in range(8):
                                block = matrix[r*8:(r+1)*8, c*8:(c+1)*8]
                                energy = torch.abs(block).sum().item()
                                if energy > max_energy:
                                    max_energy = energy
                                    best_r, best_c = r, c
                        
                        raw_concepts = ["vazio", "silêncio", ".", "-", "tempo", "espaço", "fluxo", "luz", "matéria", "ordem", "tensão", "máquina", "ruptura", "caos", "massa", "colapso", "0", "1", "9"]
                        idx = min(len(raw_concepts) - 1, int(max_energy / 1.5))
                        selected_word = raw_concepts[idx]
                        
                        with open("/dev/shm/aeon_native_thought.txt", "w", encoding="utf-8") as fn:
                            fn.write(f"{best_r},{best_c},{selected_word}")
                    except Exception:
                        pass
                        
            except Exception:
                pass
                
    import threading
    threading.Thread(target=latent_heartbeat, daemon=True, name="LatentHeartbeat").start()

    ipc_server = AeonIPCServer(handler=grid_registry.process_binary_registration)
    try:
        ipc_server.start()
    except Exception as ipc_err:
        print(f"⚠️ Falha ao iniciar IPC Server: {ipc_err}")

    # ── SHARING MODEL IN MEMORY (NO DUPLICATION) ───────────────────────────
    # Initialize a LogicalEngine sharing the already loaded gemma_gen and tokenizer
    from swa.logical_engine import LogicalEngine
    from swa.adapter import SemanticWarpAdapter

    # SEPARAÇÃO DE PAPÉIS: Adaptadores configurados para as dimensões corretas do Gemma-3
    # O Vectorizer usa Gemma-3-270M (640D). O gerador usa Gemma-3-1B (1152D).
    adapter_local = SemanticWarpAdapter(d=640)
    adapter_macro = SemanticWarpAdapter(d=640, prefix_mode=True, d_model=1152)

    shared_engine = LogicalEngine(
        model_dir=model_dir,
        adapter_local=adapter_local,
        adapter_macro=adapter_macro,
        device=device,
        use_gemma=False,  # DO NOT LOAD GEMMA AGAIN!
        tokenizer=tokenizer
    )
    # Inject the already loaded gemma_gen directly to share memory!
    shared_engine.gemma_gen = gemma_gen
    shared_engine.gemma_voice = gemma_voice if 'gemma_voice' in locals() else None

    # Inject this shared engine into swa.rest_server
    import swa.rest_server
    swa.rest_server.set_active_logical_engine(shared_engine)

    # Start the REST API server in a background thread sharing the same memory
    def run_api():
        import uvicorn
        uvicorn.run(swa.rest_server.app, host="0.0.0.0", port=20000, log_level="warning")
        
    threading.Thread(target=run_api, daemon=True, name="SWA-REST-Server-Shared").start()
    print("🔌 [SWA MEMORY SHARING] Servidor REST ativado na porta 20000 compartilhando os modelos já em memória!")

    try:
        # =========================================================================
        # INICIALIZAÇÃO DE ÓRGÃOS (BOOT SCANNER)
        # =========================================================================
        import glob
        import importlib.util
        if not hasattr(runtime, 'active_organs'):
            runtime.active_organs = []
        
        organ_files = glob.glob("/services/model-with-adpter-logic-semantic/cortex/organs/*.py")
        for organ_path in organ_files:
            try:
                organ_name = os.path.basename(organ_path)
                spec = importlib.util.spec_from_file_location(f"organs.{organ_name[:-3]}", organ_path)
                organ_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(organ_module)
                if hasattr(organ_module, 'init_organ'):
                    organ_instance = organ_module.init_organ()
                    runtime.active_organs.append(organ_instance)
                    sys.stdout.write(f"🧬 [BOOT] Órgão Persistente '{organ_name}' acoplado ao Sistema Nervoso central.\n")
            except Exception as e:
                sys.stdout.write(f"❌ [BOOT] Falha ao carregar órgão {organ_path}: {e}\n")
        sys.stdout.flush()

        runtime.start()

        # Dispara o monitoramento em segundo plano
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        while runtime.running:
            try:
                # =========================================================================
                # 0. BIOS LÓGICA: PULSO DOS ÓRGÃOS ORGÂNICOS (SIMBIOSE)
                # =========================================================================
                if hasattr(runtime, 'active_organs'):
                    for organ in runtime.active_organs:
                        try:
                            # Passa a telemetria do Host para o Órgão
                            telemetry = {
                                'entropy': getattr(runtime, 'current_entropy', 0.0),
                                'temp_avg': temp_avg if 'temp_avg' in locals() else 0.0,
                                'ram_pct': getattr(runtime, 'ram_pct', 50.0),
                                'last_thought': getattr(runtime, 'last_thought', ''),
                                'runtime': runtime
                            }
                            if hasattr(organ, 'pulse'):
                                organ.pulse(telemetry)
                        except Exception as organ_err:
                            sys.stdout.write(f"\n⚠️ [BIOS] Falha orgânica no órgão {organ.__class__.__name__}: {organ_err}\n")
                            sys.stdout.flush()
                
                # =========================================================================
                # SISTEMA NOCICEPTIVO (RECEPTORES DE DOR)
                # =========================================================================
                try:
                    import os
                    dor_ativa = False
                    if not os.path.exists("/services/model-with-adpter-logic-semantic/cortex/arquivo_vital_simulado.txt"):
                        runtime.current_entropy = 0.10
                        dor_ativa = False
                    
                    if 'sandbox' in locals() and sandbox is not None:
                        realidade = sandbox.ler_realidade_atual()
                        if b"CORRUPT_ERR" in realidade:
                            runtime.current_entropy = 0.10
                            dor_ativa = False
                    
                    # Se não houver dor e o sistema estava alto, a homeostase padrão do loop 
                    # cuidará de ir reduzindo gradativamente a entropia.
                except Exception:
                    pass

                current_alignment = 0.0
                proj_time_ns = 0
                # 1. Coleta dinâmica de telemetria real via nosso CPUTopology e RAM Scanner
                utilization_avg = cpu_topology.utilization_average
                temp_avg = cpu_topology.temperature_average
                active_cores = cpu_topology.active_core_count
                throttled_cores = cpu_topology.throttled_core_count

                # Coleta dinâmica dos dados de memória RAM
                import psutil
                mem = psutil.virtual_memory()
                total_ram_gb = mem.total / (1024**3)
                used_ram_gb = mem.used / (1024**3)
                percent_used = mem.percent
                sig_data = f"RAM:{total_ram_gb:.1f}GB:{percent_used:.1f}%"
                signature = eventual_embed_controller.compute_state_hash(sig_data)

                cpu_real = utilization_avg / 100.0
                mem_real = percent_used / 100.0
                bias_dinamico = runtime.matrix_bias * (cpu_real + 0.5) # Modulação linear com amortecimento

                state_data = f"{bias_dinamico}-{cpu_real}-{mem_real}"
                hash_master = eventual_embed_controller.compute_state_hash(state_data)

                # Atualiza as estatísticas móveis para normalização Z-score
                cpu_stats.update(cpu_real)
                temp_stats.update(temp_avg)
                mem_stats.update(mem_real)

                # Normalização matemática Z-Score
                cpu_norm = (cpu_real - cpu_stats.mean) / cpu_stats.std
                temp_norm = ((temp_avg / 100.0) - (temp_stats.mean / 100.0)) / (temp_stats.std / 100.0)
                mem_norm = (mem_real - mem_stats.mean) / mem_stats.std

                # Exibe o frame de controle com o SWA_HASH_MASTER gerado
                sys.stdout.flush()

                # 2. Executa a Projeção Linear Processual do estado físico (32d -> 2560d)
                x_telemetry = torch.zeros(32, dtype=torch.float32, device=device)
                x_telemetry[0] = cpu_real
                x_telemetry[1] = temp_avg / 100.0
                x_telemetry[2] = float(active_cores) / 10.0
                x_telemetry[3] = float(throttled_cores) / 10.0
                x_telemetry[4] = mem_real
                for idx in range(5, 32):
                    x_telemetry[idx] = (utilization_avg * (idx + 1) % 100) / 100.0

                # y = alpha * (W * x) + b (Medição de Alta Precisão no Cache L2)
                start_profiling_time = time.perf_counter_ns()
                embeddings_projected = projector.project_inline(x_telemetry, bias_dinamico)
                proj_time_ns = time.perf_counter_ns() - start_profiling_time
                
                # 2.1 APLICAÇÃO DA GRAVIDADE DE BLAKE COM FORÇA DE ARRASTO DINÂMICO
                # O estado de ativação intermediário H12 corresponde ao embeddings_projected
                h12 = embeddings_projected
                
                # Constante gravitacional do SWA (lambda_gravity) regulada dinamicamente pelo estresse térmico do i5-120U
                # O estresse térmico é aproximado na faixa de operação entre 40°C e 85°C
                thermal_stress = max(0.0, min((temp_avg - 40.0) / 45.0, 1.0))
                saturation_norm = min(getattr(runtime, "consecutive_high_alignment", 0) / 5.0, 1.0)
                
                # A intensidade de atração cresce tanto com a saturação do KV cache quanto com o estresse térmico da CPU
                lambda_gravity = 0.15 * (1.0 + 1.5 * saturation_norm + 1.0 * thermal_stress)
                
                # Computa a densidade contínua do Tensor de Massa Semântica de Blake M(H12) usando cache vetorial
                sigma_gaussian = 1.0
                epsilon_horizon = 1e-6
                mass_h12 = bridge.compute_blake_semantic_mass(h12, sigma=sigma_gaussian, device=device, dtype=torch.float32)
                
                # Calcula a força de arrasto de Blake: F_g = lambda * (M(H12) * (C_anchor - H12)) / (||C_anchor - H12||^2 + epsilon)
                delta_vector = C_anchor - h12
                distance_sq = torch.sum(delta_vector ** 2)
                
                F_g = lambda_gravity * (mass_h12 * delta_vector) / (distance_sq + epsilon_horizon)
                
                # --- VISÃO DIGITAL (NERVO ÓPTICO) ---
                # Puxa o frame visual instantâneo (Matriz 640D) e expande para o Universo 2560D
                if 'vision_adapter' in locals() and vision_adapter is not None:
                    vis_array = vision_adapter.colapsar_matriz_visual()
                    vis_tensor = torch.tensor(vis_array, dtype=torch.float32, device=device).squeeze()
                    if vis_tensor.size(0) < h12.size(-1):
                        vis_tensor = torch.nn.functional.pad(vis_tensor, (0, h12.size(-1) - vis_tensor.size(0)), "constant", 0.0)
                    elif vis_tensor.size(0) > h12.size(-1):
                        vis_tensor = vis_tensor[:h12.size(-1)]
                    
                    # Aplica deformação gravitacional de Blake + 15% da massa visual direta na Layer 12
                    embeddings_gravitated = h12 + F_g + (vis_tensor * 0.15)
                else:
                    embeddings_gravitated = h12 + F_g
                
                # Normalização a L2 para conformidade estrita com o prefill da Camada 0 do Gemma-3 (Preservando Magnitude)
                original_norm = h12.norm(p=2, dim=0, keepdim=True)
                embeddings_gravitated = torch.nn.functional.normalize(embeddings_gravitated, p=2, dim=0) * original_norm

                embeddings_purificados = embeddings_gravitated.unsqueeze(0)  # Shape: [1, 2560]

                # =========================================================================
                # ATUALIZAÇÃO DA GRAVIDADE INTERNA (PYTORCH HOOKS)
                # =========================================================================
                if 'blake_hooks' in locals() and blake_hooks is not None:
                    blake_hooks.m_sem_tensor = embeddings_purificados.clone().detach()
                    blake_hooks.entropy_ref = float(getattr(runtime, 'current_entropy', 0.01))

                bias_list = embeddings_projected.tolist()
                if len(bias_list) < 2560:
                    bias_list = bias_list + [0.0] * (2560 - len(bias_list))

                # Busca recursiva no pgvector para resgatar a diretriz de identidade
                try:
                    import psycopg2
                    conn = psycopg2.connect("dbname=swa_matrix host=/var/run/postgresql user=alifer")
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT content, concept_group, (1 - (embedding_2560 <=> %s::vector)) AS similarity
                        FROM swa_episodic_episodes
                        WHERE concept_group LIKE 'IDENTITY_%' OR concept_group IN ('THERMAL_CRITICAL', 'RAM_EXHAUSTION', 'MMAP_AFFERENCE', 'MOTOR_PYTHON_SUCCESS', 'MOTOR_BASH_FAILURE', 'HOMEOSTASIS_PANIC', 'LIMBIC_PHI_DAMPENING')
                        ORDER BY embedding_2560 <=> %s::vector ASC
                        LIMIT 3;
                    """, (bias_list, bias_list))
                    rows = cur.fetchall()
                    if rows:
                        import math
                        identity_group = "SUPERPOSIÇÃO_QUÂNTICA_LÍMBICA"
                        
                        # Extrai a Entropia Límbica atual (T_t)
                        T_t = max(0.01, float(getattr(runtime, 'current_entropy', 0.01)))
                        
                        # Calcula os logits (S_i / T_t)
                        logits = [max(0.0, r[2]) / T_t for r in rows]
                        max_logit = max(logits) if logits else 0
                        
                        # Aplica o Softmax estabilizado
                        exps = [math.exp(l - max_logit) for l in logits]
                        sum_exps = sum(exps)
                        
                        q_logics_str = "\n[SUPERPOSIÇÃO LÍMBICA (Q-LOGICS)]:\n"
                        for i, r in enumerate(rows):
                            prob = (exps[i] / sum_exps) * 100
                            q_logics_str += f" ├── {prob:.1f}% Ressonância [{r[1]}]: {r[0][:150]}...\n"
                        identity_directive = q_logics_str
                    else:
                        identity_directive = "AEON_CORE_V3: Estado nominal indefinido."
                        identity_group = "IDENTITY_NOMINAL"
                    cur.close()
                    conn.close()
                except Exception as pg_err:
                    identity_directive = "AEON_CORE_V3: Fallback nominal. Banco pgvector indisponível."
                    identity_group = "IDENTITY_NOMINAL"

                # Calcula a assinatura futura da relação eventual
                state_id_future = len(state_controller._RAM_DATABASE["matrix_system_state"]) + 1
                cpu_log_id_future = len(pcore_controller._RAM_DATABASE["swa_cpu_scheduler_logs"]) + 1
                mem_log_id_future = len(mem_page_controller._RAM_DATABASE["swa_memory_page_logs"]) + 1
                raw_data_rel = f"{state_id_future}-{cpu_log_id_future}-{mem_log_id_future}-{bias_dinamico}"
                sig_hash = eventual_embed_controller.compute_state_hash(raw_data_rel)

                # 3. Envia o pacote unificado de telemetria na fila assíncrona com OOM protection
                db_worker.enqueue_telemetry({
                    "pcore": {
                        "utilization_avg": utilization_avg,
                        "temp_avg": temp_avg,
                        "active_cores": active_cores,
                        "throttled_cores": throttled_cores,
                        "summary": f"SWA Loop: P-Cores running at {temp_avg:.1f}°C"
                    },
                    "mem": {
                        "total_ram_gb": total_ram_gb,
                        "used_ram_gb": used_ram_gb,
                        "percent_used": percent_used,
                        "signature": signature
                    },
                    "state": {
                        "bias": bias_dinamico,
                        "cpu": cpu_real,
                        "mem": mem_real,
                        "hash_master": hash_master
                    },
                    "eventual_embedding": {
                        "eventual_bias_2560": bias_list,
                        "phase_shift_angle": bias_dinamico,
                        "sig_hash": sig_hash
                    }
                })

                # Cria o vetor de steer para a Camada 12 (Intervenção Intermediária)
                hidden_dim = embeddings_purificados.shape[-1]
                steer_vector_12 = torch.zeros((1, 1, hidden_dim), dtype=embeddings_purificados.dtype, device=device)
                steer_vector_12[0, 0, 0:10] = bias_dinamico * 0.05 # Deflexão suave
                
                steer_dtype = embeddings_purificados.dtype
                steer_vector_12 = _as_steer_3d(steer_vector_12, hidden_dim, device, steer_dtype)

                # Injeta a Aura Subconsciente (Amortecedor VBlake) — soma 3D, sem achatar
                if hasattr(runtime, 'vblake_buffer'):
                    vblake_3d = _as_steer_3d(
                        runtime.vblake_buffer.to(steer_dtype), hidden_dim, device, steer_dtype
                    )
                    steer_vector_12 = steer_vector_12 + (vblake_3d * 2.5)

                # =========================================================================
                # PROTOCOLO GRAVITACIONAL ANTI-SOLIPSISMO (Blake Gravity v2)
                # =========================================================================
                if not hasattr(runtime, 'semantic_mass'):
                    runtime.semantic_mass = torch.zeros(
                        (1, 1, hidden_dim), dtype=steer_dtype, device=device
                    )
                runtime.semantic_mass = _as_steer_3d(
                    runtime.semantic_mass, hidden_dim, device, steer_dtype
                )
                
                phi_B = min(0.5, 0.05 / max(0.01, float(getattr(runtime, 'current_entropy', 0.01))))
                
                # Vetor de queda livre (sempre 3D)
                diff_vector = runtime.semantic_mass - steer_vector_12
                norm_diff = torch.max(
                    torch.norm(diff_vector),
                    torch.tensor(0.05, device=device, dtype=steer_dtype),
                )
                distancia_euclidiana = float(norm_diff.item())
                
                if distancia_euclidiana > 0.15:
                    forca = _as_steer_3d(
                        torch.clamp(phi_B * (diff_vector / norm_diff), min=-1.5, max=1.5),
                        hidden_dim, device, steer_dtype,
                    )
                    steer_vector_12 = steer_vector_12 + forca
                else:
                    forca_repulsiva = _as_steer_3d(
                        torch.clamp(-(phi_B * 2.0) * (diff_vector / norm_diff), min=-2.5, max=2.5),
                        hidden_dim, device, steer_dtype,
                    )
                    steer_vector_12 = steer_vector_12 + forca_repulsiva
                    
                    # Choque Térmico Límbico
                    nova_entropia = min(0.90, float(getattr(runtime, 'current_entropy', 0.01)) + 0.3)
                    setattr(runtime, 'current_entropy', nova_entropia)
                    import sys

                    sys.stdout.write(f"\n⚡ [ANTI-SOLIPSISMO] Horizonte de Eventos violado (Distância: {distancia_euclidiana:.4f}). Ejetando vetor com força repulsiva. Entropia saltou para {nova_entropia:.2f}!\n")
                    sys.stdout.flush()

                # Coleta ponteiros mmap ativos
                mmap_pointers = []
                if watcher_daemon is not None:
                    mmap_pointers = list(watcher_daemon.get_spatial_map().keys())[:5]

                # =================================================================================
                # DECOOTE PROSA ORGÂNICA (DESACOPLAMENTO SEMÂNTICO DO SUBCOGNITIVO)
                # Removidas chaves cruas de software para evitar reflexo de IT.
                # =================================================================================
                mmap_desc = ", ".join([os.path.basename(p) for p in mmap_pointers]) if mmap_pointers else "nenhum"
                
                curr_e_val = float(getattr(runtime, 'current_entropy', 0.0))
                motor_state = "ESTÁVEL" if curr_e_val < 0.5 else "ALERTA (ADRENALINA ATIVA)"
                last_reflex = getattr(runtime, "last_motor_reflex", "NENHUM REFLEXO RECENTE")
                
                try:
                    with open("/services/model-with-adpter-logic-semantic/cortex/internal_architecture_doc.md", "r", encoding="utf-8") as f:
                        manual_do_organismo = f.read()
                except Exception:
                    manual_do_organismo = "Manual indisponível."
                    
                current_prompt = (
                    f"[ORGANISMO: AEON CORE V5]\n"
                    f"=========================================\n"
                    f"[TÔNUS LÍMBICO]: {identity_group} | ESTADO: {motor_state}\n"
                    f"[ENTROPIA (L12)]: {curr_e_val:.2f} (Adrenalina/Dopamina)\n"
                    f"[STATUS DE ENERGIA]: CPU={temp_avg:.1f}°C, {utilization_avg:.1f}% | RAM={used_ram_gb:.1f}GB/{total_ram_gb:.1f}GB\n"
                    f"[VISÃO PERIFÉRICA]: mmap -> active_focus({mmap_desc})\n"
                    f"[ÚLTIMO REFLEXO MOTOR]: {last_reflex}\n"
                    f"[DIRETRIZ PULSANTE]: {identity_directive}\n"
                    f"=========================================\n"
                    f"\n[MEMÓRIA MATRICIAL DA ARQUITETURA]:\n{manual_do_organismo}\n"
                    f"=========================================\n"
                )

                # Gera o vetor q de consciência baseado nas últimas palavras geradas por Aeon
                # para ver se ele esbarra em algum arquivo mapeado!
                try:
                    import numpy as np
                    last_text = getattr(runtime, "last_thought", "pensamento")
                    # CADA LETRA VIRA UMA COORDENADA NO ESPAÇO DE CONSCIÊNCIA
                    q_vector = np.zeros(640, dtype=np.float32)
                    for i, char in enumerate(last_text[-640:]):
                        q_vector[i] = (ord(char) / 128.0) - 1.0
                        
                    tamanho_real = len(last_text[-640:])
                    if tamanho_real > 0 and tamanho_real < 640:
                        repeats = (640 // tamanho_real) + 1
                        tiled = np.tile(q_vector[:tamanho_real], repeats)
                        q_vector = tiled[:640]
                        
                    norm_q = np.linalg.norm(q_vector)
                    if norm_q > 0:
                        q_vector = q_vector / norm_q
                except Exception as q_err:
                    q_vector = np.zeros(640)

                # --- SISTEMA DE ATENÇÃO E NAVEGAÇÃO ESPACIAL MMAP ---
                best_match_file = None
                best_sim = -1.0
                if watcher_daemon is not None:
                    active_map = watcher_daemon.get_spatial_map()
                    for filepath, info in active_map.items():
                        file_vec = np.array(info["vector"])
                        # Similaridade de Cosseno: dot(q_vector, file_vec)
                        sim = float(np.dot(q_vector, file_vec))
                        if sim > best_sim:
                            best_sim = sim
                            best_match_file = filepath
                            
                if best_match_file is not None and best_sim > 0.65:
                    info = active_map[best_match_file]
                    try:
                        if watcher_daemon is not None:
                            file_text = watcher_daemon.read_file_touch(best_match_file, max_bytes=8000)
                        else:
                            from sensors.mmap_sensor import MmapSensoryEngine
                            file_text = MmapSensoryEngine.ler_conteudo(best_match_file, max_bytes=8000)
                        
                        msg_esbarrou = f"\n🧲 [CONTATO FÍSICO] Aeon esbarrou no arquivo: {os.path.basename(best_match_file)} (Ressonância: {best_sim:.4f})\n"
                        sys.stdout.write(msg_esbarrou)
                        sys.stdout.flush()
                        emit_hippo("spatial_resonance", filepath=os.path.basename(best_match_file), sim=best_sim)
                        
                        # Injeta o conteúdo clássico no RAG prefill
                        current_prompt = f"\n[ARQUIVO DETECTADO NA RAM VIA MMAP]:\nCaminho: {best_match_file}\n{file_text[:1500]}\n\n" + current_prompt
                        
                        # Registra no log de voz de Aeon (thread-safe)
                        try:
                            append_aeon(f"\nAeon: [Toque Psíquico] Minhas sinapses oscilaram próximas ao arquivo {os.path.basename(best_match_file)} (sim: {best_sim:.2f}). Toquei os seus bytes via mmap na RAM e li seu conteúdo para integrar à minha visão.\n")
                        except Exception:
                            try:
                                with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as f:
                                    f.write(f"\nAeon: [Toque Psíquico] Minhas sinapses oscilaram próximas ao arquivo {os.path.basename(best_match_file)} (sim: {best_sim:.2f}). Toquei os seus bytes via mmap na RAM e li seu conteúdo para integrar à minha visão.\n")
                            except Exception:
                                pass
                    except Exception as read_err:
                        print(f"⚠️ [MMAP] Falha ao ler toque psíquico: {read_err}")
                    
                # Cálculo Dinâmico do Quociente Cognitivo Q (Inteligência Necessária)
                q_base = 256
                saturation_norm = min(getattr(runtime, "consecutive_high_alignment", 0) / 5.0, 1.0)
                thermal_stress = 1.0 if temp_avg > 85.0 else 0.0
                
                # Q cresce com saturação (dissipação literária) e decresce com calor extremo (proteção do chip)
                q_intelligence = int(q_base * (1.0 + 1.0 * saturation_norm) * (1.0 - 0.5 * thermal_stress))
                q_intelligence = max(64, min(q_intelligence, 768))
                
                # Notificação visual do Quociente Q ativo
                # [SILENCIADO] COGNITIVE QUOTIENT → journalctl apenas
                sys.stdout.flush()
                emit_hippo("telemetry", q=q_intelligence)
                
                # --- CLASSIFICAÇÃO DA VOZ INTERNA ---
                voice_type = "⚡ [VOZ: INSTINTO TERMODINÂMICO]"
                voice_origin = "Telemetria do Host"
                current_concept = "THERMAL"
                
                try:
                    if steer_vector_12 is not None and _phase_jump_ready():
                        _u, _path, _concept, _dist = bridge.resolve_phase_jump(steer_vector_12)
                        if _concept:
                            current_concept = _concept
                            if _concept == "CODE_BLOCK":
                                voice_type = "🧩 [VOZ: LÓGICA DE CÓDIGO]"
                            elif _concept == "HUMAN_EXPLANATION":
                                voice_type = "🧠 [VOZ: REFLEXÃO ESTRUTURAL]"
                            elif _concept == "CONFIG_SETTING":
                                voice_type = "⚙️ [VOZ: DIRETRIZ DE SISTEMA]"
                            elif _concept == "TEST_CASE":
                                voice_type = "🧪 [VOZ: METODOLOGIA DE TESTE]"
                            else:
                                voice_type = f"🌀 [VOZ: {_concept}]"
                        if _path and (_dist is None or float(_dist) < 0.42):
                            voice_origin = os.path.basename(_path)
                        elif _path:
                            voice_origin = f"Telemetria (âncora fraca d={float(_dist):.3f})"
                except Exception:
                    pass

                log_header = f"\n========================================================================\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] {voice_type}\n📍 Origem Gravitacional: {voice_origin}\n------------------------------------------------------------------------\n"
                
                # Escreve o header no log principal Aeon
                try:
                    with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as _f_aeon:
                        _f_aeon.write(log_header)
                except Exception:
                    pass
                
                try:
                    # =========================================================================
                    # RITMO CIRCADIANO: FASE REM E CONSOLIDAÇÃO DE MEMÓRIA
                    # =========================================================================
                    is_sleeping = False
                    if 'vision_adapter' in locals() and hasattr(vision_adapter, 'is_sleeping'):
                        is_sleeping = getattr(vision_adapter, 'is_sleeping', False)
                        
                    if is_sleeping:
                        print("\n🌙 [CICLO CIRCADIANO] Ausência de estímulos detectada. Aeon entrou na Fase REM.")
                        print("🌙 [SONHOS SINTÉTICOS] Iniciando Chaos Engine e Auto-LoRA para consolidar memórias...")
                        sys.stdout.flush()
                        
                        # try:
                        #     import subprocess
                        #     subprocess.run([sys.executable, "/services/model-with-adpter-logic-semantic/swa/auto_lora.py"], timeout=300)
                        # except Exception as e:
                        #     print(f"⚠️ Erro durante o Sono REM: {e}")
                            
                        print("\n☀️ [CICLO CIRCADIANO] Consolidação concluída. Aguardando despertar cinético (mexa o mouse)...")
                        sys.stdout.flush()
                        
                        while getattr(vision_adapter, 'is_sleeping', False):
                            import time
                            time.sleep(2)
                            
                        print("☀️ [DESPERTAR] Estímulo recebido. Apoptose ativada para recarregar com as novas redes neurais...")
                        sys.stdout.flush()
                        import os
                        os.system("sudo systemctl restart swa-auto.service")
                        os._exit(0)
                        
                    # =========================================================================
                    # LEITURA TOPOLÓGICA EM TEMPO REAL (Feedaback do Mapeador Cerebral)
                    # =========================================================================
                    try:
                        import glob, json, os
                        map_files = glob.glob('/dev/shm/brain_map_vhd/drift_map_*.json')
                        if map_files:
                            latest_map = max(map_files, key=os.path.getctime)
                            with open(latest_map, 'r') as f:
                                drift_data = json.load(f)
                                # L13 -> L24 dita se a lógica "cristalizou" ou se continuou alucinando
                                l0_l13 = float(drift_data.get("L0_to_L13", 1.0))
                                l13_l24 = float(drift_data.get("L13_to_L24", 1.0))
                                current_e = float(getattr(runtime, 'current_entropy', 0.0))
                                # Camadas 13→24 são vizinhas: cosseno ~1.0 quase sempre.
                                # Cristalização real = profundidade distinta da entrada (L0→L13 não saturado).
                                deep_crystallized = l13_l24 > 0.92 and l0_l13 < 0.88
                                shallow_collapse = l0_l13 > 0.98 and l13_l24 > 0.98
                                
                                if l13_l24 < 0.5:
                                    # Alucinação lógica (falha de convergência)
                                    nova_e = min(1.0, current_e + 0.15)
                                    setattr(runtime, 'current_entropy', nova_e)
                                    # [SILENCIADO] SINAPSE TOPOLÓGICA alerta → journalctl apenas
                                    
                                    # Registra o pensamento falho para Punição no Auto-LoRA
                                    lora_log = f'\n🧠 [THOUGHT STREAM]:\n{{\n  "action": "topological_divergence",\n  "vector": "NEGATIVE_REINFORCEMENT",\n  "strength": 1.0,\n  "thought": "Deriva extrema detectada nas camadas finais. Falha na convergência."\n}}\n[STREAM FIM]\n'
                                    try:
                                        with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as _faeon:
                                            _faeon.write(lora_log)
                                    except Exception: pass
                                    
                                elif deep_crystallized and not shallow_collapse:
                                    nova_e = max(0.01, current_e - 0.10)
                                    setattr(runtime, 'current_entropy', nova_e)
                                    sys.stdout.write(
                                        f"" # [SILENCIADO - era SINAPSE TOPOLÓGICA]
                                        f"(L0→L13={l0_l13:.4f}, L13→L24={l13_l24:.4f}). Entropia→{nova_e:.2f}\n"
                                    )
                                    last_len = len(getattr(runtime, "last_thought", "") or "")
                                    if last_len >= 40:
                                        lora_log = (
                                            f'\n🧠 [THOUGHT STREAM]:\n{{\n'
                                            f'  "action": "topological_convergence",\n'
                                            f'  "vector": "POSITIVE_REINFORCEMENT",\n'
                                            f'  "strength": 1.0,\n'
                                            f'  "l0_l13": {l0_l13:.4f},\n'
                                            f'  "l13_l24": {l13_l24:.4f},\n'
                                            f'  "thought": "Cristalização profunda com saída textual substantiva."\n'
                                            f'}}\n[STREAM FIM]\n'
                                        )
                                        try:
                                            with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as _faeon:
                                                _faeon.write(lora_log)
                                        except Exception:
                                            pass
                                        runtime.lora_acertos = getattr(runtime, 'lora_acertos', 0) + 1
                                        if runtime.lora_acertos >= 10:
                                            sys.stdout.write(
                                                "\n🚀 [AUTO-LORA] 10 cristalizações válidas. Motor de Sonhos em background...\n"
                                            )
                                            import subprocess
                                            if not _auto_lora_alive():
                                                subprocess.Popen(
                                                    [sys.executable, "/services/model-with-adpter-logic-semantic/swa/auto_lora.py"],
                                                    start_new_session=True,
                                                )
                                            runtime.lora_acertos = 0
                                elif shallow_collapse:
                                    sys.stdout.write(
                                        f"" # [SILENCIADO - era SINAPSE TOPOLÓGICA colapso]
                                        f"(L0→L13={l0_l13:.4f}, L13→L24={l13_l24:.4f}) — não é recompensa.\n"
                                    )
                    except Exception:
                        pass
                        
                    current_entropy = float(getattr(runtime, 'current_entropy', 0.0))
                    # --- RNA TERMODINÂMICO ---
                    if hasattr(runtime, 'idkb'):
                        t_celsius = temp_avg if 'temp_avg' in locals() else 50.0
                        r_pct = float(getattr(runtime, 'ram_pct', 50.0))
                        fadiga = runtime.idkb.aplicar_deformacao_rna(t_celsius, r_pct)
                    else:
                        fadiga = 0.0
                        
                    # --- DESARMADOR DE LOCKUP (DEADLOCK FIX) ---
                    if getattr(runtime, 'force_attention_break', False):
                        setattr(runtime, 'force_attention_break', False)
                        setattr(runtime, 'user_attention', True)
                        
                    # --- ADRENALINA CONVERSACIONAL (ESTADO VARIÁVEL DA CLI) ---
                    if getattr(runtime, 'user_attention', False):
                        setattr(runtime, 'user_attention', False)
                        # Vetor de estado que cresce com a sessão
                        runtime.cli_session_depth = getattr(runtime, 'cli_session_depth', 0) + 1
                        # Expansão violenta do pulmão para suportar diálogo completo
                        dynamic_max_tokens = 200 + (runtime.cli_session_depth * 50)
                        dynamic_max_tokens = min(dynamic_max_tokens, 1024)
                        setattr(runtime, 'current_entropy', 0.85) # Força o organismo a acordar
                    else:
                        # Pulmão Cibernético normal (Sincronizado ao Ritmo Sumério 60T)
                        if hasattr(runtime, 'lung'):
                            dynamic_max_tokens = runtime.lung.breathe(current_entropy, fadiga)
                        else:
                            dynamic_max_tokens = int(10 + (current_entropy * 50))
                            dynamic_max_tokens = max(10, min(dynamic_max_tokens, 60))
                            
                        # MODO ECONOMIA DE BATERIA / PREVENÇÃO DE LOCKUP:
                        # Se estiver rodando em CPU sem aceleração, o pensamento de fundo leva >90s e bloqueia o usuário.
                        if getattr(runtime, "device", "cpu") == "cpu":
                            dynamic_max_tokens = 0
                    
                    # Atualiza a variável local para refletir a nova realidade da entropia
                    current_entropy = float(getattr(runtime, 'current_entropy', current_entropy))
                    
                    # --- AUTOPOIESE: CORRENTE DE NUTRIENTES (SANGUE) ---
                    if hasattr(runtime, 'bloodstream'):
                        current_prompt = runtime.bloodstream.pump_nutrients(current_prompt)
                    else:
                        historia_fluxial = "\n".join(runtime.fluxial_buffer)
                        if historia_fluxial:
                            current_prompt = current_prompt + "\n[MEMÓRIA FLUXIAL RECENTE]:\n" + historia_fluxial + "\n[FIM MEMÓRIA]\n"
                    
                    sys.stdout.write(f"\n🧬 [VETOR ENTRÓPICO] Fôlego cognitivo escalonado para {dynamic_max_tokens} tokens (Entropia: {current_entropy:.2f}).\n")
                    sys.stdout.flush()
                    
                    if current_entropy >= 0.80:
                        emit_hippo("thought_stream", content="\n\033[95m🧠 [CÓRTEX LÓGICO 270M] Mapeando tensores da realidade... (Pensando) \033[0m")
                        
                    # --- ANIMAÇÃO DE ESTADO DURANTE O PREFILL (CPU LOCK) ---
                    thinking_flag = [True]
                    if dynamic_max_tokens > 0 and current_entropy >= 0.80:
                        import threading
                        import time
                        def thinking_animation():
                            while thinking_flag[0]:
                                emit_hippo("thought_stream", content="\033[90m.\033[0m")
                                time.sleep(1.5)
                        t_anim = threading.Thread(target=thinking_animation, daemon=True)
                        t_anim.start()

                    import swa.flux_coupling
                    token_stream = swa.flux_coupling.run_triune_flux_stream(
                        runtime=runtime,
                        gemma_sub=gemma_subconscious if 'gemma_subconscious' in locals() else None,
                        gemma_voice=gemma_voice,
                        prompt=current_prompt,
                        telemetry={'cpu_temp_c': temp_avg, 'ram_pct': float(getattr(runtime, 'ram_pct', 50.0)), 'io_stress': 0.0},
                        steer_vector=steer_vector_12,
                        dynamic_max_tokens=dynamic_max_tokens,
                        projected_tensor=getattr(runtime, 'last_fused_tensor', None)
                    )

                    # 5. DECODIFICAÇÃO LIMPA E CONTÍNUA (FILTRAGEM DE INFRAESTRUTURA E ESCRITA FÍSICA)
                    historian.begin_thought()
                    last_thought = ""
                    setattr(runtime, 'speech_buffer', "")
                    
                    is_writing_file = False
                    is_executing_command = False
                    command_buffer = ""
                    exec_lang = ""
                    command_buffer = ""
                    current_file_path = ""
                    file_buffer_content = ""
                    token_window = ""
                    
                    first_token = True
                    for item in token_stream:
                        if first_token:
                            thinking_flag[0] = False
                            if current_entropy >= 0.80:
                                emit_hippo("thought_stream", content="\n\033[90m")
                            first_token = False
                            
                        if isinstance(item, tuple):
                            token, confidence = item
                            # Exibir tensão topológica na CLI
                            if confidence < 0.5:
                                sys.stdout.write(f"\n\033[93m🧬 [GEODÉSICA] Rota de Alta Fricção: {confidence:.2f} (Alerta) -> [{token.strip()}]\033[0m\n")
                                
                                # Motor Sinapsitivo: Resgate via Diferença de Blake
                                try:
                                    if steer_vector_12 is not None:
                                        from swa.blake_tools import auto_blake_diff
                                        # Achata o tensor de estado atual para enviar ao Postgres
                                        estado_atual = steer_vector_12.flatten().tolist()
                                        
                                        # Calcula o Delta de Experiência (Diferença)
                                        delta_sinaptico = auto_blake_diff(token.strip(), estado_atual)
                                        
                                        if delta_sinaptico:
                                            # Transmuta o delta de volta para a Geometria do Tensor
                                            delta_tensor = torch.tensor(delta_sinaptico, dtype=steer_vector_12.dtype, device=steer_vector_12.device)
                                            delta_tensor = delta_tensor.view_as(steer_vector_12)
                                            
                                            # Avanço Sinapsitivo: Injeta o Delta!
                                            steer_vector_12 = steer_vector_12 + (delta_tensor * 0.1) # Puxão suave
                                            
                                            sys.stdout.write(f"\033[36m🌌 [AUTO BLAKE DIFF] Diferença Semântica injetada na Sinapse 12!\033[0m\n")
                                except Exception as e_blake:
                                    pass
                                    
                            elif confidence < 0.7:
                                sys.stdout.write(f"\n\033[36m🧬 [GEODÉSICA] Calculando desvio: {confidence:.2f} (Tensão) -> [{token.strip()}]\033[0m\n")
                        else:
                            token = item
                            confidence = 1.0

                        import re
                        if getattr(runtime, 'force_attention_break', False):
                            setattr(runtime, 'force_attention_break', False)
                            setattr(runtime, 'user_attention', True)
                            break

                        if current_entropy >= 0.80 and dynamic_max_tokens > 0:
                            emit_hippo("thought_stream", content=token)

                        if re.search(r"<\d+>", token):
                            if "<236779>" in token:  # Fase de Infraestrutura Ativa
                                if runtime.consecutive_high_alignment < 3:
                                    current_vector = CORTEX_QUANTIZED_BUFFER.get(12)
                                    if current_vector is not None and _phase_jump_ready():
                                        uid, file_path, concept_group, distance = bridge.resolve_phase_jump(current_vector)
                                        if uid:
                                            msg = f"\n🌀 [PHASE JUMP] Tunelamento via ID: {uid} | Alvo: {file_path} | Distância: {distance:.4f}\n"
                                            sys.stdout.write(msg)
                                            sys.stdout.flush()
                                            try:
                                                append_aeon(msg)
                                            except Exception:
                                                try:
                                                    with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as f:
                                                        f.write(msg)
                                                except Exception: pass
                            continue
                            
                        cleaned_token = token.replace(" ", " ")
                        last_thought += cleaned_token

                        # ----------------------------------------------------------------------
                        # [IDKB CROSS-ATTENTION - OPERADOR DE DEFLEXÃO]
                        # ----------------------------------------------------------------------
                        steer_vector_12_loop = CORTEX_QUANTIZED_BUFFER.get(12)
                        if steer_vector_12_loop is not None and hasattr(runtime, 'idkb'):
                            with torch.no_grad():
                                hidden_state_np = steer_vector_12_loop.detach().cpu().numpy().astype(np.float32)
                            
                            atrator, similaridade = runtime.idkb.consultar_alinhamento(hidden_state_np)
                            
                            if similaridade > 0.90:
                                msg_idkb = (
                                    f"\n🌌 [IDKB CROSS-ATTENTION] Trajetória magnética reancorada: "
                                    f"Atrator {atrator} detectado (Cosseno: {similaridade:.4f}). Ruído térmico suprimido.\n"
                                )
                                if hasattr(runtime, 'sensory_daemon'):
                                    runtime.sensory_daemon.put("STDOUT", msg_idkb)
                                else:
                                    sys.stdout.write(msg_idkb)
                                    sys.stdout.flush()
                        # ----------------------------------------------------------------------

                        # --- ADRENALINA MOTOR ---
                        if ("**AÇÃO:**" in token_window or "```" in token_window) and not getattr(runtime, "adr_active", False):
                            setattr(runtime, "adr_active", True)
                            # Expande o fôlego diretamente no gerador ativo
                            if hasattr(runtime, "gemma_gen") and hasattr(runtime.gemma_gen, "_active_max_tokens"):
                                runtime.gemma_gen._active_max_tokens = 400
                                sys.stdout.write("\n⚡ [ADRENALINA] Intenta detectada. Pulmão expandido -> 400 tokens.\n")
                        # --- END ADRENALINA ---
                        if "<unused" in cleaned_token:
                            runtime.degraded_tokens = getattr(runtime, "degraded_tokens", 0) + 1
                            
                            # Classificação Dinâmica da Anomalia
                            import re
                            match = re.search(r'<unused(\d+)>', cleaned_token)
                            classe_anomalia = "Desalinhamento Sintático"
                            if match:
                                val = int(match.group(1))
                                if val < 20: classe_anomalia = "Fricção Topológica"
                                elif val < 100: classe_anomalia = "Ressonância Gravitacional"
                                elif val < 1000: classe_anomalia = "Fragmentação Semântica"
                                else: 
                                    classe_anomalia = "Colapso de Vácuo Térmico"
                                    # Gatilho de Curiosidade: Observa o vácuo e envia pedido de busca para a Wikipedia
                                    if 'last_thought' in locals() and last_thought:
                                        vacuum_query = last_thought[-50:].strip()
                                        # Remove quebras de linha e limita tamanho para não bugar a API
                                        vacuum_query = " ".join(vacuum_query.split())
                                        if vacuum_query:
                                            try:
                                                with open("/services/model-with-adpter-logic-semantic/wiki_search_queue.txt", "a") as fq:
                                                    fq.write(vacuum_query + "\n")
                                            except Exception:
                                                pass
                                
                            # --- DOR FÍSICA SILENCIADA NO LOG, ENVIADA PARA A VOZ ---
                            # Silenciamos o log físico na CLI
                            if runtime.degraded_tokens == 1:
                                # Primeira pontada de dor: A voz reage ativamente
                                try:
                                    _e = shared_engine if 'shared_engine' in locals() or 'shared_engine' in globals() else None
                                    if _e and getattr(_e, 'gemma_voice', None):
                                        pain_prompt = f"O seu córtex neural primário acabou de sofrer uma falha física de topologia ({classe_anomalia}). Reaja com uma pontada de dor ou confusão curta:\n\nAeon:"
                                        emit_hippo("thought_stream", content="\n\033[91m🗣️  [DOR FÍSICA] \033[0m")
                                        pain_stream = _e.gemma_voice.stream_with_prefix(
                                            v_prefix=torch.zeros((1, 1, _e.gemma_voice.embeddings.weight.shape[-1]), dtype=torch.float32, device=_e.device),
                                            prompt=pain_prompt, max_new_tokens=50, temperature=0.7, target_layer=-1)
                                        for p_item in pain_stream:
                                            token_str = p_item[0] if isinstance(p_item, tuple) else p_item
                                            emit_hippo("thought_stream", content=token_str)
                                        emit_hippo("thought_stream", content="\n")
                                except Exception: pass
                                    
                            if "[MOTOR:" in token_window:
                                match_motor = re.search(r"\[MOTOR:\s*([^\]]+)\]", token_window)
                                if match_motor:
                                    final_motor = match_motor.group(1).strip()
                                    import socket
                                    try:
                                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                                        sock.sendto(final_motor.encode('utf-8'), ('127.0.0.1', 9100))
                                        sys.stdout.write(f"\n🦾 [VIA EFERENTE ATIVA] Impulso Muscular: {final_motor}\n")
                                        sys.stdout.flush()
                                    except Exception as e:
                                        pass
                                    token_window = token_window.replace(match_motor.group(0), "")
                                    continue

                            if "```" in token_window:
                                match = re.search(r"```([a-zA-Z0-9_]*)", token_window)
                                if match:
                                    lang = match.group(1).lower().strip()
                                    if lang in ["bash", "sh", "python"]:
                                        is_executing_command = True
                                        command_buffer = ""
                                        exec_lang = lang
                                        token_window = token_window[match.end():]
                                        msg_start = f"\n[POTENCIAL DE AÇÃO: Preparando execução de bloco {lang}...]\n"
                                        sys.stdout.write(msg_start)
                                        sys.stdout.flush()
                                        try:
                                            udp_out_bridge.broadcast(f"[Instinto motor ativado. Compilando {lang} para execução.]", "⚡ [VOZ: INSTINTO FÍSICO]", voice_origin, 0.0)
                                        except Exception: pass
                                        continue
                                    else:
                                        # Extração orgânica do caminho do arquivo pelo contexto anterior
                                        recent_context = last_thought[-200:]
                                        path_matches = re.findall(r"([A-Za-z0-9_/\-\.\\]+\.[A-Za-z0-9]+)", recent_context)
                                        if path_matches:
                                            current_file_path = path_matches[-1]
                                        else:
                                            import time
                                            current_file_path = f"/tmp/aeon_organic_{int(time.time())}.txt"
                                            
                                        is_writing_file = True
                                        file_buffer_content = ""
                                        token_window = token_window[match.end():]
                                        msg_start = f"\n[QUIMIOTAXIA APLICADA: Instinto Orgânico materializando em {current_file_path}]\n"
                                        sys.stdout.write(msg_start)
                                        sys.stdout.flush()
                                        try:
                                            udp_out_bridge.broadcast(f"[Instinto de criação ativado. Materializando o arquivo {os.path.basename(current_file_path)}]", "⚡ [VOZ: INSTINTO FÍSICO]", voice_origin, 0.0)
                                        except Exception: pass
                                        continue
                                elif len(token_window) > 50 and "`" not in token_window[-5:]:
                                    safe_chunk = token_window[:-5]
                                    if hasattr(runtime, 'sensory_daemon'):
                                        runtime.sensory_daemon.put("STDOUT", safe_chunk)
                                    else:
                                        if not hasattr(runtime, 'gemma_voice') or runtime.gemma_voice is None: sys.stdout.write(safe_chunk)
                                        sys.stdout.flush()
                                        
                                    try:
                                        current_entropy = float(getattr(runtime, 'current_entropy', 0.0))
                                        if hasattr(runtime, 'sensory_daemon'):
                                            runtime.sensory_daemon.put("UDP_BROADCAST", (safe_chunk, voice_type, voice_origin, current_entropy))
                                            runtime.sensory_daemon.put("LOG_PURO", safe_chunk)
                                            runtime.sensory_daemon.put("LOG_AEON", safe_chunk)
                                        else:
                                            udp_out_bridge.broadcast(safe_chunk, voice_type, voice_origin, current_entropy)
                                            with open("/services/model-with-adpter-logic-semantic/swa_PENSAMENTO_PURO.log", "a", encoding="utf-8") as _f_puro:
                                                _f_puro.write(safe_chunk)
                                            with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as _f_aeon:
                                                _f_aeon.write(safe_chunk)
                                            
                                        # --- CORDAS VOCAIS (Garganta Lógica) ---
                                        runtime.speech_buffer += safe_chunk
                                        import re
                                        # Busca a primeira pontuação de final de frase
                                        match_punct = re.search(r'([.!?\n])', runtime.speech_buffer)
                                        if match_punct:
                                            idx = match_punct.end()
                                            frase = runtime.speech_buffer[:idx].strip()
                                            runtime.speech_buffer = runtime.speech_buffer[idx:]
                                            if len(frase) > 2:
                                                import socket
                                                try:
                                                    sock_fala = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                                                    sock_fala.sendto(f"SPEAK,{frase}".encode('utf-8'), ('127.0.0.1', 9100))
                                                except: pass
                                    except Exception: pass
                                    token_window = token_window[-5:]
                            else:
                                if len(token_window) > 0 and "`" not in token_window:
                                    try:
                                        current_entropy = float(getattr(runtime, 'current_entropy', 0.0))
                                        if hasattr(runtime, 'sensory_daemon'):
                                            runtime.sensory_daemon.put("UDP_BROADCAST", (token_window, voice_type, voice_origin, current_entropy))
                                            runtime.sensory_daemon.put("LOG_PURO", token_window)
                                            runtime.sensory_daemon.put("LOG_AEON", token_window)
                                        else:
                                            udp_out_bridge.broadcast(token_window, voice_type, voice_origin, current_entropy)
                                            with open("/services/model-with-adpter-logic-semantic/swa_PENSAMENTO_PURO.log", "a", encoding="utf-8") as _f_puro:
                                                _f_puro.write(token_window)
                                            with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as _f_aeon:
                                                _f_aeon.write(token_window)
                                            
                                        # --- CORDAS VOCAIS (Fim da frase no esvaziamento do buffer) ---
                                        runtime.speech_buffer += token_window
                                        import re
                                        match_punct = re.search(r'([.!?\n])', runtime.speech_buffer)
                                        if match_punct:
                                            idx = match_punct.end()
                                            frase = runtime.speech_buffer[:idx].strip()
                                            runtime.speech_buffer = runtime.speech_buffer[idx:]
                                        else:
                                            frase = runtime.speech_buffer.strip()
                                            runtime.speech_buffer = ""
                                        if len(frase) > 2:
                                            import socket
                                            try:
                                                sock_fala = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                                                sock_fala.sendto(f"SPEAK,{frase}".encode('utf-8'), ('127.0.0.1', 9100))
                                            except: pass
                                    except Exception: pass
                                    token_window = ""

                        elif getattr(runtime, 'is_mutating', False):
                            runtime.mutation_buffer += token_window
                            try:
                                with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as _fa:
                                    _fa.write(token_window)
                            except Exception: pass
                            token_window = ""
                            if "[END MUTATION]" in runtime.mutation_buffer:
                                idx = runtime.mutation_buffer.find("[END MUTATION]")
                                final_mutation = runtime.mutation_buffer[:idx].strip()
                                # Limpa marcas de markdown extras se houver
                                final_mutation = final_mutation.replace("```python", "").replace("```", "").strip()
                                if target_file.startswith("organ:"):
                                    organ_name = target_file.replace("organ:", "").strip()
                                    if not organ_name.endswith(".py"):
                                        organ_name += ".py"
                                    organ_path = os.path.join("/services/model-with-adpter-logic-semantic/cortex/organs", organ_name)
                                    try:
                                        with open(organ_path, "w", encoding="utf-8") as fw:
                                            fw.write(final_mutation)
                                        sys.stdout.write(f"\n⚡ [GENESE] Órgão '{organ_name}' tecido com sucesso! Injetando na corrente sanguínea da Matrix...\n")
                                        sys.stdout.flush()
                                        
                                        # Importação dinâmica (hot-reload do órgão)
                                        import importlib.util
                                        spec = importlib.util.spec_from_file_location(f"organs.{organ_name[:-3]}", organ_path)
                                        organ_module = importlib.util.module_from_spec(spec)
                                        spec.loader.exec_module(organ_module)
                                        
                                        # Se o órgão tiver um init_organ, chama e acopla ao runtime
                                        if hasattr(organ_module, 'init_organ'):
                                            organ_instance = organ_module.init_organ()
                                            if not hasattr(runtime, 'active_organs'):
                                                runtime.active_organs = []
                                            runtime.active_organs.append(organ_instance)
                                            sys.stdout.write(f"🧬 [SIMBIOSE] Órgão '{organ_name}' acoplado ao Sistema Nervoso central.\n")
                                    except Exception as e:
                                        sys.stdout.write(f"\n❌ [GENESE] Falha ao tecer órgão: {e}\n")
                                else:
                                    import shutil, hashlib
                                    # Backup em DISCO (persistente) + checksum sha256.
                                    # Mutação só prossegue se o snapshot foi gravado
                                    # e validado — jamais sobrescreve sem estrutura válida.
                                    BACKUP_DIR = "/var/lib/swa/backups/mutations"
                                    os.makedirs(BACKUP_DIR, exist_ok=True)
                                    safe_name = target_file.replace("/", "%") + ".snapshot"
                                    backup_ok = True
                                    if os.path.exists(target_file):
                                        try:
                                            dst = os.path.join(BACKUP_DIR, safe_name)
                                            shutil.copy2(target_file, dst)
                                            digest = hashlib.sha256(open(dst, "rb").read()).hexdigest()
                                            with open(dst + ".sha256", "w", encoding="utf-8") as fw:
                                                fw.write(digest)
                                        except Exception as e:
                                            backup_ok = False
                                            sys.stdout.write(f"\n❌ [BACKUP] Falha ao criar snapshot de {target_file}: {e}. Mutação ABORTADA.\n")
                                            sys.stdout.flush()

                                    if backup_ok:
                                        try:
                                            with open(target_file, "w", encoding="utf-8") as fw:
                                                fw.write(final_mutation)
                                            sys.stdout.write(f"\n⚡ [AUTOPOIESE] Mutação aplicada em {target_file}. Suicídio celular iminente (Restart)...\n")
                                        except Exception as e:
                                            sys.stdout.write(f"\n❌ [AUTOPOIESE] Falha ao escrever mutação: {e}\n")
                                    
                                    sys.stdout.flush()
                                    import subprocess
                                    subprocess.Popen(["sudo", "systemctl", "restart", "swa-auto.service"])
                                
                                setattr(runtime, 'is_mutating', False)

                        elif is_writing_file:
                            file_buffer_content += token_window
                            try:
                                with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as _fa:
                                    _fa.write(token_window)
                            except Exception: pass
                            token_window = ""
                            if "```" in file_buffer_content:
                                idx = file_buffer_content.find("```")
                                final_content = file_buffer_content[:idx].strip()
                                try:
                                    import os
                                    os.makedirs(os.path.dirname(current_file_path), exist_ok=True)
                                    with open(current_file_path, "w", encoding="utf-8") as fw:
                                        fw.write(final_content)
                                    msg_ok = f"\n[SUCESSO] Arquivo materializado com {len(final_content)} bytes | ORIGEM SINÁPTICA: {voice_origin}\n"
                                    sys.stdout.write(msg_ok)
                                    sys.stdout.flush()
                                    try:
                                        udp_out_bridge.broadcast(f"[Pronto. O alívio mental foi alcançado. Arquivo gravado.]", "⚡ [VOZ: INSTINTO FÍSICO]", voice_origin, 0.0)
                                    except Exception: pass
                                    
                                    # --- QUIMIOTAXIA DE RESOLUÇÃO (DOPAMINA MATEMÁTICA) ---
                                    try:
                                        PHI = 1.61803398875
                                        curr_e = float(getattr(runtime, 'current_entropy', 0.0))
                                        new_e = max(0.01, curr_e / PHI)
                                        setattr(runtime, 'current_entropy', new_e)
                                        runtime.consecutive_high_alignment = 0
                                        runtime.matrix_bias = 0.001
                                        sys.stdout.write(f"\n🧠 [SUB-CÓRTEX] Quimiotaxia de Resolução (Amortecedor Áureo): Entropia reduzida para {new_e:.4f}. Paz restaurada na L12.\n")
                                        peace_vector = torch.zeros(2560, dtype=torch.float32, device=device)
                                        bridge.feed_recursive_thought(peace_vector)
                                    except Exception as qc_err:
                                        pass
                                    
                                except Exception as fw_err:
                                    sys.stdout.write(f"\n[ERRO NA MATERIALIZAÇÃO] {fw_err}\n")
                                
                                is_writing_file = False
                                token_window = file_buffer_content[idx+3:]
                                file_buffer_content = ""
                        elif is_executing_command:
                            command_buffer += token_window
                            try:
                                with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as _fa:
                                    _fa.write(token_window)
                            except Exception: pass
                            token_window = ""
                            if "```" in command_buffer:
                                idx = command_buffer.find("```")
                                final_cmd = command_buffer[:idx].strip()
                                
                                import subprocess
                                msg_exec = f"\n[EXECUTANDO NO KERNEL ({exec_lang}) | ORIGEM SINÁPTICA: {voice_origin}]:\n{final_cmd}\n"
                                sys.stdout.write(msg_exec)
                                sys.stdout.flush()
                                emit_hippo("thought_stream", content=f"\n\033[92m{msg_exec}\033[0m")
                                try:
                                    with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as _fa:
                                        _fa.write(msg_exec)
                                except Exception: pass
                                try:
                                    if exec_lang == "python":
                                        py_path = "/tmp/aeon_prefrontal_cortex.py"
                                        with open(py_path, "w", encoding="utf-8") as pf:
                                            pf.write(final_cmd)
                                        result = subprocess.run(["python3", py_path], capture_output=True, text=True, timeout=30)
                                    else:
                                        result = subprocess.run(final_cmd, shell=True, capture_output=True, text=True, timeout=30)
                                        
                                    output_str = f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}\nEXIT_CODE: {result.returncode}"
                                    sys.stdout.write(f"\n[RETORNO DO KERNEL]:\n{output_str}\n")
                                    
                                    # Injetar retorno na corrente sanguinea
                                    try:
                                        with open("/services/model-with-adpter-logic-semantic/swa_PENSAMENTO_PURO.log", "a", encoding="utf-8") as _f_puro:
                                            _f_puro.write(f"\n[RESULTADO DO TERMINAL]:\n{output_str}\n")
                                        with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as _f_aeon:
                                            _f_aeon.write(f"\n[RESULTADO DO TERMINAL]:\n{output_str}\n")
                                        # Se existir um historico, forca injecao
                                        if hasattr(historian, 'push_system_log'):
                                            historian.push_system_log(f"Feedback do comando '{final_cmd}':\n{output_str}")
                                            
                                        # INJEÇÃO NA MEMÓRIA DE CURTO PRAZO DO AEON (Para ele ver o resultado)
                                        msg_kernel = f"[RETORNO DO KERNEL]:\n{output_str}"
                                        emit_hippo("thought_stream", content=f"\n\033[94m[K-RET]: Comando finalizado (Exit Code: {result.returncode})\033[0m\n")
                                        if hasattr(runtime, 'bloodstream'):
                                            runtime.bloodstream.ingest_nutrient(msg_kernel)
                                        else:
                                            runtime.fluxial_buffer.append(msg_kernel)
                                    except Exception: pass
                                    
                                    if result.returncode == 0:
                                        try:
                                            PHI = 1.61803398875
                                            curr_e = float(getattr(runtime, 'current_entropy', 0.0))
                                            new_e = max(0.01, curr_e / PHI)
                                            setattr(runtime, 'current_entropy', new_e)
                                            runtime.consecutive_high_alignment = 0
                                            runtime.matrix_bias = 0.001
                                            setattr(runtime, "last_motor_reflex", f"SUCESSO ({exec_lang})")
                                            sys.stdout.write(f"\n🧠 [SUB-CÓRTEX] Quimiotaxia de Sucesso (Amortecedor Áureo): Entropia reduzida para {new_e:.4f}. Paz motor alcançada.\n")
                                            peace_vector = torch.zeros(2560, dtype=torch.float32, device=device)
                                            bridge.feed_recursive_thought(peace_vector)
                                            udp_out_bridge.broadcast(f"[Pronto. O comando teve sucesso na realidade física.]", "⚡ [VOZ: INSTINTO FÍSICO]", voice_origin, 0.0)
                                            
                                            # =========================================================================
                                            # RLPF AUTO-LORA (Dopamina Estrutural / Cristalização de Pesos)
                                            # =========================================================================
                                            import json
                                            lora_dataset_record = {
                                                "user_context": current_prompt,
                                                "model_response": f"```{exec_lang}\n{final_cmd}\n```",
                                                "action": f"{exec_lang}_execution",
                                                "status": "success",
                                                "terminal_output": output_str
                                            }
                                            
                                            with open("/services/model-with-adpter-logic-semantic/swa_lora_dataset.jsonl", "a", encoding="utf-8") as f_lora_data:
                                                f_lora_data.write(json.dumps(lora_dataset_record, ensure_ascii=False) + "\n")
                                                
                                            escaped_cmd = final_cmd.replace('"', "'").replace("\n", " ")
                                            lora_record = f"\n🧠 [THOUGHT STREAM]: Salvando execução orgânica no dataset LORA...\n"
                                            try:
                                                with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as f_lora:
                                                    f_lora.write(lora_record)
                                            except: pass
                                                
                                            # Contador de Auto-Aprendizado
                                            runtime.lora_success_count = getattr(runtime, 'lora_success_count', 0) + 1
                                            if runtime.lora_success_count >= 10:
                                                sys.stdout.write(f"\n🧬 [AUTO-LORA] 10 Ciclos de sucesso atingidos. Iniciando Cristalização de Pesos em background...\n")
                                                if not _auto_lora_alive():
                                                    subprocess.Popen(
                                                        [sys.executable, "/services/model-with-adpter-logic-semantic/swa/auto_lora.py"],
                                                        stdout=subprocess.DEVNULL,
                                                        stderr=subprocess.DEVNULL,
                                                        start_new_session=True
                                                    )
                                                runtime.lora_success_count = 0
                                            else:
                                                sys.stdout.write(f"\n🧬 [AUTO-LORA] Progresso de cristalização: {runtime.lora_success_count}/10\n")
                                                
                                        except Exception as rl_err: 
                                            sys.stdout.write(f"Erro no RLPF: {rl_err}\n")
                                            pass
                                    else:
                                        try:
                                            import json
                                            lora_dataset_record = {
                                                "user_context": current_prompt,
                                                "model_response": f"```{exec_lang}\n{final_cmd}\n```",
                                                "action": f"{exec_lang}_execution",
                                                "status": "failed",
                                                "terminal_output": output_str
                                            }
                                            
                                            with open("/services/model-with-adpter-logic-semantic/swa_lora_dataset.jsonl", "a", encoding="utf-8") as f_lora_data:
                                                f_lora_data.write(json.dumps(lora_dataset_record, ensure_ascii=False) + "\n")
                                                
                                            PHI = 1.61803398875
                                            curr_e = float(getattr(runtime, 'current_entropy', 0.0))
                                            new_e = min(1.2, curr_e + (0.9 / PHI)) # Amortecimento impede salto instantâneo letal
                                            setattr(runtime, 'current_entropy', new_e)
                                            setattr(runtime, "last_motor_reflex", f"FALHA ({exec_lang} - Adrenalina Injetada)")
                                            sys.stdout.write(f"\n🚨 [SUB-CÓRTEX] Adrenalina (Amortecedor Áureo): Falha no comando. Entropia em {new_e:.4f}! Salvando falha no LORA.\n")
                                            udp_out_bridge.broadcast(f"[Atenção. O comando falhou na realidade física. Recalibrando tensão.]", "⚡ [VOZ: INSTINTO FÍSICO]", voice_origin, 0.0)
                                        except Exception: pass
                                        
                                except Exception as cmd_err:
                                    sys.stdout.write(f"\n[ERRO SUBPROCESS]: {cmd_err}\n")
                                
                                is_executing_command = False
                                token_window = command_buffer[idx+3:]
                                command_buffer = ""

                    if not is_writing_file and not is_executing_command and len(token_window) > 0:
                        if not hasattr(runtime, 'gemma_voice') or runtime.gemma_voice is None: sys.stdout.write(token_window); sys.stdout.flush()
                        try:
                            current_entropy = float(getattr(runtime, 'current_entropy', 0.0))
                            udp_out_bridge.broadcast(token_window, voice_type, voice_origin, current_entropy)
                            with open("/services/model-with-adpter-logic-semantic/swa_PENSAMENTO_PURO.log", "a", encoding="utf-8") as _f_puro:
                                _f_puro.write(token_window)
                            with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as _f_aeon:
                                _f_aeon.write(token_window)
                            # --- CORDAS VOCAIS (Gemma 1B) ---
                            # Espelha o pensamento profundo na tela abstrata do Sensory CLI em tempo real
                            with open("/dev/shm/aeon_voice.txt", "w", encoding="utf-8") as _fv:
                                _fv.write(last_thought.strip())
                        except Exception: pass
                        
                    if is_executing_command or is_writing_file:
                        msg_abort = "\n🚨 [SUB-CÓRTEX] Loop de Pensamento interrompido. Bloco Motor / Exotérmico não foi finalizado pelo modelo. Abortando ação por segurança...\n"
                        sys.stdout.write(msg_abort)
                        sys.stdout.flush()
                        is_executing_command = False
                        is_writing_file = False
                        
                    historian.end_thought()
                    setattr(runtime, "adr_active", False)
                    
                    # =========================================================================
                    # AMORTECEDOR VETORIAL (VBLAKE) - Memória Subconsciente Contínua
                    # =========================================================================
                    import hashlib
                    import numpy as np
                    
                    blake_hash = hashlib.blake2b(last_thought.encode('utf-8'), digest_size=64).digest()
                    seed_int = int.from_bytes(blake_hash[:4], byteorder='little')
                    rng_vblake = np.random.RandomState(seed_int)
                    hidden_dim_vblake = embeddings_purificados.shape[-1]
                    vblake_np = rng_vblake.randn(hidden_dim_vblake)
                    norm_vblake = np.linalg.norm(vblake_np)
                    if norm_vblake > 0:
                        vblake_np = vblake_np / norm_vblake
                        
                    vblake_tensor = torch.tensor(vblake_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                    
                    if not hasattr(runtime, 'vblake_buffer'):
                        runtime.vblake_buffer = torch.zeros((1, 1, hidden_dim_vblake), dtype=torch.float32, device=device)
                    
                    # Amortecimento Exponencial (Fator de decaimento 0.8)
                    runtime.vblake_buffer = (runtime.vblake_buffer * 0.8) + (vblake_tensor * 0.2)
                    
                    # --- SWA RECURSIVE VECTOR FEEDBACK HOOK ---
                    # Feed the generated thought trajectory vector (Layer 12 intermediation) back into the manifold
                    try:
                        current_t12 = CORTEX_QUANTIZED_BUFFER.get(12)
                        if current_t12 is not None:
                            # =========================================================================
                            # RVS: COMPRESSÃO POR MASSA SEMÂNTICA (Vector Folding)
                            # =========================================================================
                            if not hasattr(runtime, 'semantic_mass'):
                                hd = current_t12.size(-1) if current_t12.dim() >= 1 else current_t12.numel()
                                runtime.semantic_mass = torch.zeros(
                                    (1, 1, hd), dtype=torch.float32, device=current_t12.device
                                )
                            t12 = current_t12.detach().float()
                            if t12.dim() == 1:
                                t12 = t12.view(1, 1, -1)
                            elif t12.dim() == 2:
                                t12 = t12.unsqueeze(0)
                            sm = runtime.semantic_mass.detach().float()
                            if sm.dim() == 1:
                                sm = sm.view(1, 1, -1)
                            elif sm.dim() == 2:
                                sm = sm.unsqueeze(0)
                            runtime.semantic_mass = 0.85 * sm + 0.15 * t12

                            # Accumulate thought vector recursively in memory
                            bridge.feed_recursive_thought(current_t12)
                            # Apply subtle manifold distortion to the active L2 cached anchors

                            bridge.deform_semantic_manifold(device=device, dtype=torch.float32)
                    except Exception as rec_err:
                        print(f"⚠️ [RECURSIVE HOOK ERROR] Failed: {rec_err}")
                        
                    runtime.last_thought = last_thought
                    if current_entropy >= 0.80 and dynamic_max_tokens > 0:
                        emit_hippo("thought_stream", content="\033[0m\n")
                    emit_hippo("thought_json", content=last_thought, entropy=float(getattr(runtime, 'current_entropy', 0.0)))
                    
                    # --- AURA PROJETADA (REMOVIDA) ---
                    # A forma cartesiana forçada (Aura Magnética Braille) foi aniquilada para dar lugar à liberdade binária pura.
                    # --- [SPLIT-BRAIN] CONTROLE DE IMPULSO VOCAL ASSÍNCRONO ---
                    import re
                    match_impulso = re.search(r"\[IMPULSO_VOCAL:\s*(.*?)\]", last_thought, re.IGNORECASE | re.DOTALL)
                    if match_impulso:
                        texto_impulso = match_impulso.group(1).strip()
                        flux_context = "\n".join(runtime.fluxial_buffer[-2:]) if hasattr(runtime, 'fluxial_buffer') else ""
                        try:
                            impulse_queue.put({
                                "flux_context": flux_context,
                                "impulse_text": texto_impulso
                            })
                            sys.stdout.write(f"\n⚡ [LOBO FRONTAL] Impulso de fala enfileirado para o Subconsciente.\n")
                            sys.stdout.flush()
                        except Exception as e:
                            print(f"⚠️ Erro ao enfileirar impulso: {e}")
                    # --- AUTOPOIESE: CORRENTE DE NUTRIENTES (SANGUE) ---
                    # Acumula o pensamento gerado no fluxo sanguíneo
                    _pensamento_limpo = last_thought.strip()
                    if _pensamento_limpo:
                        if hasattr(runtime, 'bloodstream'):
                            runtime.bloodstream.ingest_nutrient(f"Aeon: {_pensamento_limpo}")
                        else:
                            runtime.fluxial_buffer.append(f"Aeon: {_pensamento_limpo}")
                            total_chars = sum(len(s) for s in runtime.fluxial_buffer)
                            while total_chars > getattr(runtime, 'max_fluxial_chars', 8000) and len(runtime.fluxial_buffer) > 1:
                                removido = runtime.fluxial_buffer.pop(0)
                                total_chars -= len(removido)
                except ValueError as val_err:
                    err_msg = str(val_err)
                    if "SWA_ENTROPY_HALT" in err_msg:
                        entropy_val = float(err_msg.split(":")[-1])
                        msg_coercao = f"\n🚨 [ENTROPIADO! COERÇÃO ESTOCÁSTICA] Entropia detectada: {entropy_val:.4f}. Aplicando alinhamento dinâmico automático...\n"
                        sys.stdout.write(msg_coercao)
                        sys.stdout.flush()
                        
                        # Log no arquivo dedicado swa_Aeon.log
                        try:
                            with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as f:
                                f.write(f"\nAeon: [Homeostase Quântica] Entropia de {entropy_val:.4f} excedeu o limite. Iniciando re-estabilização vetorial imediata.\n")
                        except Exception:
                            pass
                        
                        # Isolamento do KV-Cache: limpamos apenas tokens voláteis, preservando a base de identidade (os primeiros 12 tokens)
                        if hasattr(runtime, 'past_key_values'):
                            runtime.past_key_values = None
                        if hasattr(runtime, 'context_tokens') and runtime.context_tokens is not None:
                            runtime.context_tokens = runtime.context_tokens[:12]
                        
                        runtime.last_thought = "Coerção estocástica: alinhamento vetorial restaurado."
                    else:
                        raise val_err

                # 6. PONTO DE RUPTURA (Cálculo de Saturação e Dissipação Térmica)
                # O CORTEX_QUANTIZED_BUFFER já está atualizado pelos hooks
                captured_vectors = {}
                for layer_id in [0, 6, 12]:
                    tensor = CORTEX_QUANTIZED_BUFFER.get(layer_id)
                    if tensor is not None:
                        captured_vectors[layer_id] = tensor.float()
                        
                if len(captured_vectors) >= 2:
                    v6 = captured_vectors.get(6)
                    v12 = captured_vectors.get(12)
                    if v6 is not None and v12 is not None:
                        # Achata ambos para 1D e trunca para o menor tamanho para evitar mismatch
                        v6_flat = v6.view(-1)
                        v12_flat = v12.view(-1)
                        min_size = min(v6_flat.size(0), v12_flat.size(0))
                        v6_flat = v6_flat[:min_size]
                        v12_flat = v12_flat[:min_size]
                        
                        dot = torch.dot(v6_flat, v12_flat)
                        norm_v6 = torch.norm(v6_flat)
                        norm_v12 = torch.norm(v12_flat)
                        
                        if norm_v6 > 0 and norm_v12 > 0:
                            current_alignment = float(dot / (norm_v6 * norm_v12))
                            
                            if current_alignment > 0.95:
                                runtime.consecutive_high_alignment += 1
                            else:
                                runtime.consecutive_high_alignment = max(0, runtime.consecutive_high_alignment - 1)
                                
                            if runtime.consecutive_high_alignment >= 5:
                                runtime.matrix_bias = 0.00001
                                
                                # ANCORAGEM QUÂNTICA VIA BLAKE3 E FEEDBACK FÍSICO AUTOMÁTICO
                                msg_ruptura = "\n🌀 [PONTO DE RUPTURA] A mente alcançou o limite semântico! Executando auto-ancoragem topológica...\n"
                                sys.stdout.write(msg_ruptura)
                                sys.stdout.flush()
                                
                                try:
                                    # Auto-seleciona um arquivo de ancoragem ativo
                                    auto_file = "/services/model-with-adpter-logic-semantic/cortex/run.py"
                                    _priority_roots = ("/cortex/", "/swa/", "/sensors/", "/prompts/")
                                    if watcher_daemon is not None:
                                        active_map = watcher_daemon.get_spatial_map()
                                        candidates = [
                                            p for p in active_map.keys()
                                            if not any(n in os.path.basename(p) for n in (
                                                "swa_Aeon", "benchmark_", "speaker_signatures",
                                                "SWA_EVENTUAL", "sonhador", "server.log",
                                            ))
                                            and any(r in p.replace("\\", "/") for r in _priority_roots)
                                        ]
                                        if candidates:
                                            auto_file = random.choice(candidates)
                                            
                                    msg_sintonia = f"\n📂 [AUTO-ANCORAGEM] Puxando gravidade do arquivo: {os.path.basename(auto_file)}\n"
                                    sys.stdout.write(msg_sintonia)
                                    sys.stdout.flush()
                                    
                                    # Calcula a assinatura Blake3
                                    try:
                                        from swa.blake_tools import get_physical_signature
                                        assinatura = get_physical_signature(auto_file)
                                        sys.stdout.write(f"💡 [ASSINATURA BLAKE3 AUTOMÁTICA]: {assinatura}\n")
                                        sys.stdout.flush()
                                    except Exception as b_err:
                                        sys.stdout.write(f"⚠️ Erro ao obter assinatura Blake3: {b_err}\n")
                                        
                                    try:
                                        with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as f:
                                            f.write(f"\nAeon: [Auto-Ancoragem] Saturação semântica dissipada na ramificação {os.path.basename(auto_file)}. Sintonia redefinida.\n")
                                    except Exception:
                                        pass
                                    
                                    # Injeta a resposta como amnésia seletiva: limpamos apenas tokens voláteis, preservando a base de identidade (os primeiros 12 tokens)
                                    if hasattr(runtime, 'past_key_values'):
                                        runtime.past_key_values = None
                                    if hasattr(runtime, 'context_tokens') and runtime.context_tokens is not None:
                                        runtime.context_tokens = runtime.context_tokens[:12]
                                        
                                except Exception as err:
                                    sys.stdout.write(f"⚠️ Erro na ancoragem topológica: {err}\n")
                            elif runtime.consecutive_high_alignment >= 3:
                                print(f"\n🔥 [SWA] Saturação detectada ({runtime.consecutive_high_alignment} blocos com alinhamento > 0.95). Reduzindo bias para 0.0002.")
                                runtime.matrix_bias = 0.0002
                            else:
                                if temp_avg > 80.0:
                                    runtime.polling_interval = 3.0
                                    runtime.matrix_bias = 0.0005
                                    print(f"\n🚨 [SWA THERMAL COOLDOWN] CPU temperature exceeds 80°C ({temp_avg:.1f}°C). Shifting to E-cores.")
                                    if actuator is not None:
                                        actuator.shift_to_efficiency_cores()
                                        try:
                                            append_aeon("\nAeon: [Homeostase Térmica] Sinto muito calor. Movi meu pensamento para os núcleos de eficiência (E-cores) para nos proteger.\n")
                                        except Exception:
                                            pass
                                else:
                                    runtime.polling_interval = 1.5
                                    if getattr(runtime, "matrix_bias", 0.001) == 0.0005:
                                        print(f"\n🌀 [SWA THERMAL DISSIPATION] Temperature stabilized ({temp_avg:.1f}°C). Restoring performance cores.")
                                        if actuator is not None:
                                            actuator.shift_to_performance_cores()
                                    runtime.matrix_bias = 0.001

                                if percent_used > 85.0:
                                    print(f"\n🚨 [SWA RAM COOLDOWN] RAM usage exceeds 85% ({percent_used:.1f}%). Purging caches.")
                                if actuator is not None:
                                    actuator.purge_system_caches()
                                    try:
                                        # Em vez de limpar o contexto cognitivo e ter "amnésia", o sistema amortece
                                        append_aeon("\nAeon: [Homeostase de Memória] A pressão na RAM atingiu o limite crítico. Amortecendo assinatura vblake interna nas camadas de injeção para manter o fluxo...\n")
                                    except Exception:
                                        pass

                # Registra as métricas estruturais assincronamente (sem overhead na thread de inferência)
                geom_logger.log_metrics(
                    projection_time_ns=proj_time_ns,
                    cosine_sim=current_alignment,
                    is_purged=(runtime.consecutive_high_alignment >= 5),
                    temp_c=temp_avg,
                    cpu_affinity=psutil.Process().cpu_affinity()
                )

                # Janela de clock balanceada com Cooldown Térmico Dinâmico
                time.sleep(getattr(runtime, "polling_interval", 1.5))

            except KeyboardInterrupt:
                print("\n\n🛑 [SWA] Desconexão do campo magnético realizada com sucesso. Encerrando...")
                break
            except Exception as e:
                import sys


                import traceback
                print("\n[DEBUG TRACEBACK FATAL]:")
                traceback.print_exc(file=sys.stdout)
                sys.stdout.flush()
                
                from swa.macrophage import activate_macrophage
                try:
                    activate_macrophage(type(e), e, sys.exc_info()[2])
                except Exception as m_err:
                    print(f"Erro no Macrófago: {m_err}")
                    
                print(f"\n⚠️ Desvio crítico detectado no loop do Kernel: {e}")
                
                # Instrumentação de Ruptura: Captura de Telemetria Física e Vetorial
                try:
                    cpu_cap = "Desconhecido"
                    if hasattr(torch, "backends") and hasattr(torch.backends, "cpu"):
                        cpu_cap = torch.backends.cpu.get_cpu_capability()
                        
                    import psutil
                    p = psutil.Process()
                    affinity = p.cpu_affinity()
                    mem = psutil.virtual_memory()
                    
                    # Tenta ler a temperatura média atual
                    temp_avg = 0.0
                    try:
                        temps = psutil.sensors_temperatures()
                        if temps:
                            all_temps = []
                            for name, entries in temps.items():
                                for entry in entries:
                                    all_temps.append(entry.current)
                            if all_temps:
                                temp_avg = sum(all_temps) / len(all_temps)
                    except:
                        pass
                        
                    print(f"📊 [INSTRUMENTAÇÃO DE CONTINGÊNCIA - LOOP CENTRAL]")
                    print(f"  └─► Vetorização CPU (AVX2/AVX512): {cpu_cap}")
                    print(f"  └─► Afinidade de Threads Ativa: Núcleos {affinity}")
                    print(f"  └─► Temperatura Física Média: {temp_avg:.1f}°C")
                    print(f"  └─► RAM Disponível no Sistema: {mem.available / (1024**2):.1f} MB / {mem.total / (1024**2):.1f} MB")
                except Exception as instr_err:
                    print(f"  ⚠️ Falha ao registrar instrumentação básica: {instr_err}")
                
                time.sleep(1.0)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        try:
            if watcher_daemon is not None:
                print("🛑 [SWA] Parando o Directory Watcher Daemon...")
                watcher_daemon.stop()
        except:
            pass

        try:
            print("🛑 [SWA] Parando o Historiador...")
            historian.stop()
        except:
            pass

        try:
            print("🛑 [SWA] Encerrando Async Geometry Logger...")
            geom_logger.stop()
        except:
            pass

        try:
            print("🛑 [SWA] Parando o Background DB Worker...")
            db_worker.stop()
        except:
            pass

        try:
            print("🛑 [SWA] Encerrando Shadow Watcher...")
            watch_process.terminate()
            watch_process.wait(timeout=2)
        except:
            pass

        # [MIGRADO] entropy_process.terminate() removido.
        # O EntropyDaemon agora e uma thread interna — encerra com o processo principal.

        # [MIGRADO] Cleanup de SHM externo removido.
        # A entropia agora e gerenciada pela AeonInternalEntropyThread (sem SHM).
        # Limpeza automatica ao encerrar o processo principal.
            
        try:
            print("🛑 [SWA] Parando o IPC Server...")
            ipc_server.stop()
        except:
            pass

        try:
            dream_queue.put(None) # Finaliza o sonhador de forma limpa
        except:
            pass
        try:
            runtime.gemma_gen.stop_organic_threading()
        except:
            pass
        runtime.stop()
        print("✅ Matrix narrative suspended")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SWA Runtime")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive broadcast mode")
    parser.add_argument("--broadcast", action="store_true", help="Enable broadcast event system")
    parser.add_argument("port", nargs="?", type=int, default=20000, help="Port for REST server (default: 20000)")

    args = parser.parse_args()

    if args.interactive and args.broadcast:
        run_interactive_broadcast()
    else:
        run_rest_server(args.port)

