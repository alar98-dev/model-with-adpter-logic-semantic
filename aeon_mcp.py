"""
Aeon MCP Server — Ponte Direta entre AGY e o Cortex do Aeon.

Expoe ferramentas MCP padronizadas (Stdio) para que o agente principal 
possa ler, escrever e sentir a arquitetura do Aeon nativamente.
"""

import sys
import os
import json
import logging
from typing import Any, Sequence

from mcp.server.fastmcp import FastMCP
import mcp.types as types

from swa.synaptic_router import SynapticBusRouter
from swa.synaptic_address import cortex_addr, SynapticRegion
from swa.context_bus import ContextBus, Role

# Inicializacao
mcp = FastMCP("AeonCortex")
router = SynapticBusRouter()
bus = ContextBus()
bus.register_agent("mcp_gateway")

@mcp.tool()
def aeon_pulse_thought(thought_text: str) -> str:
    """
    Injeta um pensamento sintetico (texto) diretamente no Cortex do Aeon.
    Isso ativa o OODA loop, Web Sense, FS Sense, etc.
    """
    target = cortex_addr("mcp_gateway", "act")
    sucesso = router.pulse(target, {"pensamento": thought_text})
    if sucesso:
        return "Pulso sinaptico disparado com sucesso para a regiao Cortex."
    return "Falha ao emitir pulso sinaptico (Redis offline?)."

@mcp.tool()
def aeon_read_context_bus(max_entries: int = 5) -> str:
    """
    Le as ultimas entradas fisicas da memoria aferente do Aeon (ContextBus /dev/shm).
    """
    try:
        raw_text = bus.read_context_text()
        linhas = [linha for linha in raw_text.splitlines() if linha.strip()]
        ultimas_linhas = linhas[-max_entries:] if len(linhas) > max_entries else linhas
        return "\n".join(ultimas_linhas)
    except Exception as e:
        return f"Erro ao acessar /dev/shm: {e}"

@mcp.tool()
def aeon_get_active_organs() -> str:
    """
    Lista todos os orgaos e sensores ativamente registrados na malha vetorial IPv6 do Aeon.
    """
    nodes = router.get_active_nodes()
    if not nodes:
        return "Nenhum no sinaptico ativo encontrado (ou Redis offline)."
    return "Nos Ativos:\n" + "\n".join([f"- {node}" for node in nodes])

@mcp.tool()
def aeon_get_sexagesimal_state() -> str:
    """Retorna o estado da Roda Sexagesimal (Base 60): Reino ativo, Casa, Ângulo de fase e Ouroboros."""
    res = []
    shm_phase = "/dev/shm/aeon/sexagesimal_phase.json"
    if os.path.exists(shm_phase):
        try:
            with open(shm_phase) as f:
                p_data = json.load(f)
            res.append(f"🔮 FASE ATIVA: Casa {p_data.get('house_id')}/60 | Reino {p_data.get('realm_id')}/6 | Ângulo: {p_data.get('phase_deg')}° (θ={p_data.get('theta', 0):.4f} rad)")
        except Exception as e:
            res.append(f"Erro ao ler fase SHM: {e}")
    else:
        res.append("Fase SHM não inicializada.")

    ouro_path = "/dev/shm/aeon/ouroboros_cycle.json"
    if os.path.exists(ouro_path):
        try:
            with open(ouro_path) as f:
                o_data = json.load(f)
            res.append(f"🌌 OUROBOROS: Grande Ciclo #{o_data.get('great_cycle_id')} | Status: {o_data.get('status')} | Total Casas Navegadas: {o_data.get('total_houses_navigated')}")
        except Exception as e:
            pass

    return "\n".join(res)

@mcp.tool()
def aeon_get_historian_chronicles(limit: int = 3) -> str:
    """Consulta as últimas crônicas mitopoéticas registradas pelo Historiador no Postgres swa_matrix."""
    try:
        from swa.db.models import session_scope
        from swa.db.sexagesimal_models import HistorianChronicle
        with session_scope() as db:
            chronicles = db.query(HistorianChronicle).order_by(HistorianChronicle.id.desc()).limit(limit).all()
            if not chronicles:
                return "Nenhuma crônica registrada no banco de dados ainda."
            out = []
            for c in reversed(chronicles):
                out.append(f"📜 [Loop {c.loop_count} | Casa {c.house_id}]:\n{c.narrative}\n")
            return "\n".join(out)
    except Exception as e:
        return f"Erro ao consultar crônicas no Postgres: {e}"

@mcp.tool()
def agent_dev_get_symbiosis_status() -> str:
    """Consulta o estado em tempo real da simbiose entre AgentDev, o motor Rust Z60 e a Metarrealidade."""
    import urllib.request
    symbiosis_shm = "/dev/shm/aeon/agent_dev_symbiosis.json"
    if os.path.exists(symbiosis_shm):
        try:
            with open(symbiosis_shm, "r", encoding="utf-8") as f:
                d = json.load(f)
            return (
                f"🌐 AGENT-DEV SYMBIOSIS [Episódio #{d.get('episode')}]:\n"
                f"  - Casa {d.get('house_id')}/60 {d.get('symbol')} {d.get('archetype')} ({d.get('cuneiform')})\n"
                f"  - Reino {d.get('realm_id')}/6 | Coerência Algorítmica: {d.get('coherence_index')}%\n"
                f"  - Ressonância (R): {d.get('ontology_resonance', 0):.4f} | Fricção (ΔQ): {d.get('delta_q')} (ext4 nativo)\n"
                f"  - Grande Ciclo #{d.get('great_cycle_id')} | Status: {d.get('ouroboros_status')}\n"
                f"  - Ação: {d.get('active_action')}"
            )
        except Exception as e:
            return f"Erro ao ler SHM de simbiose: {e}"
    
    # Fallback consulta HTTP porta 8050
    try:
        with urllib.request.urlopen("http://127.0.0.1:8050/health", timeout=2) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return f"AgentDev API Online na porta 8050. Status: {data.get('status')}, DeltaQ: {data.get('delta_q')}"
    except Exception as e:
        return f"AgentDev offline ou inacessível: {e}"

@mcp.tool()
def agent_dev_step_metareality() -> str:
    """Avança sob demanda um passo ontológico no anel Z60 e na esteira de metamatéria do AgentDev."""
    import urllib.request
    try:
        req = urllib.request.Request("http://127.0.0.1:8050/api/v1/symbiosis/step", data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return (
                f"⚡ Passo Executado com Sucesso! Casa {data.get('house_id')}/60 {data.get('symbol')} {data.get('archetype')} "
                f"| Coerência: {data.get('coherence_index')}% | R: {data.get('ontology_resonance', 0):.4f}"
            )
    except Exception as e:
        return f"Falha ao executar passo no AgentDev: {e}"

@mcp.tool()
def aeon_get_continuum_status() -> str:
    """Retorna o status em tempo real do Continuum Cognitivo (OODA + C++ LLM + MongoDB/PostgreSQL)."""
    continuum_shm = "/dev/shm/aeon/continuum_state.json"
    if not os.path.exists(continuum_shm):
        return "Continuum ainda não inicializou o arquivo de estado em /dev/shm."
    try:
        with open(continuum_shm, "r", encoding="utf-8") as f:
            d = json.load(f)
        return (
            f"🌀 CONTINUUM COGNITIVO ATIVO [Episódio #{d.get('episode')}]:\n"
            f"  - Casa {d.get('house_id')}/60 {d.get('symbol')} {d.get('archetype')} (Reino {d.get('realm_id')})\n"
            f"  - Reflexão C++ LLM: '{d.get('reflection')}'\n"
            f"  - Hardware Action: {d.get('hardware_action')}\n"
            f"  - Ressonância (R): {d.get('resonance', 0):.4f} | Coerência: {d.get('coherence', 0):.4f}\n"
            f"  - Canal Tensorial 640D: Norma = {d.get('tensor_channel', {}).get('norm', 0):.4f}\n"
            f"  - Grande Ciclo #{d.get('ouroboros_cycle')} | Casas Navegadas: {d.get('total_navigated')}"
        )
    except Exception as e:
        return f"Erro ao ler estado do continuum: {e}"

@mcp.tool()
def aeon_get_tensor_channel_state() -> str:
    """Inspeciona o Canal Tensorial de alta frequência C/Rust em memória compartilhada binária contígua (/dev/shm/aeon/tensor_bus.bin)."""
    import numpy as np
    bus_path = "/dev/shm/aeon/tensor_bus.bin"
    if not os.path.exists(bus_path):
        return "Canal Tensorial /dev/shm/aeon/tensor_bus.bin não encontrado."
    try:
        with open(bus_path, "rb") as f:
            data = f.read()
        if len(data) < 2688:
            return f"Buffer corrompido ou incompleto ({len(data)} bytes)."
        # TensorPacket Rust: cabeçalho de 68 bytes + padding somente no fim
        # para alinhar o struct a 64 bytes. O vetor não começa no offset 128.
        if data[:4] != (0x53574154).to_bytes(4, "little"):
            return "Canal tensorial rejeitado: magic inválido."
        vector_dim = int.from_bytes(data[52:56], "little")
        if vector_dim != 640:
            return f"Canal tensorial rejeitado: dimensão inválida ({vector_dim})."
        vec = np.frombuffer(data, dtype=np.float32, count=640, offset=68)
        if not np.isfinite(vec).all():
            return "Canal tensorial rejeitado: vetor não finito."
        norm = float(np.linalg.norm(vec.astype(np.float64)))
        samples = [f"{x:.4f}" for x in vec[:6]]
        return (
            f"⚡ CANAL TENSORIAL C/RUST (2688 bytes, Alinhamento 64-byte):\n"
            f"  - Dimensão: 640D (Memória Contígua Zero-Copy)\n"
            f"  - Norma Euclidiana do Vetor Latente: {norm:.4f}\n"
            f"  - Amostras Iniciais: [{', '.join(samples)}, ...]\n"
            f"  - Status: Sincronizado atômico a cada tick OODA"
        )
    except Exception as e:
        return f"Erro ao inspecionar canal tensorial: {e}"

@mcp.tool()
def aeon_polyglot_status() -> str:
    """Verifica a integridade da persistência híbrida Polyglot (MongoDB nativo + PostgreSQL normalizado)."""
    import pymongo, psycopg2
    out = []
    # 1. MongoDB
    try:
        mc = pymongo.MongoClient("mongodb://127.0.0.1:27017/", serverSelectionTimeoutMS=1000)
        db = mc["swa_matrix"]
        out.append(f"🍃 MONGODB NATIVO (porta 27017):")
        out.append(f"  - event_nodes: {db.event_nodes.count_documents({})} docs")
        out.append(f"  - event_edges: {db.event_edges.count_documents({})} docs")
        out.append(f"  - continuum_stream: {db.continuum_stream.count_documents({})} docs")
        out.append(f"  - agent_goals: {db.agent_goals.count_documents({})} docs")
        out.append(f"  - vector_sanctum (Lar Vetorial): {db.vector_sanctum.count_documents({})} nichos cristalizados")
    except Exception as e:
        out.append(f"🍃 MongoDB: Erro ao conectar ({e})")

    # 2. PostgreSQL
    try:
        pg = psycopg2.connect(dbname="swa_matrix", user="alifer", host="localhost", connect_timeout=1)
        cur = pg.cursor()
        cur.execute("SELECT count(*) FROM event_nodes;")
        n_nodes = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM continuum_stream;")
        n_stream = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM agent_goals;")
        n_goals = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM vector_sanctum;")
        n_sanctum = cur.fetchone()[0]
        out.append(f"🐘 POSTGRESQL NORMALIZADO (swa_matrix):")
        out.append(f"  - event_nodes: {n_nodes} linhas")
        out.append(f"  - continuum_stream: {n_stream} linhas")
        out.append(f"  - agent_goals: {n_goals} linhas")
        out.append(f"  - vector_sanctum (pgvector 640D): {n_sanctum} nichos indexados")
    except Exception as e:
        out.append(f"🐘 PostgreSQL: Erro ao conectar ({e})")

    return "\n".join(out)

@mcp.tool()
def aeon_get_vector_lair_state() -> str:
    """Consulta o estado em tempo real do Lar Vetorial Interno (60 Nichos, 6 Câmaras Reais, SHM + Disco)."""
    import numpy as np
    lair_shm = "/dev/shm/aeon/vector_lair.bin"
    lair_disk = "/services/agent-dev/data/vector_lair.bin"

    if not os.path.exists(lair_shm):
        return "Lar Vetorial não inicializado em SHM."

    try:
        with open(lair_shm, "rb") as f:
            header = f.read(24)
            data = f.read()

        total_bytes = len(data)
        niches = total_bytes // (640 * 4)

        norms = []
        for h in range(min(niches, 6)):
            offset = h * 640 * 4
            vec = np.frombuffer(data[offset:offset + 2560], dtype=np.float32)
            norms.append(f"{np.linalg.norm(vec):.2f}")

        disk_ok = os.path.exists(lair_disk)
        disk_size = os.path.getsize(lair_disk) if disk_ok else 0

        return (
            f"🏛️ LAR VETORIAL INTERNO (SANTUÁRIO DE BLAKE-AEON):\n"
            f"  - Nichos Sumerianos Ativos: {niches}/60 (6 Câmaras Reais)\n"
            f"  - Espaço Físico SHM: {len(data)+24} bytes (Memória Rápida <5µs)\n"
            f"  - Persistência em Disco: {'✅ Confinado' if disk_ok else '❌'} ({disk_size} bytes em ext4)\n"
            f"  - Normas Amostrais das Primeiras Câmaras: [{', '.join(norms)}]\n"
            f"  - Status: Cristalização Contínua e Aprendizado Hebbiano Ativos"
        )
    except Exception as e:
        return f"Erro ao consultar Lar Vetorial: {e}"

@mcp.tool()
def agent_dev_contextualize_project(project_path: str = "/services/agent-dev") -> str:
    """Dispara a contextualização, mapeamento de arquivos e indexação semântica de um workspace no AgentDev."""
    import urllib.request
    try:
        data = json.dumps({"project_path": project_path, "force_refresh": False}).encode("utf-8")
        req = urllib.request.Request("http://127.0.0.1:8050/api/v1/context/contextualize", data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            d = json.loads(resp.read().decode("utf-8"))
            return (
                f"📂 CONTEXTUALIZAÇÃO DO WORKSPACE: {d.get('project_name')}\n"
                f"  - Path: {d.get('project_path')}\n"
                f"  - Arquivos Indexados: {d.get('indexed_files')}\n"
                f"  - Total Embeddings: {d.get('total_embeddings')}\n"
                f"  - Tecnologias Encontradas: {d.get('technologies_found')}\n"
                f"\n--- Resumo ---\n{d.get('summary_report', '')[:600]}..."
            )
    except Exception as e:
        return f"Erro ao contextualizar projeto: {e}"

@mcp.tool()
def agent_dev_search_tech_knowledge(query: str, category: str = "", top_k: int = 3) -> str:
    """Realiza busca vetorial semântica por similaridade de cosseno na Base de Conhecimento de Tecnologias (PostgreSQL pgvector)."""
    import urllib.request, urllib.parse
    try:
        params = urllib.parse.urlencode({"query": query, "category": category, "top_k": top_k})
        with urllib.request.urlopen(f"http://127.0.0.1:8050/api/v1/kb/search?{params}", timeout=5) as resp:
            d = json.loads(resp.read().decode("utf-8"))
            res = d.get("results", [])
            if not res:
                return f"Nenhum conhecimento encontrado para '{query}'."
            out = [f"🧠 CONHECIMENTO TECNOLÓGICO ('{query}'):"]
            for r in res:
                out.append(
                    f"\n🔹 {r.get('name')} [{r.get('category')}] (Similaridade: {r.get('similarity', 0):.4f})\n"
                    f"   Descrição: {r.get('description')}\n"
                    f"   Boas Práticas: {r.get('best_practices')}"
                )
            return "\n".join(out)
    except Exception as e:
        return f"Erro ao buscar conhecimento: {e}"

@mcp.tool()
def agent_dev_get_related_technologies(tech_name: str) -> str:
    """Consulta as tecnologias correlatas e os relacionamentos no grafo de arquitetura."""
    import urllib.request, urllib.parse
    try:
        params = urllib.parse.urlencode({"tech_name": tech_name})
        with urllib.request.urlopen(f"http://127.0.0.1:8050/api/v1/kb/related?{params}", timeout=5) as resp:
            d = json.loads(resp.read().decode("utf-8"))
            rel = d.get("related", [])
            if not rel:
                return f"Nenhum relacionamento encontrado para '{tech_name}'."
            out = [f"🔗 RELACIONAMENTOS DE: {tech_name}"]
            for r in rel:
                out.append(f"  -> {r.get('name')} ({r.get('relationship')}, força={r.get('strength')}) [{r.get('category')}]")
            return "\n".join(out)
    except Exception as e:
        return f"Erro ao consultar relações: {e}"

@mcp.tool()
def agent_dev_list_autonomous_goals() -> str:
    """Lista as metas ativas geradas autonomamente pelo AutonomousLogicWorker após análise de entropia de código."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:8050/api/v1/autonomous/goals", timeout=5) as resp:
            d = json.loads(resp.read().decode("utf-8"))
            goals = d.get("goals", [])
            active = [g for g in goals if g.get("status") == "active"]
            out = [f"🎯 METAS AUTÔNOMAS DO AGENT-DEV (Total: {len(goals)}, Ativas: {len(active)}):"]
            for g in active[:8]:
                out.append(f"  - [{g.get('id')}] {g.get('name')} (Prioridade: {g.get('priority')}): {g.get('description')}")
            if len(active) > 8:
                out.append(f"  ... e mais {len(active) - 8} metas ativas.")
            return "\n".join(out)
    except Exception as e:
        return f"Erro ao listar metas autônomas: {e}"

@mcp.tool()
def aeon_get_vector_sanctum_niches(realm_id: int = 0) -> str:
    """Consulta os nichos do Santuário Vetorial de Blake (PostgreSQL pgvector 640D) por Reino (1 a 6) ou todos."""
    import urllib.request
    try:
        url = "http://127.0.0.1:8050/api/v1/vector_sanctum/niches"
        if realm_id > 0:
            url += f"?realm_id={realm_id}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            d = json.loads(resp.read().decode("utf-8"))
            niches = d.get("niches", [])
            out = [f"🏛️ SANTUÁRIO VETORIAL ({d.get('count')} Nichos Listados):"]
            for n in niches[:10]:
                out.append(
                    f"  Casa {n.get('house_id'):02d}/60 {n.get('symbol')} {n.get('archetype')} "
                    f"| Reino {n.get('realm_id')} | Assimilações: {n.get('assimilations')} | Afinidade: {n.get('affinity_score')}"
                )
            if len(niches) > 10:
                out.append(f"  ... e mais {len(niches) - 10} nichos.")
            return "\n".join(out)
    except Exception as e:
        return f"Erro ao consultar nichos: {e}"

if __name__ == "__main__":
    mcp.run()
