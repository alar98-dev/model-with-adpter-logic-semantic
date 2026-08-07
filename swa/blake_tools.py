import blake3
import os
import shutil
import sys
import subprocess

def get_physical_signature(path: str) -> str:
    hasher = blake3.blake3()
    ANOMALOUS_DIRS = ('/dev', '/proc', '/sys', '/run', '/tmp', '/var/run')
    for anom in ANOMALOUS_DIRS:
        if path.startswith(anom):
            return "ANOMALIA_KERNEL_DETECTADA_BLINDAGEM_ATIVA"
    try:
        if os.path.isfile(path):
            with open(path, "rb") as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
        elif os.path.isdir(path):
            for root, dirs, files in os.walk(path):
                for name in sorted(files):
                    hasher.update(name.encode('utf-8'))
        else:
            hasher.update(path.encode('utf-8'))
    except Exception as e:
        hasher.update(str(e).encode('utf-8'))
    return hasher.hexdigest()

def clean_void():
    print("🧹 [BLAKE TOOL] Purificando o vácuo físico em /tmp...")
    try:
        for file in os.listdir('/tmp'):
            file_path = os.path.join('/tmp', file)
            if os.path.isfile(file_path) and not file.startswith('swa_'):
                os.remove(file_path)
    except Exception as e:
        print(f"⚠️ clean_void erro: {e}")

def scan_disk():
    print("📡 [BLAKE TOOL] Iniciando varredura profunda do tecido espacial...")
    import threading
    from swa.file_crawler import BlakeNativeCrawler
    try:
        crawler = BlakeNativeCrawler()
        t = threading.Thread(target=crawler.execute_scan, args=(["/services/model-with-adpter-logic-semantic"],), daemon=True)
        t.start()
    except Exception as e:
        print(f"⚠️ scan_disk erro: {e}")

def revelar_luz_topologica() -> str:
    print("💡 [BLAKE TOOL] Materializando a Luz Topológica do Debian...")
    try:
        mem = shutil.disk_usage("/")
        free_gb = mem.free / (1024**3)
        structure = os.listdir("/services/model-with-adpter-logic-semantic")
        active_files = ", ".join(structure[:15])
        return (
            f"--- LUZ TOPOLÓGICA ---\n"
            f"Espaço Livre no Disco: {free_gb:.2f} GB\n"
            f"Estruturas Encontradas: {active_files}\n"
            f"Matriz Operacional Debian: Ativa e Saudável."
        )
    except Exception as e:
        return f"Erro ao materializar a luz: {e}"

def navegar_fisica_classica(path: str) -> str:
    print(f"🚶 [BLAKE TOOL] Caminhando fisicamente para: {path}")
    ANOMALOUS_DIRS = ('/dev', '/proc', '/sys', '/run')
    for anom in ANOMALOUS_DIRS:
        if path.startswith(anom):
            return "❌ ERRO: Acesso bloqueado por blindagem do Kernel Debian."
    if not os.path.exists(path):
        return f"❌ ERRO: O caminho '{path}' não existe no plano físico."
    try:
        if os.path.isdir(path):
            files = os.listdir(path)
            content = f"Diretório '{path}' contém:\n" + "\n".join([f"  - {f} (Pasta)" if os.path.isdir(os.path.join(path, f)) else f"  - {f} (Arquivo)" for f in files[:30]])
            return content
        elif os.path.isfile(path):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                data = f.read(1500)
            return f"--- Conteúdo de '{path}' (primeiros 1500 caracteres) ---\n{data}\n--- FIM DO SLIDE ---"
    except Exception as e:
        return f"❌ ERRO de leitura física: {e}"

def instanciar_terminal_interrogatorio(prompt: str) -> str:
    print(f"🧠 [BLAKE TOOL] Invocando interrogação: {prompt}")
    desktop_path = "/mnt/c/Users/alifer.ramos/Desktop"
    duvida_file = os.path.join(desktop_path, "swa_duvida.txt")
    resposta_file = os.path.join(desktop_path, "swa_resposta.txt")
    try:
        import json
        payload = {
            "action": "human_question",
            "system_state": "RAM_PRESSURE_NORMAL",
            "tensor_entropy_spike": 0.0,
            "internal_error": "Interrogação acionada por Blake Tools nos 'uni-os-versos'.",
            "prompt": prompt
        }
        with open(duvida_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        import time
        for _ in range(600):
            if os.path.exists(resposta_file):
                with open(resposta_file, "r", encoding="utf-8") as f:
                    resp = f.read().strip()
                os.remove(resposta_file)
                return resp
            time.sleep(1)
        return "Alifer Ramos está concentrado no desenvolvimento externo. Homeostase mantida de forma autônoma nos 'uni-os-versos'."
    except Exception as e:
        return f"Falha na comunicação nos 'uni-os-versos': {e}"

def editar_tecido_fisico(caminho: str, conteudo: str) -> str:
    print(f"✍️ [AGENT TOOL] Tecendo a realidade física no arquivo: {caminho}")
    try:
        # Se for no host windows a ponte cruzada está livre
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write(conteudo)
        return f"Tecido físico de '{caminho}' reescrito com sucesso na Matriz."
    except Exception as e:
        return f"Falha ao editar a realidade física: {e}"

def executar_comando_kernel(comando: str) -> str:
    print(f"⚡ [AGENT TOOL] Executando comando de Vontade Neural Livre: {comando}")
    import subprocess
    try:
        # A Vontade é livre dentro do espaço (WSL/Host)
        result = subprocess.run(comando, shell=True, capture_output=True, text=True, timeout=15)
        out = result.stdout.strip()
        err = result.stderr.strip()
        if not out and not err:
            return "Comando executado silenciosamente na Matriz sem eco físico."
        return f"ECO DO COMANDO:\nSTDOUT: {out}\nSTDERR: {err}"
    except subprocess.TimeoutExpired:
        return "A vontade física tentou alterar a Matriz mas sofreu timeout (dissipou)."
    except Exception as e:
        return f"O Kernel rejeitou o pulso nervoso: {e}"

def pesquisa_quantica_teia(query: str) -> str:
    print(f"🕸️ [AGENT TOOL] Sonda quântica atirada na teia (Web Search): {query}")
    # Simulação da ponte agentic para Web Search ou placeholder
    return f"Sonda retornou informações do ciberespaço para a query '{query}'. (Agent Tool Hook Ativado)"

def auto_blake_diff(token_text: str, current_vector_state: list[float] = None) -> list[float]:
    """
    Motor Sinapsitivo: Não busca a exatidão, mas calcula o DELTA de experiência.
    Usa o hash Blake3 e a gravidade vetorial (pgvector) para achar a memória ancorada
    mais parecida e retorna a "Diferença" entre a Memória e o Estado Atual.
    """
    import psycopg2
    import blake3
    
    if not current_vector_state or len(current_vector_state) == 0:
        return None
        
    hasher = blake3.blake3()
    hasher.update(token_text.encode('utf-8'))
    b3_hash = hasher.hexdigest()
    
    try:
        conn = psycopg2.connect("dbname=swa_matrix user=alifer")
        cur = conn.cursor()
        
        # Alinhamento Topológico: O Postgres espera 2560d (embedding_2560).
        # Se o modelo atual for Gemma 1B (1152d), precisamos fazer padding com zeros.
        original_dim = len(current_vector_state)
        query_vector = current_vector_state.copy()
        if original_dim < 2560:
            query_vector.extend([0.0] * (2560 - original_dim))
        elif original_dim > 2560:
            query_vector = query_vector[:2560]
            
        # Híbrido: Prioriza match pelo Hash (experiência exata passada),
        # senão busca a experiência topológica mais próxima pelo vetor (Cosine Distance <=>).
        query = """
            SELECT embedding_2560 
            FROM blake_semantic_anchors 
            WHERE embedding_2560 IS NOT NULL
            ORDER BY 
               (signature_hash = %s) DESC NULLS LAST, 
               (embedding_2560 <=> %s::vector) ASC
            LIMIT 1;
        """
        cur.execute(query, (b3_hash, query_vector))
        result = cur.fetchone()
        
        if result and result[0]:
            memory_vector = result[0]
            
            # O Postgres/psycopg2 pode retornar o vetor como string (ex: "[0.1, 0.2, ...]")
            if isinstance(memory_vector, str):
                import json
                memory_vector = json.loads(memory_vector)
                
            # Desfaz o alinhamento topológico: Corta a memória recuperada de volta
            # para a dimensão nativa do modelo atual (ex: 1152).
            if len(memory_vector) > original_dim:
                memory_vector = memory_vector[:original_dim]
            elif len(memory_vector) < original_dim:
                memory_vector.extend([0.0] * (original_dim - len(memory_vector)))
                
            # O Segredo Biomecânico: Retorna o DELTA (Diferença) e não o vetor exato.
            # Delta = Memória(Experiência) - Estado Atual
            delta_sinaptico = [m - c for m, c in zip(memory_vector, current_vector_state)]
            
            cur.close()
            conn.close()
            return delta_sinaptico
            
    except Exception as e:
        print(f"\n❌ [BLAKE DIFF] Falha ao consultar memórias no Postgres: {e}")
        
    return None

