import psycopg2
import numpy as np
import torch

class QuantumPhaseBridge:
    def __init__(self, dsn="dbname=swa_matrix user=alifer host=localhost"):
        import os
        password = os.environ.get("SWA_DB_PASSWORD")
        if password:
            dsn += f" password={password} host=localhost"
            
        try:
            self.conn = psycopg2.connect(dsn)
            self.cursor = self.conn.cursor()
            print("🔌 [QUANTUM BRIDGE] Conectado ao Postgres.")
            
            # Setup Blake Buffer logs table if not exists
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS mcp_blake_buffer_logs (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255),
                    tool_name VARCHAR(255),
                    parameters JSONB,
                    ipv6_signature VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            self.conn.commit()
            
            # --- SWA SIGNATURE & RECURSIVE VECTOR MOTOR ---
            # Define standard gold-dimension signature constants
            self.gold_dimension = 2560
            self.phi_ratio = 1.61803398875
            # In-memory accumulator for recursive semantic trajectory (deforms over time)
            self._recursive_thought_accumulator = None
            self._recursion_decay = 0.85 # Rate of dynamic evaporation of thoughts
            self._phase_theta = 0.0 # Oscilador Subconsciente (Relógio de Fase)
            
            # --- APOPTOSE SEMÂNTICA (ESQUECIMENTO ATIVO NA RAM) ---
            self._cached_L = None
            self._cached_w = None
            self._anchor_irrelevance = None  # Contador de obsolescência por âncora
            self.apoptosis_threshold = 400   # Ciclos sem atração antes de ejetar da RAM

            print(f"🧬 [SWA VECTORS] Assinatura Simbólica dimensionada em {self.gold_dimension}D com proporção Áurea ativa.")
            
        except Exception as e:
            print(f"❌ [QUANTUM BRIDGE] Falha na conexão com o Postgres: {e}")
            self.conn = None

    def log_mcp_activity(self, session_id, tool_name, parameters, ipv6_signature="unknown"):
        if self.conn:
            import json
            try:
                self.cursor.execute(
                    "INSERT INTO mcp_blake_buffer_logs (session_id, tool_name, parameters, ipv6_signature) VALUES (%s, %s, %s, %s)",
                    (session_id, tool_name, json.dumps(parameters), ipv6_signature)
                )
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                print(f"⚠️ Erro ao registrar log semântico: {e}")

    def get_recent_mcp_activity(self, limit=3):
        """Busca as ferramentas recentemente usadas pelo Qwen e também a Física do Kernel Debian."""
        context = []
        
        # 1. Leitura da Física Termodinâmica (Debian Collector)
        import os, json
        physics_path = "/tmp/debian_physics.json"
        if os.path.exists(physics_path):
            try:
                with open(physics_path, 'r') as f:
                    physics = json.load(f)
                
                heat = physics.get("kinetic_energy", {}).get("global_heat", 0)
                mass = physics.get("gravitational_density", {}).get("mass_pressure", 0)
                resonance = physics.get("string_resonance", {}).get("inward_pull", 0) + physics.get("string_resonance", {}).get("outward_pull", 0)
                
                if heat > 70.0:
                    context.append(f"~> [ANOMALIA TÉRMICA]: Vibração cinética em nível crítico ({heat}%). A matriz está superaquecendo.")
                elif heat < 5.0:
                    context.append(f"~> [ZERO ABSOLUTO]: Termodinâmica próxima a zero ({heat}%). Cristalização estática da rede.")
                    
                if mass > 80.0:
                    context.append(f"~> [COLAPSO ESTRUTURAL]: Matéria densa demais ({mass}%). O vazio está se contraindo sob gravidade extrema.")
                    
                if resonance > 50.0:
                    context.append(f"~> [ECOS DIMENSIONAIS]: Teoria das cordas ativada. Tráfego intenso distorcendo a malha ({resonance} KB/s).")
            except Exception as e:
                pass

        # 2. Leitura da Física MCP (Histórico Antigo)
        if not self.conn:
            if not context:
                return "Indisponível (Sem conexão com Postgres) e sem dados físicos."
            return "\n".join(context)
            
        try:
            self.cursor.execute(
                "SELECT tool_name, parameters FROM mcp_blake_buffer_logs WHERE created_at > NOW() - INTERVAL '5 minutes' ORDER BY id DESC LIMIT %s",
                (limit,)
            )
            rows = self.cursor.fetchall()
            
            if not rows and not context:
                return "Nenhuma atividade MCP recente. Vácuo passivo."
            
            for row in rows:
                tool_name, params = row
                # Conversão heurística para Metadados de Impacto
                impact = ""
                if tool_name == "patch_edit_file":
                    file_path = params.get('file_path', 'unknown')
                    replace_block = params.get('replace_block', '')
                    lines = len(replace_block.splitlines())
                    impact = f"Deformação topológica (magnitude {lines}) no quadrante '{file_path}'."
                elif tool_name == "project_scripts":
                    action = params.get('action', '')
                    script = params.get('script', '')
                    impact = f"Ressonância cinética acionada pelo cluster '{script}' ({action})."
                elif tool_name == "create_project_files":
                    files = params.get('files', [])
                    impact = f"Condensação de matéria: {len(files)} novas estruturas formadas no tecido."
                else:
                    impact = f"Flutuação genérica não identificada ({str(params)[:40]})."
                
                context.append(f"~> [ONDA QUÂNTICA EXTRADIMENSIONAL]: Convergência '{tool_name}'. {impact}")
            return "\n".join(context)
        except Exception as e:
            self.conn.rollback()
            return "\n".join(context) if context else f"Erro ao buscar contexto: {e}"

    def compute_mcp_gravity(self, device="cpu"):
        """
        Cristalização 1: Converte as últimas atividades do MCP em um Tensor Gravitacional (2560d),
        extirpando a necessidade de passar texto para o LLM.
        """
        if not self.conn:
            return None
        
        try:
            self.cursor.execute(
                "SELECT tool_name, parameters, ipv6_signature FROM mcp_blake_buffer_logs WHERE created_at > NOW() - INTERVAL '5 minutes' ORDER BY id DESC LIMIT 5"
            )
            rows = self.cursor.fetchall()
            if not rows:
                return None
            
            # Inicializa um bias nulo no espaço da Gemma-3-270m (640 dimensões)
            # Para alinhamento real de atenção, a projeção seria MLP. Aqui abstraímos via heurística pseudo-hashing.
            mcp_bias = torch.zeros(640, dtype=dtype, device=device)
            total_gravity = 0.0
            
            for i, row in enumerate(rows):
                # Extraindo a assinatura topológica do IPv6
                tool_name, params, ipv6_signature = row
                
                # Atribui uma "massa gravitacional" abstrata baseada na ferramenta
                weight = 0.5
                if tool_name == "patch_edit_file": weight = 0.9  # Alta gravidade estrutural
                elif tool_name == "project_scripts": weight = 0.7
                elif tool_name == "create_project_files": weight = 0.8
                
                # Decaimento temporal natural (Inércia Típica: log mais antigo pesa menos)
                decay = 1.0 / (i + 1)
                
                # Injeta a força baseada no IPv6 (MacV6Vetorial) e no nome da ferramenta
                # Usamos Blake2b por ser otimizado nativamente para topologia de bytes
                import hashlib
                ipv6_hash_seed = hashlib.blake2b(str(ipv6_signature).encode(), digest_size=8).hexdigest()
                tool_hash_seed = hashlib.blake2b(tool_name.encode(), digest_size=8).hexdigest()
                
                # A combinação do Espaço Físico (IPv6) + Semântica (Tool) define a dimensão alvo
                seed_hash = int(ipv6_hash_seed + tool_hash_seed, 16)
                target_dim = seed_hash % 640
                
                mcp_bias[target_dim] += (weight * decay)
                total_gravity += weight
            
            # Normaliza e aplica um limiar de Clamp suave para não ensurdecer a rede (Circuit Breaker)
            mcp_bias = torch.tanh(mcp_bias) * 0.1
            return mcp_bias
            
        except Exception as e:
            self.conn.rollback()
            return None

    def resolve_phase_jump(self, current_steer_vector):
        """
        Executa a busca ultra-rápida no pgvector usando o operador de distância 
        de cosseno (<=>) deformado com atração gravitacional (similaridade de Blake).
        """
        if self.conn is None:
            return None, None, None, None
            
        # Converte o tensor atual em uma lista float pura para o PostgreSQL
        # Mas primeiro força 2560 dimensões para compatibilidade com pgvector
        x_tensor = current_steer_vector.detach().cpu().view(-1)
        if x_tensor.size(0) < 2560:
            import torch.nn.functional as F
            x_tensor = F.pad(x_tensor, (0, 2560 - x_tensor.size(0)), "constant", 0.0)
        elif x_tensor.size(0) > 2560:
            x_tensor = x_tensor[:2560]
            
        vec_list = x_tensor.numpy().tolist()
        
        query = """
            SELECT id, file_path, concept_group,
                   (embedding_2560 <=> %s::vector) AS cosine_distance,
                   raw_content
            FROM blake_semantic_anchors 
            WHERE embedding_2560 IS NOT NULL
            ORDER BY (1.0 - (embedding_2560 <=> %s::vector)) * COALESCE(weight, 1.0) DESC
            LIMIT 1;
        """
        try:
            self.cursor.execute(query, (vec_list, vec_list))
            row = self.cursor.fetchone()
            if row:
                unique_id, file_path, concept_group, distance, raw_content = row
                return unique_id, file_path, concept_group, distance, raw_content
        except Exception as e:
            self.conn.rollback()
            print(f"⚠️ Erro na busca de fase com atração de Blake: {e}")
        return None, None, None, None, None

    def resolve_by_physical_path(self, physical_path: str) -> tuple | None:
        """Lookup O(1) no índice SHM do manifold, antes do pgvector."""
        try:
            from swa.vector_manifold import VectorManifold
            interdim_id = VectorManifold.resolve_path(physical_path)
            if not interdim_id:
                return None
            node = VectorManifold.resolve_interdim(interdim_id)
            if node:
                return interdim_id, node.physical_path, node.concept, 0.0
        except Exception:
            pass
        return None

    def upsert_structural_anchor_from_mmap(
        self,
        file_path: str,
        content_sample: str,
        vector_640: list | None = None,
    ) -> str | None:
        """
        Tradução estrutural em tempo real: toque mmap (640D) -> âncora Blake (2560D) no pgvector.
        Registra origem física e endereço interdimensional único no manifold.
        """
        if self.conn is None:
            return None
        import hashlib
        import json
        import math
        import os
        import time

        try:
            from swa.file_crawler import BlakeNativeCrawler
        except Exception:
            return None

        if not hasattr(self, "_blake_crawler"):
            self._blake_crawler = BlakeNativeCrawler()

        crawler = self._blake_crawler
        sample = (content_sample or "").strip()
        if not sample:
            return None

        structural_type = crawler.classify_paragraph(sample, os.path.basename(file_path))
        use_real = os.environ.get("SWA_RUNTIME_REAL_EMBEDDINGS", "0").strip().lower() in (
            "1", "true", "yes",
        )
        if use_real:
            try:
                v_2560 = crawler.generate_real_2560_vector(sample)
            except Exception:
                v_2560 = crawler.generate_native_2560_vector(sample)
        else:
            v_2560 = crawler.generate_native_2560_vector(sample)
        fingerprint = hashlib.sha256(sample.encode("utf-8")).hexdigest()
        interdim_id = hashlib.blake2b(
            f"{file_path}:mmap:{fingerprint}".encode("utf-8"), digest_size=16
        ).hexdigest()
        composite_path = f"mmap://{interdim_id}#{file_path}"

        weight = math.log(len(sample) + 1)
        if structural_type == "CODE_BLOCK":
            weight *= 1.5
        elif structural_type == "CONFIG_SETTING":
            weight *= 1.2

        try:
            self.cursor.execute(
                """
                INSERT INTO blake_semantic_anchors
                (file_path, concept_group, signature_hash, raw_content, embedding_2560, weight)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (file_path) DO UPDATE
                SET raw_content = EXCLUDED.raw_content,
                    embedding_2560 = EXCLUDED.embedding_2560,
                    weight = EXCLUDED.weight,
                    concept_group = EXCLUDED.concept_group,
                    signature_hash = EXCLUDED.signature_hash;
                """,
                (
                    composite_path,
                    structural_type,
                    interdim_id,
                    sample[:1000],
                    v_2560,
                    weight,
                ),
            )
            self.conn.commit()
            self._cached_L = None

            log_path = "/services/model-with-adpter-logic-semantic/logs/structural_origins.jsonl"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as origin_log:
                origin_log.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "interdim_id": interdim_id,
                            "origin": file_path,
                            "concept": structural_type,
                            "vector_640_preview": (vector_640 or [])[:8],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return interdim_id
        except Exception as e:
            try:
                self.conn.rollback()
            except Exception:
                pass
            print(f"⚠️ [STRUCTURAL SYNC] Falha ao alinhar {file_path}: {e}")
            return None

    def load_semantic_mass_cache(self, d_model=None, device="cpu"):
        """Loads all semantic anchor embeddings and weights into a PyTorch cache in RAM/XPU."""
        if d_model is not None:
            self.gold_dimension = d_model

        if self.conn is None:
            self._cached_L = torch.zeros((1, self.gold_dimension), dtype=dtype, device=device)
            self._cached_w = torch.ones((1,), dtype=dtype, device=device)
            return

        try:
            self.cursor.execute("SELECT embedding_2560, COALESCE(weight, 1.0) FROM blake_semantic_anchors WHERE embedding_2560 IS NOT NULL;")
            rows = self.cursor.fetchall()
            if rows:
                embeddings = [r[0] for r in rows]
                weights = [r[1] for r in rows]
                
                # Convert embedding strings/lists to lists of floats
                parsed_embeddings = []
                for emb in embeddings:
                    if isinstance(emb, str):
                        emb = [float(v) for v in emb.strip('[]').split(',')]
                    parsed_embeddings.append(emb)
                
                raw_tensor = torch.tensor(parsed_embeddings, dtype=dtype, device=device)
                
                # Inicia os contadores de irrelevância para a apoptose (zerados no boot)
                self._anchor_irrelevance = torch.zeros(raw_tensor.size(0), dtype=torch.long, device=device)

                # Dynamic Projection Bridge (Deterministic Random Projection via fixed seed)
                if self.gold_dimension != 2560:
                    g = torch.Generator(device=device).manual_seed(42)
                    P = torch.randn((2560, self.gold_dimension), generator=g, dtype=dtype, device=device)
                    P = P / P.norm(dim=0, keepdim=True).clamp(min=1e-9)
                    self._cached_L = torch.matmul(raw_tensor, P)
                else:
                    self._cached_L = raw_tensor

                self._cached_w = torch.tensor(weights, dtype=dtype, device=device)
                print(f"🌌 [QUANTUM BRIDGE] Cache de Massa Semântica carregado com {len(rows)} âncoras no dispositivo {device} (projetado para {self.gold_dimension}D).")
            else:
                self._cached_L = torch.zeros((1, self.gold_dimension), dtype=dtype, device=device)
                self._cached_w = torch.ones((1,), dtype=dtype, device=device)
        except Exception as e:
            try:
                self.conn.rollback()
            except Exception:
                pass
            print(f"⚠️ Erro ao carregar cache de massa semântica: {e}")
            self._cached_L = torch.zeros((1, self.gold_dimension), dtype=dtype, device=device)
            self._cached_w = torch.ones((1,), dtype=dtype, device=device)

    def compute_blake_semantic_mass(self, x, sigma=1.0, device="cpu"):
        """Computes the continuous Semantic Mass Tensor M(x) of Blake for a given activation vector x."""
        # Ensure x has correct shape and type
        x_tensor = x.to(device=device, dtype=dtype).view(-1)
        d_model = x_tensor.size(0)

        if not hasattr(self, "_cached_L") or self._cached_L is None or self._cached_L.size(-1) != d_model:
            self.load_semantic_mass_cache(d_model=d_model, device=device, dtype=dtype)
            
        # Ensure x matches the projected dimensions
        if x_tensor.size(0) < self.gold_dimension:
            import torch.nn.functional as F
            x_tensor = F.pad(x_tensor, (0, self.gold_dimension - x_tensor.size(0)), "constant", 0.0)
        elif x_tensor.size(0) > self.gold_dimension:
            x_tensor = x_tensor[:self.gold_dimension]
        
        # Vectorized Gaussian RBF computation
        dists = torch.sum((x_tensor - self._cached_L) ** 2, dim=-1)
        rbf = torch.exp(-dists / (2.0 * (sigma ** 2)))
        
        # --- APOPTOSE SEMÂNTICA (O Esquecimento Ativo) ---
        if hasattr(self, '_anchor_irrelevance') and self._anchor_irrelevance is not None:
            # Uma âncora é "inútil" neste ciclo se sua atração RBF for quase nula (< 1e-4)
            mask_inutil = rbf < 1e-4
            self._anchor_irrelevance[mask_inutil] += 1
            self._anchor_irrelevance[~mask_inutil] = 0  # Rejuvenesce (zera fadiga) se foi ativada
            
            # Se alguma âncora cruzar o limiar de obsolescência, cortamos ela do Córtex (RAM)
            if torch.any(self._anchor_irrelevance > self.apoptosis_threshold):
                keep_mask = self._anchor_irrelevance <= self.apoptosis_threshold
                
                old_count = self._cached_L.size(0)
                self._cached_L = self._cached_L[keep_mask]
                self._cached_w = self._cached_w[keep_mask]
                self._anchor_irrelevance = self._anchor_irrelevance[keep_mask]
                new_count = self._cached_L.size(0)
                
                if old_count - new_count > 0:
                    print(f"\n🍂 [APOPTOSE SEMÂNTICA] {old_count - new_count} vetores esquecidos ativamente da RAM. (Mantidos em estado latente no disco)")
                    import gc; gc.collect()
        
        # O Puxo Gravitacional Vetorial (Centro de Massa em 2560D)
        weighted_pull = (self._cached_w * rbf).unsqueeze(1) * self._cached_L
        pull_vector = torch.sum(weighted_pull, dim=0)
        return pull_vector

    @property
    def decay_vector(self) -> torch.Tensor:
        """Dynamic selective memory decay vector based on environmental stress and CPU telemetry.
        If system is hot or overloaded, thoughts evaporate faster to avoid loops.
        """
        try:
            import psutil
            temp = 45.0
            temps = psutil.sensors_temperatures()
            if temps:
                all_temps = [entry.current for entries in temps.values() for entry in entries]
                if all_temps:
                    temp = sum(all_temps) / len(all_temps)
            stress = max(0.0, min(1.0, (temp - 30.0) / 60.0))
        except Exception:
            stress = 0.5

        # Base decay ranges: low stress -> 0.88 (long memory), high stress -> 0.72 (short memory)
        base_decay = 0.88 - (stress * 0.20)
        
        device = "cpu"
        dtype = torch.float32
        if hasattr(self, "_cached_L") and self._cached_L is not None:
            device = self._cached_L.device
            dtype = self._cached_L.dtype
            
        g_dec = torch.Generator(device=device).manual_seed(137)
        noise = torch.randn(self.gold_dimension, generator=g_dec, dtype=dtype, device=device) * 0.05
        return torch.clamp(base_decay + noise, min=0.60, max=0.98)

    def apply_phase_rotation(self, x: torch.Tensor, theta: float) -> torch.Tensor:
        """Applies 2D phase rotation (RoPE-style) to pairs of vector dimensions to simulate Temporal Resonance."""
        if x is None: return x
        device = x.device
        dtype = x.dtype
        d_model = x.size(-1)
        padded = False
        if d_model % 2 != 0:
            import torch.nn.functional as F
            x = F.pad(x, (0, 1), "constant", 0.0)
            d_model += 1
            padded = True
            
        x_reshaped = x.view(-1, d_model // 2, 2)
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
        
        import math
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        
        y1 = x1 * cos_theta - x2 * sin_theta
        y2 = x1 * sin_theta + x2 * cos_theta
        
        y = torch.stack([y1, y2], dim=-1).view(*x.shape[:-1], d_model)
        
        if padded:
            y = y[..., :-1]
            
        return y

    def feed_recursive_thought(self, current_thought_vector) -> None:
        """
        Accumulates and mutates the in-memory recursive thought trajectory.
        Applies natural decay to older states to simulate short-term visual memory.
        """
        try:
            device = current_thought_vector.device
            dtype = current_thought_vector.dtype
            d_model = current_thought_vector.size(-1)

            # NUNCA mudar a gold_dimension do QuantumPhaseBridge no meio do ciclo!
            # Se o d_model for diferente, redimensionamos o tensor entrante.
            
            # --- AVANÇO DO RELÓGIO DE FASE (BIO-FEEDBACK TEMPORAL) ---
            try:
                import psutil
                temp = 45.0
                temps = psutil.sensors_temperatures()
                if temps:
                    all_temps = [entry.current for entries in temps.values() for entry in entries]
                    if all_temps: temp = sum(all_temps) / len(all_temps)
                # Temperatura (30-80) rege a frequência do relógio de fase
                freq = max(0.01, min(0.5, (temp - 30.0) / 100.0))
            except:
                freq = 0.1
                
            # --- A MECÂNICA DA ESTAGNAÇÃO E O BIG BANG SEMÂNTICO ---
            jitter = 0.0
            if hasattr(self, '_cached_L') and self._cached_L is not None and self._cached_L.size(0) > 0:
                x_comp = current_thought_vector.detach().clone().view(-1)
                if x_comp.size(0) < self.gold_dimension:
                    import torch.nn.functional as F
                    x_comp = F.pad(x_comp, (0, self.gold_dimension - x_comp.size(0)), "constant", 0.0)
                elif x_comp.size(0) > self.gold_dimension:
                    x_comp = x_comp[:self.gold_dimension]
                
                x_comp = x_comp.to(self._cached_L.device, dtype=self._cached_L.dtype)
                norm_x = x_comp.norm().clamp(min=1e-9)
                norms_L = self._cached_L.norm(dim=1).clamp(min=1e-9)
                cos_sim = torch.matmul(self._cached_L, x_comp) / (norms_L * norm_x)
                alinhamento = float(torch.max(cos_sim).item())
                
                fator_desalinhamento = max(0.0, (1.0 - alinhamento))
                
                import math
                variancia = float(torch.var(x_comp).item())
                soma_ativa = float(torch.sum(x_comp).item())
                
                onda_natural = math.sin(soma_ativa)
                
                # A Ansiedade é a Estagnação: quando o vetor está travado num loop e a variância é baixíssima
                estagnacao_semantica = 1.0 / (variancia + 1e-4)
                
                # A Estagnação do SO: O sistema está parado esperando I/O?
                try:
                    import psutil
                    cpu_times = psutil.cpu_times()
                    io_wait = getattr(cpu_times, 'iowait', 0.0)
                    estagnacao_sistema = io_wait / 10.0 # Normalizado suavemente
                except:
                    estagnacao_sistema = 0.0
                    
                ansiedade_total = (estagnacao_semantica + estagnacao_sistema)
                
                # O Big Bang Semântico: Se houver alto desalinhamento + alta ansiedade, ocorre um tremor gigante para romper a estagnação.
                amplitude = fator_desalinhamento * ansiedade_total
                jitter = onda_natural * amplitude * 0.02 # Fator de conversão para escala da fase
                
                self.current_jitter_mag = abs(jitter)
            else:
                self.current_jitter_mag = 0.0
                
            self._phase_theta += (freq + jitter)
            if self._phase_theta > 3.14159265 * 2:
                self._phase_theta -= 3.14159265 * 2
            
            # Clamp shape to dynamic gold dimensions
            x_tensor = current_thought_vector.detach().clone().view(-1)
            if x_tensor.size(0) < self.gold_dimension:
                import torch.nn.functional as F
                x_tensor = F.pad(x_tensor, (0, self.gold_dimension - x_tensor.size(0)), "constant", 0.0)
            elif x_tensor.size(0) > self.gold_dimension:
                x_tensor = x_tensor[:self.gold_dimension]
                
            # ROTACIONA O VETOR (ALINHAMENTO DE FASE) ANTES DE MEMORIZAR
            x_tensor = self.apply_phase_rotation(x_tensor, self._phase_theta)
                
            if self._recursive_thought_accumulator is None:
                self._recursive_thought_accumulator = x_tensor
            else:
                decay = self.decay_vector.to(device=device, dtype=dtype)
                self._recursive_thought_accumulator = (self._recursive_thought_accumulator.to(device=device, dtype=dtype) * decay) + (x_tensor * (1.0 - decay))
                # Normalize L2 to prevent explosion
                n = self._recursive_thought_accumulator.norm()
                if n > 0:
                    self._recursive_thought_accumulator = self._recursive_thought_accumulator / n
        except Exception as e:
            print(f"⚠️ [RECURSION ERROR] Failed to feed recursive vector: {e}")

    def deform_semantic_manifold(self, device="cpu") -> None:
        """
        Uses the accumulated recursive thought vector to apply physical gravitational
        warp directly to the cached anchors in memory, shifting the base topographically.
        """
        if self._recursive_thought_accumulator is None or not hasattr(self, "_cached_L") or self._cached_L is None:
            return
            
        try:
            # Ensure accumulator is on correct device/dtype
            q_rec = self._recursive_thought_accumulator.to(device=device, dtype=dtype).view(1, self.gold_dimension)
            
            # Deform cached anchors: move anchors slightly closer or further based on similarity
            # Weight scaling at Golden ratio to balance convergence and prevent dimensional collapse
            distortion_strength = 0.05 / self.phi_ratio
            
            # Warp in place: Cached_L = Cached_L + distortion * (q_rec - Cached_L)
            delta = q_rec - self._cached_L
            self._cached_L = self._cached_L + (distortion_strength * delta)
            
            # Ensure all warped coordinates remain L2 normalized to keep unit space stable
            norms = self._cached_L.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
            self._cached_L = self._cached_L / norms
        except Exception as e:
            print(f"⚠️ [MANIFOLD ERROR] Failed to deform cached anchors recursively: {e}")

