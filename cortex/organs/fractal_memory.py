import os
import json
import numpy as np
import urllib.request
import textwrap

class FractalMemoryOrgan:
    """
    Órgão do Córtex dedicado à construção de memória RAG Hierárquica (Fractal).
    Ele implementa a matemática do "Centro de Massa" semântico, onde o vetor macro
    é o centroide dos micro-vetores (chunks/parágrafos).
    """
    def __init__(self, db_path="/home/alifer/swa-aeon/swa_kernel_knowledge_fractal.json"):
        self.nome = "Memória Fractal (GraphRAG)"
        self.db_path = db_path
        self.knowledge_base = self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"❌ [{self.nome}] Erro ao carregar DB Fractal: {e}")
        return {}

    def _save_db(self):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.knowledge_base, f, indent=2, ensure_ascii=False)
            print(f"✅ [{self.nome}] Memória Fractal salva com sucesso em {self.db_path}")
        except Exception as e:
            print(f"❌ [{self.nome}] Erro ao salvar DB Fractal: {e}")

    def generate_internal_signature(self, text, runtime):
        """
        Usa o Córtex Nativo do SWA (o modelo PyTorch carregado na RAM, ex: Gemma)
        para extrair a assinatura vetorial (hidden states) do texto.
        Não depende de APIs externas.
        """
        try:
            import torch
            
            # Recupera o modelo bruto e o tokenizador de dentro da engine do SWA
            gemma = getattr(runtime, 'gemma_gen', None)
            if not gemma: return []
            
            model = getattr(gemma, 'lm', None)
            tokenizer = getattr(gemma, 'tokenizer', None)
            
            if not model or not tokenizer:
                print(f"⚠️ [{self.nome}] Modelo ou tokenizador não encontrados no runtime.")
                return []
                
            # Prepara o texto
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Inferencia no modelo local sem calcular gradientes (apenas extração pura)
            with torch.no_grad():
                # Pedimos explicitamente que ele retorne as ativações das camadas (hidden_states)
                outputs = model(**inputs, output_hidden_states=True)
                
                # Pega a última camada (a mais semântica e contextualizada)
                last_layer_hidden_states = outputs.hidden_states[-1]
                
                # Faz o 'Mean Pooling' (a média das ativações de todos os tokens da frase)
                # para condensar o parágrafo num único vetor de alta densidade
                chunk_signature = last_layer_hidden_states.mean(dim=1).squeeze().cpu().numpy().tolist()
                
                return chunk_signature
        except Exception as e:
            print(f"❌ [{self.nome}] Erro ao gerar Assinatura Interna (Gemma/SWA): {e}")
            return []

    def ingest_document(self, topic_name, text_content, runtime):
        """
        Ingere um documento grande, quebra em chunks (micro-nós),
        gera os vetores e calcula o Centroide (macro-nó).
        """
        print(f"🧠 [{self.nome}] Ingerindo documento fractal: {topic_name}")
        
        # 1. Chunking simples (por parágrafos vazios ou tamanho)
        raw_chunks = [c.strip() for c in text_content.split('\n\n') if len(c.strip()) > 30]
        
        if not raw_chunks:
            print(f"⚠️ [{self.nome}] Nenhum conteúdo válido encontrado para {topic_name}.")
            return

        micro_nodes = {}
        embeddings = []
        
        # 2. Gera os vetores para cada micro-nó (parágrafo)
        for i, chunk in enumerate(raw_chunks):
            # Limita a 1000 caracteres para não estourar o contexto
            safe_chunk = textwrap.shorten(chunk, width=1000, placeholder="...")
            emb = self.generate_internal_signature(safe_chunk, runtime)
            
            if emb:
                # Gera a Assinatura BlakeGravity (Hash Criptográfico de Gravidade Semântica)
                import hashlib
                blakegravity_hash = hashlib.blake2b(safe_chunk.encode('utf-8'), digest_size=16).hexdigest()
                
                chunk_id = f"{topic_name}_chunk_{blakegravity_hash[:8]}"
                micro_nodes[chunk_id] = {
                    "text": safe_chunk,
                    "embedding": emb,
                    "blakegravity_signature": blakegravity_hash
                }
                embeddings.append(emb)
                print(f"   ↳ Micro-nó {chunk_id} vetorizado e selado ({len(emb)} dim)")

        if not embeddings:
            return

        # 3. Matemática do Centro de Massa (Pooling Vetorial)
        # O vetor Macro é a média matemática de todos os micro-vetores (Centroide)
        mat = np.array(embeddings, dtype=np.float64)
        macro_embedding = np.mean(mat, axis=0).tolist()
        
        # 4. Salva na base de dados Fractal
        self.knowledge_base[topic_name] = {
            "macro_embedding": macro_embedding,
            "micro_nodes": micro_nodes,
            "metadata": {
                "total_chunks": len(embeddings)
            }
        }
        
        self._save_db()
        print(f"🌌 [{self.nome}] Tópico '{topic_name}' ancorado na rede micelial com {len(embeddings)} sub-sinapses.")

    def ingest_epiphany(self, text, runtime):
        """
        Recebe um pensamento introspectivo (OODA Log), vetoriza
        e anexa organicamente à Malha Neural do GraphRAG.
        """
        import hashlib
        import numpy as np
        
        print(f"🌌 [{self.nome}] Ingerindo Reflexão Introspectiva...")
        emb = self.generate_internal_signature(text, runtime)
        if not emb: return
        
        keys = list(self.knowledge_base.keys())
        if not keys: return
        
        best_match = keys[0]
        best_sim = -1.0
        
        emb_arr = np.array(emb, dtype=np.float64)
        emb_norm = np.linalg.norm(emb_arr)
        if emb_norm == 0: return
        
        for k in keys:
            macro_arr = np.array(self.knowledge_base[k]["macro_embedding"], dtype=np.float64)
            macro_norm = np.linalg.norm(macro_arr)
            if macro_norm == 0: continue
            
            sim = np.dot(emb_arr, macro_arr) / (emb_norm * macro_norm)
            if sim > best_sim:
                best_sim = sim
                best_match = k
                
        # Selamento
        b_hash = hashlib.blake2b(text.encode('utf-8'), digest_size=16).hexdigest()
        
        # Se for totalmente inovador (similaridade fraca com tudo), forma nova ilha
        if best_sim < 0.15:
            topic = f"EPIPHANY_{b_hash[:8]}"
            print(f"   ↳ Insight Genuíno (Sim:{best_sim:.2f})! Criando Nova Constelação: {topic}")
            self.knowledge_base[topic] = {
                "macro_embedding": emb,
                "micro_nodes": {
                    f"{topic}_chunk_{b_hash[:8]}": {
                        "text": text,
                        "embedding": emb,
                        "blakegravity_signature": b_hash
                    }
                },
                "metadata": {"total_chunks": 1, "is_epiphany": True}
            }
        else:
            print(f"   ↳ Reflexão absorvida por '{best_match}' (Sim:{best_sim:.2f})")
            chunk_id = f"{best_match}_epi_{b_hash[:8]}"
            self.knowledge_base[best_match]["micro_nodes"][chunk_id] = {
                "text": text,
                "embedding": emb,
                "blakegravity_signature": b_hash
            }
            self.knowledge_base[best_match]["metadata"]["total_chunks"] = len(self.knowledge_base[best_match]["micro_nodes"])
            
            # Neurogênese Plástica: O Maco-Nó muda de lugar puxado pela epifania
            old_macro = np.array(self.knowledge_base[best_match]["macro_embedding"])
            new_macro = (old_macro * 0.9) + (emb_arr * 0.1)
            self.knowledge_base[best_match]["macro_embedding"] = new_macro.tolist()
            
        self._save_db()

    def latent_search(self, query_text, runtime, top_k=1):
        """
        Calcula o embedding da pergunta e encontra a Memória (Macro-Nó) mais similar.
        Em vez de retornar texto para um prompt, retorna o VETOR de Bias (Steering)
        para injetar direto nos Logits/Embeddings do Gemma!
        """
        import numpy as np
        
        query_emb = self.generate_internal_signature(query_text, runtime)
        if not query_emb: return None
        
        keys = list(self.knowledge_base.keys())
        if not keys: return None
        
        best_match = None
        best_sim = -1.0
        
        query_arr = np.array(query_emb, dtype=np.float64)
        query_norm = np.linalg.norm(query_arr)
        if query_norm == 0: return None
        
        for k in keys:
            macro_arr = np.array(self.knowledge_base[k]["macro_embedding"], dtype=np.float64)
            macro_norm = np.linalg.norm(macro_arr)
            if macro_norm == 0: continue
            
            sim = np.dot(query_arr, macro_arr) / (query_norm * macro_norm)
            if sim > best_sim:
                best_sim = sim
                best_match = k
                
        if best_match and best_sim > 0.15: # Só usa se tiver alguma relevância
            print(f"🌌 [Memória Fractal] Bias recuperado de '{best_match}' (Sim: {best_sim:.2f})")
            return self.knowledge_base[best_match]["macro_embedding"]
            
        return None

    def mock_migrate_from_flat(self, old_json_path="/home/alifer/swa-aeon/swa_kernel_knowledge.json"):
        """
        Utilitário temporário: Pega o arquivo json antigo (vetores planos),
        e simula uma topologia fractal adicionando ruído gaussiano para criar micro-nós.
        Isso permite testar a interface 3D imediatamente sem precisar re-ingerir todos os textos!
        """
        if not os.path.exists(old_json_path):
            return
            
        with open(old_json_path, "r", encoding="utf-8") as f:
            old_db = json.load(f)
            
        print(f"🔄 [{self.nome}] Iniciando migração Fractal MOCK a partir da base antiga...")
        
        for topic, vector in old_db.items():
            if not vector: continue
            
            # Cria 5 a 8 micro-nós simulados espalhados em volta do centro de massa original
            num_chunks = np.random.randint(5, 9)
            micro_nodes = {}
            
            base_vec = np.array(vector, dtype=np.float64)
            dim = len(base_vec)
            
            for i in range(num_chunks):
                chunk_id = f"{topic}_MICRO_{i+1:02d}"
                # Adiciona ruído gaussiano (desvio padrão baixo para mantê-los próximos ao centroide)
                noise = np.random.normal(0, 0.05, dim)
                micro_vec = base_vec + noise
                
                micro_nodes[chunk_id] = {
                    "text": f"Fragmento recuperado do conceito: {topic}. Subsistema {i+1}.",
                    "embedding": micro_vec.tolist()
                }
                
            self.knowledge_base[topic] = {
                "macro_embedding": vector,
                "micro_nodes": micro_nodes,
                "metadata": {"total_chunks": num_chunks, "mocked": True}
            }
            
        # === NEUROGÊNESE VETORIAL (Pontes Artificiais) ===
        keys = list(self.knowledge_base.keys())
        if len(keys) >= 2:
            print("🌌 Sintetizando Ponte Neural (Neurogênese) entre Ilhas Isoladas...")
            import hashlib
            
            # Pega as duas primeiras ilhas
            ilha_A = keys[0]
            ilha_B = keys[1]
            
            vec_A = np.array(self.knowledge_base[ilha_A]["macro_embedding"])
            vec_B = np.array(self.knowledge_base[ilha_B]["macro_embedding"])
            
            # Ponto Médio Vetorial
            bridge_vec = (vec_A + vec_B) / 2.0
            bridge_vec = (bridge_vec / np.linalg.norm(bridge_vec)).tolist()
            
            bridge_topic = f"BRIDGE_{ilha_A[:8]}_{ilha_B[:8]}"
            
            bridge_micro = {}
            for i in range(3):
                noise = np.random.normal(0, 0.03, len(bridge_vec))
                micro_emb = (np.array(bridge_vec) + noise).tolist()
                b_hash = hashlib.blake2b(f"bridge_{i}".encode(), digest_size=16).hexdigest()
                bridge_micro[f"{bridge_topic}_chunk_{b_hash[:8]}"] = {
                    "text": f"Ponte neural lógica estruturando {ilha_A} e {ilha_B}.",
                    "embedding": micro_emb,
                    "blakegravity_signature": b_hash
                }
                
            self.knowledge_base[bridge_topic] = {
                "macro_embedding": bridge_vec,
                "micro_nodes": bridge_micro,
                "metadata": {"total_chunks": 3, "mocked": True, "is_bridge": True}
            }
            
        self._save_db()
        print(f"✅ [{self.nome}] Migração Mock + Neurogênese completa! Você já pode testar o Zoom Fractal na interface 3D.")

if __name__ == "__main__":
    organ = FractalMemoryOrgan()
    # Para testes imediatos da UI, rodamos a migração mock:
    organ.mock_migrate_from_flat()
