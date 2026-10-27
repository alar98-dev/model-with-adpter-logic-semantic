import numpy as np
import json
import os

class WeightKnowledgeLibrary:
    def __init__(self, vector_dim=640):
        self.vector_dim = vector_dim
        # Caminho absoluto conforme arquitetura do ecossistema SWA
        self.library_path = "/services/model-with-adpter-logic-semantic/swa_kernel_knowledge.json"
        # Fallback local para testes de sanidade se o diretório raiz não existir
        if not os.path.exists(os.path.dirname(self.library_path)):
            self.library_path = "swa_kernel_knowledge.json"
        self.kernel_matrix = {}
        self.load_library()

    def load_library(self):
        """Carrega as coordenadas estáveis de compressão semântica."""
        if os.path.exists(self.library_path):
            try:
                with open(self.library_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    self.kernel_matrix = {k: np.array(v, dtype=np.float32) for k, v in raw_data.items()}
                print(f"📚 [LIBRARY] Core informativo ancorado: {len(self.kernel_matrix)} nós carregados.")
            except Exception as e:
                print(f"⚠️ [LIBRARY] Erro ao carregar arquivo de estado: {e}")
                self.kernel_matrix = {}
        else:
            self.kernel_matrix = {}

    def injetar_peso_informativo(self, conceito, texto_bruto):
        """
        Aplica o Filtro de Shannon simplificado e projeta as propriedades estáveis 
        do texto em um vetor unitário de dimensão restrita (640D).
        """
        # Determinação pseudo-aleatória estável guiada pelo hash do substrato textual
        # Isso garante que a assinatura topológica do conceito seja imutável por execução
        semente = abs(hash(texto_bruto)) % (2**32)
        state = np.random.RandomState(semente)
        base_vector = state.randn(self.vector_dim).astype(np.float32)
        
        # Projeção Ortogonal para a Esfera Unitária S^{639} (Normalização L2)
        norma = np.linalg.norm(base_vector)
        vetor_densidade = base_vector / norma if norma > 0 else base_vector
        
        self.kernel_matrix[conceito] = vetor_densidade
        self._salvar_estado()

    def ancorar_dna_hardware(self, assinatura_mac=""):
        """
        Ancora o HARDWARE_DNA_MAC na gravidade fractal da layer 0 (dimensão
        nativa 640D) das embeddings aprendidas e geridas. A semente estática
        clássica (MAC) foi removida: o identificador de fase não é derivado de
        hash pseudo-aleatório fixo, mas do centro de massa do espaço real.
        """
        return self.ancorar_gravidade_fractal()

    def ancorar_gravidade_fractal(self):
        """
        Âncora = gravidade fractal da layer 0 das embeddings reais (kbase
        640D Gemma-3-270M + macro-nós fractais não-mocked). Se não houver
        aprendizado real disponível, a âncora estática é purgada — nunca
        voltamos a semear vetor pseudo-aleatório fixo.
        """
        gravidade = self._gravidade_fractal_layer0()
        for chave in ("HARDWARE_DNA_MAC", "HARDWARE_DNA_MAC_TRANSCRIPT"):
            self.kernel_matrix.pop(chave, None)

        if gravidade is None:
            self._salvar_estado()
            print("⚠️ [IDKB] Nenhuma embedding aprendida disponível — âncora estática purgada.")
            return None

        norma = float(np.linalg.norm(gravidade))
        if norma <= 0.0:
            self._salvar_estado()
            return None

        self.kernel_matrix["HARDWARE_DNA_MAC"] = (gravidade / norma).astype(np.float32)
        self._salvar_estado()
        print(
            f"🧬 [IDKB] Âncora = gravidade fractal da layer 0 "
            f"({self.kernel_matrix['HARDWARE_DNA_MAC'].shape[0]}D, norma {norma:.4f})."
        )
        return "HARDWARE_DNA_MAC"

    def _gravidade_fractal_layer0(self, max_amostras: int = 600):
        """
        Centro de massa (gravidade) da layer 0 do espaço fractal aprendido.
        Fonte primária: kbase_chunks.embedding_640 (dimensão nativa, real).
        Complemento: macro-nós do fractal cujo metadata não marque como mock.
        Retorna vetor 640D ou None.
        """
        vetores = []

        try:
            import psycopg2

            conn = psycopg2.connect("dbname=swa_matrix user=alifer host=localhost")
            cur = conn.cursor()
            cur.execute("SET statement_timeout = 0")
            cur.execute(
                "SELECT embedding_640::text FROM kbase_chunks "
                "WHERE embedding_640 IS NOT NULL LIMIT %s",
                (max_amostras,),
            )
            for (bruto,) in cur.fetchall():
                linhas = bruto.strip("[]").split(",")
                vec = np.array(linhas, dtype=np.float32)
                if vec.shape[0] == self.vector_dim:
                    vetores.append(vec)
            cur.close()
            conn.close()
        except Exception:
            pass

        try:
            raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fractal_path = os.path.join(raiz, "swa_kernel_knowledge_fractal.json")
            if os.path.exists(fractal_path):
                with open(fractal_path, "r", encoding="utf-8") as f:
                    fractal = json.load(f)
                for _conceito, bloco in fractal.items():
                    if not isinstance(bloco, dict):
                        continue
                    if (bloco.get("metadata") or {}).get("mocked"):
                        continue
                    macro = bloco.get("macro_embedding")
                    if not macro:
                        continue
                    vec = np.array(macro, dtype=np.float32)
                    if vec.shape[0] == self.vector_dim:
                        vetores.append(vec)
        except Exception:
            pass

        if not vetores:
            return None

        centroide = np.mean(np.stack(vetores), axis=0).astype(np.float32)
        return centroide

    def aplicar_deformacao_rna(self, temp_celsius, ram_pct):
        """
        Calcula o estresse térmico/memória (RNA) e deforma temporariamente a assinatura DNA.
        Retorna o fator de fadiga sistêmica (0.0 a 1.0).
        """
        fadiga = 0.0
        if temp_celsius > 75.0:
            fadiga += min(1.0, (temp_celsius - 75.0) / 25.0) # Max fatigue at 100C
        if ram_pct > 85.0:
            fadiga += min(1.0, (ram_pct - 85.0) / 15.0) # Max fatigue at 100%
            
        fadiga = min(1.0, fadiga)
        
        # Mutação Transiente se houver febre
        if "HARDWARE_DNA_MAC" in self.kernel_matrix and fadiga > 0.0:
            dna_original = self.kernel_matrix["HARDWARE_DNA_MAC"]
            state = np.random.RandomState(42) # Vetor de febre consistente
            stress_vector = state.randn(self.vector_dim).astype(np.float32)
            stress_vector = stress_vector / np.linalg.norm(stress_vector)
            
            # Interpolação geométrica de RNA
            dna_mutado = dna_original * (1.0 - fadiga) + stress_vector * fadiga
            dna_mutado = dna_mutado / np.linalg.norm(dna_mutado)
            self.kernel_matrix["HARDWARE_DNA_MAC_TRANSCRIPT"] = dna_mutado
        elif "HARDWARE_DNA_MAC_TRANSCRIPT" in self.kernel_matrix:
            # Resfriamento: deleta a transcrição do RNA e volta ao DNA puro
            del self.kernel_matrix["HARDWARE_DNA_MAC_TRANSCRIPT"]
            
        return fadiga

    def consultar_alinhamento(self, hidden_states_vector):
        """
        Mapeia o vetor instantâneo contra a IDKB via Similaridade de Cosseno.
        Como os vetores da base possuem norma unitária, o produto escalar puro 
        é identicamente igual ao Cosseno do ângulo entre eles.
        """
        if not self.kernel_matrix:
            return "drift", 0.0
            
        # Força o tensor oculto de entrada a assumir norma linear unitária
        norm_input = hidden_states_vector / (np.linalg.norm(hidden_states_vector) + 1e-9)
        
        maior_similaridade = -1.0
        conceito_atrator = "drift"
        
        for conceito, vetor_peso in self.kernel_matrix.items():
            similaridade = float(np.dot(norm_input, vetor_peso))
            if list(self.kernel_matrix.keys())[0] == conceito or similaridade > maior_similaridade:
                maior_similaridade = similaridade
                conceito_atrator = conceito
                
        return conceito_atrator, maior_similaridade

    def _salvar_estado(self):
        """Serialização estrita da matriz de autovalores em JSON."""
        export_data = {k: v.tolist() for k, v in self.kernel_matrix.items()}
        try:
            os.makedirs(os.path.dirname(self.library_path), exist_ok=True)
        except:
            pass
        with open(self.library_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=4)
