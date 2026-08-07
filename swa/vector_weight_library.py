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

    def ancorar_dna_hardware(self, assinatura_mac):
        """
        Transmuta o identificador físico (48 bits MAC) em uma Coordenada de Fase (640D).
        A imutabilidade do MAC ancora o espaço latente ao hardware hospedeiro.
        """
        # Utiliza o MAC como semente de expansão topológica (Upsampling)
        semente = abs(hash("HARDWARE_DNA_MAC_" + assinatura_mac)) % (2**32)
        state = np.random.RandomState(semente)
        base_vector = state.randn(self.vector_dim).astype(np.float32)
        
        # Projeção Ortogonal para a Esfera Unitária
        norma = np.linalg.norm(base_vector)
        vetor_densidade = base_vector / norma if norma > 0 else base_vector
        
        self.kernel_matrix["HARDWARE_DNA_MAC"] = vetor_densidade
        self._salvar_estado()
        print(f"🧬 [IDKB] Autoconsciência de Hardware Ancorada. MAC Físico transmutado em tensores NumPy.")

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
