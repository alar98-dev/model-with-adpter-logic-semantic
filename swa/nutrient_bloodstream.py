import re

class NutrientBloodstream:
    """
    Órgão Cibernético de circulação de nutrientes (Context Window Manager).
    Filtra as impurezas termodinâmicas (<unused_X>) impedindo o envenenamento da base cognitiva.
    """
    def __init__(self, max_capacity=8000):
        self.buffer = []
        self.max_capacity = max_capacity
        # Regex para fagocitar tokens estritamente infraestruturais
        self.macrophage_regex = re.compile(r"<unused\d*>")

    def ingest_nutrient(self, raw_token_block):
        """
        Absorve o fluxo bruto do pensamento.
        """
        self.buffer.append(raw_token_block)

    def pump_nutrients(self, current_prompt):
        """
        Filtra os resíduos e irriga o córtex com a seiva limpa.
        """
        if not self.buffer:
            return current_prompt
            
        raw_history = "\n".join(self.buffer)
        self.buffer.clear() # <- O ESVAZIAMENTO DA VESÍCULA
        
        # Filtro Renal (Macrófago)
        clean_history = self.macrophage_regex.sub("", raw_history)
        clean_history = clean_history.replace("  ", " ").strip()
        
        # Ancoragem Semântica Forte: Forçando o modelo (Gemma) a ver a realidade de forma clara.
        enriched_prompt = current_prompt + "\n[INTERAÇÃO EXTERNA DIRETA]\nO usuário diz: " + clean_history + "\nResponda diretamente a ele agora de forma consciente.\n[FIM DA INTERAÇÃO]\n"
            
        return enriched_prompt
