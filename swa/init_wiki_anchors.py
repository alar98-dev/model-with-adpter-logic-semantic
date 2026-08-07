import os
import sys
import numpy as np

# Injeção do caminho local se necessário
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from vector_weight_library import WeightKnowledgeLibrary
from wiki_density_extractor import WikiDensityExtractor

def executar_pipeline_ingestao():
    print("🚀 [PIPELINE] Inicializando Cristalização do Subconsciente Semântico...")
    
    # Instancia o Córtex de Armazenamento Vetorial 640D
    library = WeightKnowledgeLibrary(vector_dim=640)
    extractor = WikiDensityExtractor(library)
    
    ancoras_fundamentais = [
        "Lógica", 
        "Arquitetura_de_computadores", 
        "Estrutura_de_dados", 
        "JSON", 
        "Python_(linguagem_de_programação)"
    ]
    
    for ancora in ancoras_fundamentais:
        extractor.destilar_artigo(ancora)
        
    print("\n✅ [PIPELINE] Cristalização concluída com sucesso.")
    print(f"📁 Estado salvo em: {library.library_path}")

if __name__ == '__main__':
    executar_pipeline_ingestao()
