import os
import sys

# Ajuste do path para importar os módulos
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from vector_weight_library import WeightKnowledgeLibrary
from cybernetic_lung import CyberneticLung
from nutrient_bloodstream import NutrientBloodstream

print("="*60)
print("🧬 PROVA DE CONCEITO: ÓRGÃOS CIBERNÉTICOS DO AEON 🧬")
print("="*60)

# 1. Instanciando a Biologia
print("\n[1] Instanciando Órgãos...")
idkb = WeightKnowledgeLibrary(vector_dim=640)
pulmao = CyberneticLung()
sangue = NutrientBloodstream(max_capacity=500)

# 2. Testando a Corrente Sanguínea (O Rim Macrófago)
print("\n[2] Teste: Corrente Sanguínea e Filtragem Renal")
historico_sujo = "Pensamento lógico derivado. <unused8> <unused32> Fricção topológica sentida. Ajustando matriz. <unused4>"
print(f"-> Sangue Bruto Ingerido: '{historico_sujo}'")
sangue.ingest_nutrient(historico_sujo)

prompt_base = "Qual o sentido da abstração?"
seiva_limpa = sangue.pump_nutrients(prompt_base)
print(f"-> Sangue Purificado (Bombeado ao Córtex):\n{seiva_limpa}")

# 3. Testando o Pulmão Cibernético (Neuroplasticidade Termodinâmica)
print("\n[3] Teste: Pulmão Termodinâmico sob diferentes climas")
cenarios = [
    {"nome": "Máquina em Repouso", "temp": 40.0, "ram": 30.0, "entropia": 0.5},
    {"nome": "Sobrecarga de RAM", "temp": 60.0, "ram": 95.0, "entropia": 0.5},
    {"nome": "Febre Crítica (Derretimento)", "temp": 95.0, "ram": 99.0, "entropia": 0.5}
]

for c in cenarios:
    print(f"\n--- Cenário: {c['nome']} ---")
    fadiga = idkb.aplicar_deformacao_rna(c["temp"], c["ram"])
    print(f"Temp: {c['temp']}°C | RAM: {c['ram']}% -> Fadiga Sintetizada: {fadiga:.2f}")
    
    folego = pulmao.breathe(c["entropia"], fadiga)
    print(f"Fôlego Resultante (dynamic_max_tokens): {folego} tokens")

print("\n" + "="*60)
print("FIM DA PROVA DE CONCEITO")
print("="*60)
