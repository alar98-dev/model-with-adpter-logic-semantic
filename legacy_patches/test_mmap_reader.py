import os
from sensors.mmap_sensor import MmapSensoryEngine

filepath = '/services/model-with-adpter-logic-semantic/logs/emergent_2kb_matrix.jsonl'
anomalias = MmapSensoryEngine.ler_matriz_emergente(filepath)
print(f'Total lido: {len(anomalias)}')
if anomalias:
    primeira = anomalias[0]
    print(f'Primeira anomalia: {primeira.get("concept")}')
    if "vector_numpy" in primeira:
        vec = primeira["vector_numpy"]
        print(f'Vetor convertido shape: {vec.shape}, dtype: {vec.dtype}')
        print(f'Primeiros 5 valores: {vec[:5]}')
