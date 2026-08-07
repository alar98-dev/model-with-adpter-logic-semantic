from cortex.organs.emergent_sense import EmergentSenseOrgan
organ = EmergentSenseOrgan()
ancora_nativa = {'metadata': {'caminho_tracado': 'foo', 'descricao_semantica': 'bar'}}
b = organ.serialize_aligned_2kb(ancora_nativa)
print(len(b))
