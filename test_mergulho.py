import sys
import time
import json
sys.path.append('/home/alifer/swa-aeon')

from swa.synaptic_router import SynapticBusRouter
from swa.synaptic_address import SynapticRegion, sensory_addr, cortex_addr, SynapticNodeState
from swa.context_bus import ContextBus, Role

print('1. Inicializando Router e Barramento...')
router = SynapticBusRouter()
bus = ContextBus()
if bus.agent_id < 0:
    bus.register_agent('hippocampus_test')

start_seq = bus.write_seq

print('\n2. Simulando FS Sense guardando uma Memória Profunda no Redis...')
# Cria um endereço e salva um texto gigante
vector_addr = sensory_addr('fs', 'file_test_999')
memoria_pesada = 'EXTREMAMENTE SECRETO: A arquitetura OODA depende do modulo de metacognição localizado no quadrante 4. Os dados originais foram forjados pelo Obelisk Devourer e estão ocultos na matriz topológica.'
node = SynapticNodeState(
    address=vector_addr,
    model_id='fs_sense',
    status='collapsed',
    telemetry={'size': len(memoria_pesada), 'content': memoria_pesada}
)
# Essa é a blindagem que fizemos antes: a memória está salva no Redis (não se perde)
router.register_node(node)

# Para simular o problema do ContextBus, inserimos apenas o resumo que cabe no mmap
bus.write_token(f'[SENSOR FS: Arquivo Colapsado em {vector_addr.vector_path}]\nConteúdo: EXTREMAMENTE SECRETO: A arq...', role=Role.SYSTEM)


print('\n3. Acordando o Órgão Hipocampo (Escutando o Córtex)...')
def hipocampo_on_thought(v_path, payload):
    pensamento = payload.get('pensamento', '')
    if '[MERGULHO_FRACTAL:' in pensamento:
        # Extrai o ID do vetor
        start = pensamento.find('[MERGULHO_FRACTAL:') + 18
        end = pensamento.find(']', start)
        alvo = pensamento[start:end].strip()
        
        print(f'   -> Hipocampo detectou pedido de mergulho para: {alvo}')
        # Busca no banco de longo prazo (Redis)
        from swa.synaptic_address import SynapticAddress
        import swa.synaptic_address
        # Reconstroi endereco
        try:
            import json
            raw = router.r.get(alvo.replace(':stream', ':state'))
            estado = json.loads(raw) if raw else None
            if estado and 'telemetry' in estado:
                memoria = estado['telemetry'].get('content', 'Vazio')
                # Injeta de volta no barramento curto (agora focado na resposta)
                bus.write_token(f'[HIPOCAMPO] Memória Recuperada: {memoria}', role=Role.SYSTEM)
                print('   -> Hipocampo injetou a memória de volta no ContextBus!')
            else:
                print('   -> Hipocampo falhou: Memória não encontrada no Redis.')
        except Exception as e:
            print(f'Erro no hipocampo: {e}')

router.subscribe(SynapticRegion.CORTEX, hipocampo_on_thought)

print('\n4. Simulando o Córtex (LLM) percebendo que a informação está cortada...')
print('   LLM Pensa: A informação está truncada. Preciso de mais detalhes.')
# Injetamos a intenção de regressão via Pulse
router.pulse(cortex_addr('cortex', 'act'), {'pensamento': f'[MERGULHO_FRACTAL: {vector_addr.vector_path}]'})

time.sleep(2) # Tempo para o Hipocampo reagir

print('\n5. Verificando o ContextBus final...')
final = bus.read_context_text(start_seq)
print('--- CONTEÚDO DO BARRAMENTO ---')
print(final)

