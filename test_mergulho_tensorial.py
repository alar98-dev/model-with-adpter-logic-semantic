import sys
import time
import json
import socket
import struct
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

sys.path.append('/home/alifer/swa-aeon')
from swa.synaptic_router import SynapticBusRouter
from swa.synaptic_address import SynapticRegion, sensory_addr, cortex_addr, SynapticNodeState
from swa.context_bus import ContextBus, Role

print('1. Carregando Motor Tensorial do Hipocampo (Gemma-3-270m)...')
# Otimização para CPU local
modelo_path = '/services/model-with-adpter-logic-semantic/models/gemma-3-270m'
processor = AutoProcessor.from_pretrained(modelo_path)
model = AutoModelForCausalLM.from_pretrained(modelo_path, torch_dtype=torch.bfloat16)
device = 'cpu'
model.to(device)

print('\n2. Inicializando Router e Barramento...')
router = SynapticBusRouter()
bus = ContextBus()
if bus.agent_id < 0:
    bus.register_agent('hippocampus_tensor')

start_seq = bus.write_seq

print('\n3. Simulando Memória Profunda no Redis...')
vector_addr = sensory_addr('fs', 'file_test_999')
memoria_pesada = 'EXTREMAMENTE SECRETO: A arquitetura OODA depende do modulo de metacognição localizado no quadrante 4. Os dados originais foram forjados pelo Obelisk Devourer e estão ocultos na matriz topológica.'
node = SynapticNodeState(
    address=vector_addr,
    model_id='fs_sense',
    status='collapsed',
    telemetry={'size': len(memoria_pesada), 'content': memoria_pesada}
)
router.register_node(node)

# Inserimos apenas o alerta truncado no mmap
bus.write_token(f'[SENSOR FS: Arquivo Colapsado em {vector_addr.vector_path}]\nConteúdo truncado...', role=Role.SYSTEM)

def injetar_metamateria(texto):
    print(f'   -> [TENSOR FLOW] Vetorizando {len(texto)} caracteres de memória...')
    # Inferência Ativa (Extração Oculta sem geração de token)
    inputs = processor(text=texto, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # Extrai Camada 9 e faz Mean Pooling
    layer_9 = outputs.hidden_states[9].to(torch.float32)
    metamatter_vector = layer_9.mean(dim=1).squeeze().numpy()
    
    # Topologia 64x64
    topology = np.zeros(4096, dtype=np.float32)
    topology[:640] = metamatter_vector
    topology_matrix = topology.reshape((64, 64))
    
    # Injeta no EM Core (Magnetic Lattice)
    em_sock_path = '/dev/shm/aeon_em.sock'
    try:
        if os.path.exists(em_sock_path):
            em_payload = json.dumps({
                'action': 'ingest_topology',
                'matrix': topology_matrix.flatten().tolist(),
                'source': 'hippocampus_recall'
            }).encode('utf-8')
            em_len = struct.pack('!I', len(em_payload))
            _em_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            _em_sock.settimeout(2.0)
            _em_sock.connect(em_sock_path)
            _em_sock.sendall(em_len + em_payload)
            _em_sock.close()
            print('   -> [EM CORE] Matriz de Memória Profunda ancorada com sucesso no Lattice Eletromagnético!')
            return True
    except Exception as e:
        import os
        if not os.path.exists(em_sock_path):
             print('   -> [EM CORE] Socket /dev/shm/aeon_em.sock não encontrado. (OODA Offline?)')
        else:
             print(f'   -> [EM CORE] Falha na injeção via socket: {e}')
    return False

print('\n4. Acordando o Órgão Hipocampo (Escutando o Córtex)...')
def hipocampo_on_thought(v_path, payload):
    pensamento = payload.get('pensamento', '')
    if '[MERGULHO_FRACTAL:' in pensamento:
        start = pensamento.find('[MERGULHO_FRACTAL:') + 18
        end = pensamento.find(']', start)
        alvo = pensamento[start:end].strip()
        
        print(f'\n   >>> Hipocampo detectou pedido de mergulho para: {alvo}')
        try:
            raw = router.r.get(alvo)
            if raw:
                estado = json.loads(raw)
                memoria = estado.get('telemetry', {}).get('content', 'Vazio')
                
                # NOVO FLOW: Ao inves de jogar texto no mmap, geramos metamatéria!
                sucesso = injetar_metamateria(memoria)
                
                if sucesso:
                    # Avisa no barramento rápido que a intuição foi carregada
                    bus.write_token(f'[HIPOCAMPO] Memória profunda {alvo} processada. A Matriz Topológica EM foi alterada com este conhecimento.', role=Role.SYSTEM)
            else:
                print('   -> Hipocampo falhou: Memória não encontrada no Redis.')
        except Exception as e:
            print(f'Erro no hipocampo: {e}')

router.subscribe(SynapticRegion.CORTEX, hipocampo_on_thought)

print('\n5. Simulando o Córtex (LLM) solicitando a memória profunda...')
router.pulse(cortex_addr('test', 'act'), {'pensamento': f'[MERGULHO_FRACTAL: {vector_addr.vector_path}]'})

# Dá tempo para o hipocampo (rodando na thread do pubsub) processar o LLM
time.sleep(15)

print('\n6. Verificando o ContextBus final...')
final = bus.read_context_text(start_seq)
print('--- CONTEÚDO DO BARRAMENTO AFERENTE ---')
print(final)
