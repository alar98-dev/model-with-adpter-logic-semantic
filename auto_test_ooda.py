import time, json, threading
from swa.synaptic_router import SynapticBusRouter
from swa.synaptic_address import SynapticRegion, cortex_addr
import cortex.organs.web_sense as web_sense

print("🧠 [Aeon Auto-Test] Booting test sequence...")
router = SynapticBusRouter()
organ = web_sense.init_organ()
results_received = []

def observer(vector_path, payload):
    print(f"\n⚡ [PULSO INTERCEPTADO] Origem: {vector_path}")
    results_received.append(payload)

router.subscribe(SynapticRegion.SENSORY, observer)
time.sleep(1.0)
print("\n📡 [Aeon Auto-Test] Injetando pensamento OODA no Cortex...")
start_time = time.time()
router.pulse(
    target=cortex_addr("test_driver", "act"),
    payload={"pensamento": "Estou curioso. [PESQUISAR: Arquitetura OODA Loop]"}
)

timeout = 10.0
while not results_received and timeout > 0:
    time.sleep(0.5)
    timeout -= 0.5
end_time = time.time()

if results_received:
    print(f"\n✅ [SUCESSO] OODA Loop Fechado em {end_time - start_time:.2f} segundos!")

# Leitura raw do mmap (ContextBus)
try:
    from swa.context_bus import ContextBus
    bus = ContextBus()
    # Ler os ultimos registros de texto do ContextBus
    raw_text = bus.read_context_text()
    linhas = [linha for linha in raw_text.splitlines() if linha.strip()]
    ultimas_linhas = linhas[-5:] if len(linhas) > 5 else linhas
    print("\n📖 [ContextBus SHM Check]: Ultimas escritas fisicas no anel de memoria:")
    for linha in ultimas_linhas:
        print(f" -> {linha}")
except Exception as e:
    print(f"⚠️ Aviso ao verificar ContextBus local: {e}")
print("🏁 Teste concluído.")
