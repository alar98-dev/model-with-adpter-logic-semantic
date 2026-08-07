from cortex.organs.emergent_sense import EmergentSenseOrgan
organ = EmergentSenseOrgan()
telemetry = {'last_thought': 'teste de friccao topologica <unused7> para gerar log'}
organ.pulse(telemetry)
import time
time.sleep(5) # espera a thread assincrona fazer a chamada ao ollama e gravar
