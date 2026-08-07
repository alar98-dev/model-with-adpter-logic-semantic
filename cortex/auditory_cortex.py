import asyncio
import json
import psycopg2
import hashlib
import sys
import os
import numpy as np
import concurrent.futures
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from swa.file_crawler import BlakeNativeCrawler
from sensory_validation_mel import SpeakerClassifier, UnifiedSensoryLogger


# ==========================================
# 1. DOMAIN LAYER
# ==========================================
@dataclass
class AuditoryStimulus:
    """Entidade de Domínio representando um estímulo sonoro/vocal."""
    raw_text: str
    source: str
    acoustic_signature: list
    speaker: str = "Desconhecido"
    accuracy_score: float = 0.0
    
    @property
    def concept_group(self) -> str:
        return "MEMORIA_AUDITIVA" if self.source == "TTS_FEEDBACK" else "ESTIMULO_EXTERNO"

    @property
    def text_with_context(self) -> str:
        if self.source == "MICROPHONE":
            return f"[LOCUTOR: {self.speaker}] {self.raw_text}"
        return self.raw_text

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.text_with_context.encode('utf-8')).hexdigest()

    @property
    def file_path(self) -> str:
        return f"{self.concept_group.lower()}_{self.source}#{self.fingerprint[:8]}"


# ==========================================
# 2. INFRASTRUCTURE LAYER
# ==========================================
class AuditoryRepository:
    """Repositório (Infraestrutura) para persistir as âncoras semânticas no PostgreSQL."""
    def __init__(self):
        dsn = "dbname=swa_matrix user=alifer"
        password = os.environ.get("SWA_DB_PASSWORD")
        if password: dsn += f" password={password} host=localhost"
        else: dsn += " host=localhost"
        self.conn = psycopg2.connect(dsn)

    def save_semantic_anchor(self, stimulus: AuditoryStimulus, embedding_2560: list):
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO blake_semantic_anchors 
            (file_path, concept_group, signature_hash, raw_content, embedding_2560, weight)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (file_path) DO NOTHING;
        """, (
            stimulus.file_path, 
            stimulus.concept_group, 
            stimulus.fingerprint, 
            stimulus.text_with_context[:1000], 
            embedding_2560, 
            2.5
        ))
        self.conn.commit()
        cur.close()


class AuditoryCortexProtocol(asyncio.DatagramProtocol):
    """Adaptador de Rede (Infraestrutura) para interceptar os datagramas UDP."""
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        
    def connection_made(self, transport):
        print("👂 [CÓRTEX AUDITIVO] Escutando Ecos Físicos na porta UDP 9200 (Modo DDD-Worker)...")
        self.transport = transport

    def datagram_received(self, data, addr):
        try:
            self.queue.put_nowait((data, addr))
        except asyncio.QueueFull:
            print("⚠️ Fila auditiva cheia! Descartando pacote (Entropia alta).")


# ==========================================
# 3. APPLICATION LAYER (SERVICES & WORKERS)
# ==========================================
class AuditoryApplicationService:
    """Serviço de Aplicação responsável por orquestrar a lógica de negócio e o Domínio."""
    def __init__(self):
        self.crawler = BlakeNativeCrawler()
        self.classifier = SpeakerClassifier()
        self.sensory_logger = UnifiedSensoryLogger()
        self.repository = AuditoryRepository()

    def handle_stimulus(self, payload: dict):
        text = payload.get("text", "").strip()
        if len(text) < 3:
            return
            
        # Instancia a Entidade de Domínio
        stimulus = AuditoryStimulus(
            raw_text=text,
            source=payload.get("source", "UNKNOWN"),
            acoustic_signature=payload.get("acoustic_signature", [])
        )
        
        # Identidade Vocal
        if stimulus.acoustic_signature:
            sig_array = np.array(stimulus.acoustic_signature)
            stimulus.speaker, stimulus.accuracy_score = self.classifier.identify_voice(sig_array)
            
        # Logging de entrada
        if stimulus.source == "MICROPHONE":
            self.sensory_logger.log_event("INPUT", f"Ouvido de '{stimulus.speaker}': '{stimulus.raw_text}'", 
                                          {"accuracy": f"{stimulus.accuracy_score*100:.2f}%"})
            
        # Geração de Embendding (Domain Rule)
        v_2560 = self.crawler.generate_native_2560_vector(stimulus.text_with_context)
        
        # Persistência via Repositório
        self.repository.save_semantic_anchor(stimulus, v_2560)
        
        # Output visual do Córtex
        self._dispatch_consciousness_echo(stimulus)

    def _dispatch_consciousness_echo(self, stimulus: AuditoryStimulus):
        if stimulus.source == "TTS_FEEDBACK":
            msg = f"\n🧠 [MEMÓRIA AUDITIVA] Feedback interno processado: {stimulus.raw_text[:60]}...\n"
            self.sensory_logger.log_event("OUTPUT", "Auto-audição (Eco)", {"text": stimulus.raw_text[:50]})
        elif stimulus.source == "MICROPHONE":
            msg = f"\n🎙️ [ESTÍMULO EXTERNO (REALIDADE)] Áudio capturado: {stimulus.raw_text[:60]}...\n"
        else:
            return
            
        sys.stdout.write(msg)
        sys.stdout.flush()
        try:
            with open("/services/model-with-adpter-logic-semantic/swa_Aeon.log", "a", encoding="utf-8") as f:
                f.write(msg)
        except Exception:
            pass


class AsyncAuditoryWorker:
    """Worker Assíncrono que processa a Fila isolando a thread principal."""
    def __init__(self):
        self.app_service = AuditoryApplicationService()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.loop = asyncio.get_event_loop()

    async def run(self, queue: asyncio.Queue):
        print("🧠 [WORKER AUDITIVO - DDD] Thread de consolidação semântica ativa.")
        while True:
            data, addr = await queue.get()
            try:
                payload = json.loads(data.decode('utf-8'))
                # Delega ao Application Service em thread separada
                await self.loop.run_in_executor(self.executor, self.app_service.handle_stimulus, payload)
            except Exception as e:
                print(f"⚠️ Erro no Worker do Córtex Auditivo: {e}")
            finally:
                queue.task_done()


# ==========================================
# 4. ENTRYPOINT
# ==========================================
async def main():
    print("🚀 Iniciando Córtex Auditivo do Aeon (Padrão DDD + Worker)...")
    loop = asyncio.get_running_loop()
    
    packet_queue = asyncio.Queue(maxsize=2000)
    worker = AsyncAuditoryWorker()
    worker_task = asyncio.create_task(worker.run(packet_queue))
    
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: AuditoryCortexProtocol(packet_queue),
        local_addr=('0.0.0.0', 9200)
    )
    
    try:
        await asyncio.sleep(3600 * 24)
    finally:
        transport.close()
        worker_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
