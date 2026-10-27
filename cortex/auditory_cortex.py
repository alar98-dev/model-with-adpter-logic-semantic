import asyncio
import json
import psycopg2
import hashlib
import sys
import os
import numpy as np
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))
from swa.file_crawler import BlakeNativeCrawler
from swa.audio_stimulus import normalize_audio_provenance
from sensory_validation_mel import SpeakerClassifier, UnifiedSensoryLogger


def _iso_para_datetime(valor):
    """ISO-8601 -> datetime com fuso. Devolve None se não for parseável."""
    if not valor:
        return None
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(valor).replace("Z", "+00:00"))
    except Exception:
        return None


def _timestamps_por_palavra(texto, payload):
    """Instantes absolutos por palavra, em epoch seconds, só se o ASR os deu.

    `vocal_words.abs_timestamp` é double precision (epoch), não timestamptz,
    então convertemos para float aqui. Nunca inventa marcação: devolve lista
    vazia quando o ASR não mandou `word_timestamps` ou o alinhamento não cobre
    a frase, e o ledger então marca a origem como estimada.
    """
    brutos = (payload or {}).get("word_timestamps")
    if not brutos or not texto:
        return []
    out = []
    for item in brutos:
        valor = None
        if isinstance(item, dict):
            valor = item.get("start") or item.get("t0")
        else:
            valor = item
        if isinstance(valor, (int, float)):
            out.append(float(valor))
            continue
        dt_ = _iso_para_datetime(valor)
        out.append(dt_.timestamp() if dt_ else None)
    out = [t for t in out if t is not None]
    return out if len(out) >= len(texto.split()) else []


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
    audio_provenance: dict = field(default_factory=dict)
    
    @property
    def concept_group(self) -> str:
        return "MEMORIA_AUDITIVA" if self.source == "TTS_FEEDBACK" else "ESTIMULO_EXTERNO"

    @property
    def text_with_context(self) -> str:
        if self.source in {"MICROPHONE", "AUDIO_INPUT"}:
            return (
                f"[LOCUTOR ESTIMADO: {self.speaker}; "
                f"ROTA: {self.audio_provenance.get('propagation_path', 'unknown')}] "
                f"{self.raw_text}"
            )
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

    def save_semantic_anchor(self, stimulus: AuditoryStimulus, embedding_2560: list) -> int | None:
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO blake_semantic_anchors 
                (file_path, concept_group, signature_hash, raw_content, embedding_2560, weight)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (file_path) DO NOTHING
                RETURNING id;
            """, (
                stimulus.file_path, 
                stimulus.concept_group, 
                stimulus.fingerprint, 
                stimulus.text_with_context[:1000], 
                embedding_2560, 
                2.5
            ))
            row = cur.fetchone()
            if row:
                return row[0]
            # Conflito: a âncora já existe — recupera o id para ligar o ledger.
            cur.execute("SELECT id FROM blake_semantic_anchors WHERE file_path = %s;",
                        (stimulus.file_path,))
            found = cur.fetchone()
            return found[0] if found else None
        finally:
            # Sem isto a conexão fica "idle in transaction" para sempre,
            # segurando snapshot e travando o vacuum.
            self.conn.commit()
            cur.close()


class AuditoryCortexProtocol(asyncio.DatagramProtocol):
    """Adaptador de Rede (Infraestrutura) para interceptar os datagramas UDP."""
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        
    def connection_made(self, transport):
        print("👂 [CÓRTEX AUDITIVO] Escutando eventos de áudio na porta UDP 9200 (Modo DDD-Worker)...")
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
            acoustic_signature=payload.get("acoustic_signature", []),
            audio_provenance=normalize_audio_provenance(
                payload.get("audio_provenance")),
        )

        # Identidade Vocal
        if stimulus.acoustic_signature:
            sig_array = np.array(stimulus.acoustic_signature)
            stimulus.speaker, stimulus.accuracy_score = self.classifier.identify_voice(sig_array)
        else:
            # A origem já avaliou a assinatura vocal (janela Windows); o espelho
            # Redis->UDP traz o veredito pronto. Sem isso, a fala chegava
            # sempre como "Desconhecido" e accuracy 0.0.
            stimulus.speaker = payload.get("speaker", "Desconhecido")
            # `score` é o nome canônico publicado pelo orquestrador; similarity e
            # confidence são aceitos para compatibilidade. Sem o fallback para
            # `score`, a accuracy chegava 0.00% mesmo com identidade casada.
            bruto = (
                payload.get("score")
                if payload.get("score") is not None
                else payload.get("similarity", payload.get("confidence", 0.0))
            )
            stimulus.accuracy_score = float(bruto or 0.0)

        # Logging de entrada
        if stimulus.source in {"MICROPHONE", "AUDIO_INPUT"}:
            self.sensory_logger.log_event(
                "INPUT",
                f"Transcrição de áudio não verificada: '{stimulus.raw_text}'",
                {
                    "speaker_estimate": stimulus.speaker,
                    "accuracy": f"{stimulus.accuracy_score*100:.2f}%",
                    "propagation_path": stimulus.audio_provenance.get(
                        "propagation_path", "unknown"),
                },
            )
            
        # Geração de Embendding (Domain Rule)
        v_2560 = self.crawler.generate_native_2560_vector(stimulus.text_with_context)
        
        # Persistência via Repositório
        anchor_id = self.repository.save_semantic_anchor(stimulus, v_2560)

        # Ledger vocal: a fala vira entidade rastreável, com snapshot tensorial
        # e tempo por palavra.
        self._record_vocal_ledger(stimulus, payload, v_2560, anchor_id)
        
        # Output visual do Córtex
        self._dispatch_consciousness_echo(stimulus)

    def _record_vocal_ledger(self, stimulus, payload, v_2560, anchor_id):
        """Grava a fala no ledger vocal (identidade + snapshot + tempo por palavra).

        Usa a via psycopg2 sobre a conexão já aberta: este processo tem
        MemoryMax=512M e carregar o SQLAlchemy aqui custava ~115MB residentes,
        o que empurrava o pico para fora do MemoryHigh e disparava OOM kill.

        Best-effort: uma falha aqui nunca pode derrubar o processamento do
        estímulo — a âncora semântica já foi persistida acima.
        """
        if stimulus.audio_provenance.get("speaker_attribution") != "verified":
            print(
                "ℹ️ Ledger vocal sem vínculo de identidade: "
                "origem/locutor ainda não verificados.")
            return None
        try:
            from swa.db.vocal_ledger import record_utterance_pg
        except Exception:
            return None

        try:
            started_at = _iso_para_datetime(payload.get("utter_started_at"))
            ended_at = _iso_para_datetime(payload.get("utter_ended_at"))
            duration_ms = payload.get("duration_ms")
            if started_at and ended_at and duration_ms is None:
                duration_ms = (ended_at - started_at).total_seconds() * 1000.0

            return record_utterance_pg(
                self.repository.conn,
                stimulus.raw_text,
                speaker=(
                    stimulus.speaker
                    if stimulus.audio_provenance.get("speaker_attribution") == "verified"
                    else None
                ),
                is_owner=(
                    bool(payload.get("is_alifer", False))
                    and stimulus.audio_provenance.get("speaker_attribution") == "verified"
                ),
                similarity=float(stimulus.accuracy_score or 0.0),
                source=stimulus.source,
                captured_at=payload.get("timestamp"),
                energy=payload.get("energy"),
                anchor_id=anchor_id,
                embedding=v_2560,
                started_at=started_at,
                ended_at=ended_at,
                duration_ms=float(duration_ms) if duration_ms is not None else None,
                word_timestamps=_timestamps_por_palavra(stimulus.raw_text, payload),
                session_key=(
                    f"live-{started_at.strftime('%Y%m%d')}" if started_at else None
                ),
                meta={
                    "channel": "auditory_cortex",
                    "concept_group": stimulus.concept_group,
                    "fingerprint": stimulus.fingerprint[:16],
                    "audio_provenance": stimulus.audio_provenance,
                    "speaker_estimate": stimulus.speaker,
                },
            )
        except Exception as e:
            print(f"⚠️ Ledger vocal não registrou a fala: {e}")
            return None

    def _dispatch_consciousness_echo(self, stimulus: AuditoryStimulus):
        if stimulus.source == "TTS_FEEDBACK":
            msg = f"\n🧠 [MEMÓRIA AUDITIVA] Feedback interno processado: {stimulus.raw_text[:60]}...\n"
            self.sensory_logger.log_event("OUTPUT", "Auto-audição (Eco)", {"text": stimulus.raw_text[:50]})
        elif stimulus.source in {"MICROPHONE", "AUDIO_INPUT"}:
            route = stimulus.audio_provenance.get("propagation_path", "unknown")
            msg = (
                f"\n🎙️ [EVENTO DE ÁUDIO | rota={route} | origem/locutor "
                f"não verificados] {stimulus.raw_text[:60]}...\n"
            )
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
        local_addr=('127.0.0.1', 9200)
    )
    
    try:
        await asyncio.sleep(3600 * 24)
    finally:
        transport.close()
        worker_task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
