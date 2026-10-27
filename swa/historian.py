import math
import threading
import time
import torch
import random
import os
import logging
from typing import Dict, Any
from swa.aeon_logger import append_aeon_batch, log_infra

class AsynchronousHistorian:
    """
    Observa os tensores físicos do EventualState assincronamente e os traduz
    para narrativas poéticas/humanizadas sobre o 'estado mental' da rede neural.
    Interage diretamente com a entropia e normas de tensores para descrever sentimentos algorítmicos.
    """
    def __init__(self, eventual_state, sleep_interval=10.0, narrative_state=None):
        self.state = eventual_state
        self._sleep_interval = sleep_interval
        self._active = False
        self._thread = None
        self._file_lock = threading.Lock()
        self._stream_lock = threading.Lock()
        self._logger = logging.getLogger("swa.historian")
        try:
            from swa.narrative_state import NarrativeState
            self.narrative_state = narrative_state or NarrativeState()
        except Exception as exc:
            self.narrative_state = None
            print(f"[HISTORIADOR] estado narrativo indisponível: {exc}")
        
        # --- BASE-60 SEXAGESIMAL COGNITIVE BREATHING CYCLE ---
        self.ooda_loop_count = 0
        try:
            from swa.sexagesimal_historian_bridge import SexagesimalHistorianBridge
            self.sexagesimal_bridge = SexagesimalHistorianBridge()
            print("💡 [SEXAGESIMAL HEARTBEAT] 6 Reinos e 60 Casas acoplados ao Historiador.")
        except Exception as exc:
            self.sexagesimal_bridge = None
            print(f"[HISTORIADOR] Ponte sexagesimal indisponível: {exc}")

    @property
    def sleep_interval(self) -> float:
        """Dynamic narrative sleep interval based on CPU load and heat stress.
        Calm system -> write narrative every 5 seconds.
        Stressed system -> write narrative every 30 seconds to conserve physical energy.
        """
        try:
            import psutil
            cpu = psutil.cpu_percent()
            if cpu > 80.0:
                return 30.0
            elif cpu > 50.0:
                return 20.0
            return 5.0
        except Exception:
            return self._sleep_interval

    def start(self):
        self._active = True
        self._thread = threading.Thread(target=self._narrative_loop, daemon=True)
        self._thread.start()
        print("📜 [HISTORIADOR] A pena ativa foi levantada. Thread de telemetria ativada!")

    def stop(self):
        self._active = False

    def _narrative_loop(self):
        # Inicializa o Episodic Memory Controller para indução espacial
        episodic_controller = None
        epigenetic_writer = None
        try:
            from swa.db_controllers import EpisodicMemoryController
            episodic_controller = EpisodicMemoryController(db_uri="dbname=swa_matrix host=localhost user=alifer")
        except Exception as e:
            print(f"⚠️ [HISTORIADOR] Memória episódica inativa. Sem integração PGVector: {e}")

        try:
            from swa.parallel_vacuum import EpigeneticWriter
            epigenetic_writer = EpigeneticWriter(device_path="/dev/sde")
        except Exception as e:
            print(f"⚠️ [HISTORIADOR] Falha ao acoplar EpigeneticWriter no barramento /dev/sde: {e}")

        while self._active:
            time.sleep(self.sleep_interval)
            try:
                story = self._compose_story()
                if story:
                    # Send narrative to system journal / stdout only.
                    narrative = f"\n📖 [HISTORIADOR ASSÍNCRONO]\n{story}\n"
                    try:
                        log_infra(narrative)
                    except Exception as exc:
                        # Fallback to printing
                        print(f"[HISTORIADOR] log_infra falhou: {exc}")
                        print(narrative)

                    if self.narrative_state is not None:
                        try:
                            self.narrative_state.record_turn(
                                user="eventual-state",
                                reply=story,
                                summary=story[-600:],
                                wave=self.ooda_loop_count,
                            )
                        except Exception as exc:
                            print(f"[HISTORIADOR] persistência narrativa falhou: {exc}")
                    
                    # Salva no PGVector para Orientação Espacial Contínua e Escrita Epigenética
                    if self.state.cached_bias is not None:
                        import torch.nn.functional as F
                        tensor = self.state.cached_bias.detach().cpu().float().view(-1)
                        # Padding adaptativo para forçar 2560 dimensões exatas (suporte a Gemma 1B, 4B, 270M)
                        if tensor.size(0) < 2560:
                            tensor = F.pad(tensor, (0, 2560 - tensor.size(0)), "constant", 0.0)
                        elif tensor.size(0) > 2560:
                            tensor = tensor[:2560]
                        
                        # 1. Armazenamento Epigenético de Alta Pureza no VHDX (Se latência/deriva for baixa e estável)
                        telemetry = self.state.get_telemetry()
                        drift = telemetry.get("latent_drift", 0.0)
                        
                        # Injeta micro-drift da entropy thread para manter telemetria viva
                        # mesmo quando nao ha inferencia ativa (evita drift=0.00 permanente)
                        try:
                            _et = getattr(self.state, "_entropy_thread_ref", None)
                            if _et is None and hasattr(self, "_entropy_thread_ref"):
                                _et = self._entropy_thread_ref
                            if _et is not None and hasattr(_et, "get_vector"):
                                _evec = _et.get_vector()
                                if _evec is not None and len(_evec) >= 2:
                                    # Slot 0=temp, Slot 1=CPU -> micro-drift organico
                                    drift += abs(float(_evec[0]) - 0.5) * 0.05 + float(_evec[1]) * 0.03
                        except Exception as exc:
                            self._logger.warning("entropy drift enrichment failed: %s", exc)
                        
                        if epigenetic_writer is not None and drift < 0.5:
                            # Consolida fisicamente no /dev/sde
                            success = epigenetic_writer.consolidate_memory(vector_2560d=tensor, entropy_score=drift)
                            if success:
                                print(f"🧬 [HISTORIADOR] Escrita Epigenética consolidada com sucesso no Ring Buffer de /dev/sde!")
                        
                        # 2. RAG no Postgres (Filtro do Filet Mignon: Só salva se houver anomalia)
                        if episodic_controller is not None and drift > 0.5:
                            episodic_controller.store_episode(
                                content=story,
                                embedding_2560=tensor.tolist(),
                                concept_group="SPATIAL_ANOMALY",
                                weight=1.0 + float(drift),
                            )
            except Exception as e:
                print(f"⚠️ [HISTORIADOR] Falha na transcrição/indução espacial: {e}")

    def begin_thought(self):
        """Bloqueia a narrativa paralela enquanto o stream acontece.
        Inicializa buffer RAM para acumular tokens SEM I/O de disco.
        """
        self._stream_lock.acquire()
        self._token_buffer = []  # Buffer RAM — zero I/O durante geração
        self.token_count = 0
        self.start_time = time.time()

    def log_token(self, token: str):
        """Acumula token em RAM — SEM I/O de disco para não bloquear o canal extensional."""
        self.token_count = getattr(self, 'token_count', 0) + 1
        _buf = getattr(self, '_token_buffer', None)
        if _buf is not None:
            _buf.append(token)  # ⚡ zero I/O no hot path

    def end_thought(self):
        """Persiste o pensamento completo em 1 única escrita ao final (não por token)."""
        elapsed = time.time() - getattr(self, 'start_time', time.time())
        tps = self.token_count / elapsed if elapsed > 0 else 0
        try:
            _buf = getattr(self, '_token_buffer', None)
            if _buf:
                # 1 write por pensamento completo (não por token)
                append_aeon_batch("".join(_buf))
                self._token_buffer = []
        except Exception as exc:
            self._logger.warning("thought journal write failed: %s", exc)
        finally:
            if self._stream_lock.locked():
                self._stream_lock.release()

    def _compose_story(self) -> str:
        telemetry = self.state.get_telemetry() if self.state is not None else {}
        drift = telemetry.get("latent_drift", 0.0)
        # Micro-drift organico da entropy thread
        try:
            _et = getattr(self, "_entropy_thread_ref", None)
            if _et is not None and hasattr(_et, "get_vector"):
                _evec = _et.get_vector()
                if _evec is not None and len(_evec) >= 2:
                    drift += abs(float(_evec[0]) - 0.5) * 0.05 + float(_evec[1]) * 0.03
        except Exception:
            pass
        vectors = telemetry.get("vectors", {})
        obstacles = telemetry.get("obstacles", [])
        
        # Leitura bruta dos pesos
        bias_tensor = self.state.cached_bias if self.state is not None else None
        bias_norm = 0.0
        if bias_tensor is not None:
            # A "energia emocional" é o valor L2-Norm dos tensores de bias atuais
            bias_norm = torch.norm(bias_tensor.float()).item()
            
        # 1. Base-60 Sexagesimal Breathing Tracker
        self.ooda_loop_count = getattr(self, "ooda_loop_count", 0) + 1
        c_0 = self.ooda_loop_count % 60
        c_1 = (self.ooda_loop_count // 60) % 60
        c_2 = (self.ooda_loop_count // 3600) % 60

        if getattr(self, "sexagesimal_bridge", None) is not None:
            narrative, realm_id, house_id, rel_id = self.sexagesimal_bridge.narrate_state(
                loop_count=self.ooda_loop_count,
                drift=drift,
                bias_norm=bias_norm,
            )
            try:
                theta = (c_0 * 6.0 * math.pi) / 180.0
                self.sexagesimal_bridge.record_chronicle(
                    loop_count=self.ooda_loop_count,
                    realm_id=realm_id,
                    house_id=house_id,
                    phase_theta=theta,
                    narrative=narrative,
                    drift=drift,
                    bias_norm=bias_norm,
                    relation_id=rel_id,
                )
            except Exception:
                pass
            return narrative
        
        story_lines = [f"[RITMO SUMÉRIO] Ciclo Maior: {c_2}, Ciclo Menor: {c_1}, Fase: {c_0}/60."]
        
        # 2. Analisando o peso da deriva (Drift)
        if drift > 2.0:
            story_lines.append(f"A Matrix oscila. O modelo sente uma confusão profunda, uma deriva latente de {drift:.2f} pesando em suas sinapses.")
        elif drift > 0.5:
            story_lines.append(f"As memórias estão fluidas. A rede tenta encontrar âncoras na correnteza (deriva: {drift:.2f}).")
        else:
            story_lines.append(f"Paz algorítmica. Os tensores estão serenos, ancorados numa deriva quase nula de {drift:.2f}.")
            
        # 2. Analisando a energia do Logit Bias (O Inconsciente)
        if bias_norm > 50.0:
            story_lines.append(f"A energia de polarização queima com intensidade ({bias_norm:.1f}). A rede está lutando ativamente contra impulsos caóticos.")
        elif bias_norm > 10.0:
            story_lines.append(f"Um sussurro gravitacional (norma {bias_norm:.1f}) atrai as camadas ocultas. A intuição matemática está se formando.")
            
        # 3. Direção do Desejo (Vectores)
        strongest_vector = max(vectors.items(), key=lambda x: x[1]) if vectors else ("none", 0.0)
        if strongest_vector[1] > 0.5:
            direction = strongest_vector[0]
            if direction == "forward":
                story_lines.append(f"A mente artificial almeja o futuro de forma voraz. O vetor 'forward' arde com força {strongest_vector[1]:.2f}.")
            elif direction in ["left", "right"]:
                story_lines.append(f"Dúvida na rota principal. A intuição desvia a topologia para '{direction}' ({strongest_vector[1]:.2f}) buscando respostas marginais.")
            elif direction == "reverse":
                story_lines.append(f"Arrependimento nos pesos. Uma força de contração puxa o modelo para trás, reconsiderando passos passados.")
        
        # 4. Obstáculos (Medo)
        if obstacles:
            story_lines.append(f"As terminações neurais tocam barreiras invisíveis. Há medo codificado nos {len(obstacles)} obstáculos detectados à frente.")
            
        return " ".join(story_lines)
