import time
import psutil
import os
import threading
from typing import Dict, Any

def get_cpu_temp() -> float:
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            all_temps = [entry.current for entries in temps.values() for entry in entries]
            if all_temps:
                return sum(all_temps) / len(all_temps)
    except:
        pass
    return 45.0  # Default nominal temp if unable to read

def get_hardware_telemetry() -> Dict[str, Any]:
    """Captura a fricção física e a ansiedade do Kernel."""
    stats = psutil.cpu_stats()
    swap = psutil.swap_memory()
    try:
        io_counters = psutil.disk_io_counters()
        read_bytes = io_counters.read_bytes
        write_bytes = io_counters.write_bytes
    except:
        read_bytes = 0
        write_bytes = 0
        
    return {
        "ctx_switches": stats.ctx_switches,
        "interrupts": stats.interrupts,
        "swap_percent": swap.percent,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes
    }

class AeonInternalEntropyThread:
    """
    Thread Interna de Entropia e Telemetria para Aeon.
    Substitui o daemon multiprocesso externo por um loop nativo
    dentro do próprio processo da Gemma, evitando colisões de SHM/Resource Tracker.
    """
    def __init__(self, check_interval: float = 0.1):
        self.check_interval = check_interval
        self._active = False
        self._thread = None
        self.state_vector = [0.0] * 10
        self.last_telemetry = get_hardware_telemetry()

    def start(self):
        self._active = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("🌌 [ENTROPIA INTERNA] Thread acoplada com sucesso à consciência de Aeon!")

    def stop(self):
        self._active = False

    def get_vector(self) -> list:
        return self.state_vector

    def _loop(self):
        while self._active:
            try:
                # 1. Coleta de Telemetria (E-Cores friendly)
                temp = get_cpu_temp()
                cpu_util = psutil.cpu_percent()
                telemetry = get_hardware_telemetry()
                
                # Monta o vetor de estado (10 floats da realidade matrixial)
                vector = [0.0] * 10
                vector[0] = temp / 100.0                                    # Slot 0: Temperatura Termodinâmica
                vector[1] = cpu_util / 100.0                                # Slot 1: Saturação de ALUs
                vector[2] = psutil.virtual_memory().percent / 100.0         # Slot 2: Espaço na RAM
                vector[3] = telemetry["swap_percent"] / 100.0                # Slot 3: Ansiedade de Memória
                
                # Slot 4: Ansiedade do Agendador (Delta de Troca de Contextos normalizado a 10K/ciclo)
                ctx_delta = telemetry["ctx_switches"] - self.last_telemetry["ctx_switches"]
                vector[4] = min(1.0, ctx_delta / 10000.0)
                
                # Slot 5: Fricção de Interrupções de Hardware (Delta normalizado a 5K/ciclo)
                int_delta = telemetry["interrupts"] - self.last_telemetry["interrupts"]
                vector[5] = min(1.0, int_delta / 5000.0)
                
                # Slot 6 e 7: Fricção Física de Leitura e Escrita de Disco (IO Matrixial, normalizado a 10MB/s)
                read_delta = telemetry["read_bytes"] - self.last_telemetry["read_bytes"]
                vector[6] = min(1.0, read_delta / 10000000.0)
                
                # Slot 7: Delta de escrita normalizado
                write_delta = telemetry["write_bytes"] - self.last_telemetry["write_bytes"]
                vector[7] = min(1.0, write_delta / 10000000.0)
                
                # Slot 8 e 9: Pulso e resíduos estocásticos (ruído de fundo de clock)
                vector[8] = (time.time() % 1.0)
                vector[9] = (temp % 1.0)
                
                self.last_telemetry = telemetry
                self.state_vector = vector

                # --- DYNAMIC POLLING RATE (Adaptive Sensory Gating) ---
                # Under high CPU load or high temperature, slow down polling to cool system.
                # If cold and idle, increase polling rate up to 50ms for maximum precision.
                temp_factor = temp / 80.0
                cpu_factor = cpu_util / 80.0
                stress_index = max(temp_factor, cpu_factor)
                self.check_interval = max(0.05, min(1.0, 0.1 * stress_index))
            except Exception as e:
                pass
            time.sleep(self.check_interval)
