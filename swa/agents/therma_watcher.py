import time
import psutil
from swa.sub_agent_base import SubAgentBase

class ThermaWatcher(SubAgentBase):
    """
    [FILIAL: THERMAWATCHER]
    Monitora temperatura e carga para garantir a homeostase do Córtex.
    """
    def __init__(self):
        super().__init__("ThermaWatcher")
        self.temp_threshold = 80.0

    def process_cycle(self):
        # Coleta telemetria física
        try:
            temps = psutil.sensors_temperatures()
            temp_avg = 0.0
            if temps:
                all_temps = [entry.current for entries in temps.values() for entry in entries]
                temp_avg = sum(all_temps) / len(all_temps) if all_temps else 0.0
            
            cpu_usage = psutil.cpu_percent()
            mem_usage = psutil.virtual_memory().percent
            
            metrics = {
                "temp_c": temp_avg,
                "cpu_percent": cpu_usage,
                "ram_percent": mem_usage
            }
            
            # Decisão de Homeostase
            status = "STABLE"
            if temp_avg > self.temp_threshold:
                status = "THERMAL_THROTTLING_ADVISORY"
            elif cpu_usage > 90.0:
                status = "HIGH_LOAD_ADVISORY"
                
            self.report_status(status, metrics)
            
        except Exception as e:
            self.logger.error(f"Falha na telemetria: {e}")
            self.report_status("ERROR", {"error": str(e)})

if __name__ == "__main__":
    watcher = ThermaWatcher()
    watcher.run()
