import threading
import sys
import os
import logging
import traceback
import time

# [SWA ROBUSTNESS] Configuração de log
logging.basicConfig(
    filename='aeon_error.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# [SWA FOCO DINÂMICO] Autodescoberta de Caminhos
def align_structural_paths():
    swa_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(swa_dir)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

align_structural_paths()

def run_bridge():
    from swa.pygame_bridge import PygameBridge
    bridge = PygameBridge(monitor_id=0)
    bridge.run()

def run_mitosis():
    from swa.mitosis_manager import MitosisManager
    manager = MitosisManager()
    manager.spawn()

def run_monitor():
    from swa.sensors.input_monitor import InputMonitor
    monitor = InputMonitor()
    monitor.start()

if __name__ == "__main__":
    try:
        # 1. Spawn Filiais (Mitose)
        threading.Thread(target=run_mitosis, daemon=True).start()
        # 2. Start input monitor
        threading.Thread(target=run_monitor, daemon=True).start()
        # 3. Main Body
        run_bridge()
    except Exception:
        err = traceback.format_exc()
        logging.critical(f"Erro Fatal na Célula Aeon: {err}")
        print(f"--- CRITICAL ERROR ---\n{err}")
        input("Pressione Enter...")
        sys.exit(1)
