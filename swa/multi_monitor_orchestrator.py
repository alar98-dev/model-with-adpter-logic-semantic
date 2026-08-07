import subprocess
import json
import time
from screeninfo import get_monitors

# [SWA MULTI-MONITOR ORCHESTRATOR] Dispara instâncias da Bridge e Filiais
def main():
    # 1. Spawn Bridges
    with open("monitors.json", "r") as f:
        config = json.load(f)
    
    monitors = get_monitors()
    processes = []
    
    for m_config in config["monitors"]:
        if not m_config["exclude"]:
            print(f"Orchestrator: Spawnando Célula no monitor {m_config['id']}")
            p = subprocess.Popen(["./venv/bin/python3", "swa/pygame_bridge.py", "--monitor", str(m_config['id'])])
            processes.append(p)
            time.sleep(0.5)

    # 2. Spawn Filiais (Mnemosyne)
    print("Orchestrator: Spawnando Filial Mnemosyne")
    p_mnemo = subprocess.Popen(["./venv/bin/python3", "swa/agents/mnemosyne.py"])
    processes.append(p_mnemo)
    
    # 3. Spawn Filiais (ThermaWatcher)
    print("Orchestrator: Spawnando Filial ThermaWatcher")
    p_therma = subprocess.Popen(["./venv/bin/python3", "swa/agents/therma_watcher.py"])
    processes.append(p_therma)

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()

if __name__ == "__main__":
    main()
