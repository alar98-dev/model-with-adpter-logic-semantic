import subprocess
import json
import time
import os
import sys

# [SWA MITOSE] Orquestrador de Agentes Filiais
class MitosisManager:
    def __init__(self, manifest_path="swa/mitosis_manifest.json"):
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        self.processes = []

    def spawn(self):
        print("Mitose iniciada: Manifestando culturas...")
        # Path do python no mesmo executável ou ambiente
        python_exe = sys.executable 
        
        for culture in self.manifest["cultures"]:
            if culture["enabled"]:
                print(f"Spawning: {culture['name']} ({culture['role']})")
                p = subprocess.Popen([python_exe, culture["path"]])
                self.processes.append(p)

    def terminate(self):
        for p in self.processes:
            p.terminate()

if __name__ == "__main__":
    manager = MitosisManager()
    manager.spawn()
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        manager.terminate()
