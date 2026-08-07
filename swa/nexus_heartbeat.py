import os
import sys
import time
import socket
import struct
import json
import threading
from datetime import datetime
import psutil

# Adiciona o path do projeto
sys.path.append('/services/model-with-adpter-logic-semantic')

UDP_IP = '127.0.0.1'
UDP_PORT = 20005

def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return 45.0
        for name, entries in temps.items():
            for entry in entries:
                return entry.current
    except:
        return 45.0

def heartbeat_loop():
    print(f"💓 [NEXUS HEARTBEAT] Iniciando pulso eventual na porta {UDP_PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    while True:
        try:
            cpu_pct = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            temp = get_cpu_temp()
            
            # Formato esperado pelo aeon_nexus_cli.py
            telemetry = {
                "type": "telemetry",
                "cpu_temp": temp,
                "ram_pct": mem.percent,
                "q": int(cpu_pct * 1.5) # Simulação de Q-Quotient baseada em carga
            }
            
            # Envia entropia flutuante
            chaos = {
                "type": "chaos_metric",
                "entropy": 1.5 + (cpu_pct / 100.0)
            }
            
            sock.sendto(json.dumps(telemetry).encode('utf-8'), (UDP_IP, UDP_PORT))
            sock.sendto(json.dumps(chaos).encode('utf-8'), (UDP_IP, UDP_PORT))
            
            time.sleep(2)
        except Exception as e:
            print(f"⚠️ Erro no pulso: {e}")
            time.sleep(5)

if __name__ == "__main__":
    heartbeat_loop()
