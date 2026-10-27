#!/bin/bash
echo "=== ESCANEANDO PORTAS (8000, 8022, 8800) EM REDES 10.0.0.x E 192.168.10.x ==="
python3 -c "
import socket, concurrent.futures

ports = [8000, 8022, 8800]
subnets = ['10.0.0', '192.168.10']

def check(ip_port):
    ip, port = ip_port
    try:
        s = socket.socket()
        s.settimeout(0.15)
        if s.connect_ex((ip, port)) == 0:
            s.close()
            return f'{ip}:{port}'
        s.close()
    except:
        pass
    return None

targets = []
for sub in subnets:
    for i in range(1, 255):
        for p in ports:
            targets.append((f'{sub}.{i}', p))

with concurrent.futures.ThreadPoolExecutor(max_workers=100) as ex:
    results = [r for r in ex.map(check, targets) if r]

print('Portas encontradas:', results)
"
