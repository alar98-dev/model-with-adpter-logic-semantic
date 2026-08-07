import time
import struct
import sys
import psutil

def main():
    while True:
        try:
            # 1. CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_freq = psutil.cpu_freq().current if psutil.cpu_freq() else 0.0
            
            # 2. RAM
            ram_percent = psutil.virtual_memory().percent
            
            # 3. Disk IO (Deltas simples - abstraídos para escala 0.0 ~ 1.0 heurística)
            disk = psutil.disk_io_counters()
            disk_read = float(disk.read_count % 1000) / 1000.0 if disk else 0.0
            disk_write = float(disk.write_count % 1000) / 1000.0 if disk else 0.0
            
            # 4. Net IO
            net = psutil.net_io_counters()
            net_sent = float(net.bytes_sent % 10000) / 10000.0 if net else 0.0
            net_recv = float(net.bytes_recv % 10000) / 10000.0 if net else 0.0
            
            # 5. Entropia / Handles / Custom
            handles = float(len(psutil.pids()) % 1000) / 1000.0
            
            # Empacotando 8 floats (32 bytes)
            packed_data = struct.pack(
                '8f',
                cpu_percent / 100.0,  # Fator normalizado
                cpu_freq / 5000.0,    # Aproximação de freq max
                ram_percent / 100.0,
                disk_read,
                disk_write,
                net_sent,
                net_recv,
                handles
            )
            
            # Escreve no Stdout Buffer de forma binária e FAZ FLUSH (Crítico para IPC)
            sys.stdout.buffer.write(packed_data)
            sys.stdout.buffer.flush()
            
            # Loop a 10Hz (0.1s)
            time.sleep(0.1)
            
        except Exception as e:
            # Em caso de falha de I/O (Pipes Quebrados), o coletor silencia e encerra
            sys.exit(1)

if __name__ == "__main__":
    main()
