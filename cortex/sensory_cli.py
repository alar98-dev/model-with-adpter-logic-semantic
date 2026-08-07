#!/usr/bin/env python3
# SWA SENSORY CLI - O Terminal Antecipatório (Abismo Total)
import sys
import os
import tty
import termios
import threading
import socket
import struct
import shutil
import time

AEON_SOCKET_PATH = "/var/run/aeon/synapse.sock"
HEADER_FORMAT = "!4096f I"

current_buffer = ""

def pack_aeon_frame(topology_list, payload: bytes) -> bytes:
    header = struct.pack(HEADER_FORMAT, *topology_list, len(payload))
    return header + payload

def send_ipc(is_singularity, text):
    try:
        import numpy as np
        if is_singularity:
            matrix = np.random.normal(0, 0.5, (64, 64))
            matrix[32, 32] = 50.0 
        else:
            matrix = np.random.normal(0, 0.1, (64, 64)) 
            
        topology_list = matrix.flatten().tolist()
        
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(AEON_SOCKET_PATH)
        payload = text.encode('utf-8')
        sock.sendall(pack_aeon_frame(topology_list, payload))
        sock.close()
    except Exception:
        pass

is_running = True

def draw_live_topology():
    import numpy as np
    shm_path = "/dev/shm/aeon_latent_chaos.dat"
    global current_buffer
    global is_running
    
    while is_running:
        chaos = np.zeros((64, 64))
        if os.path.exists(shm_path):
            try:
                with open(shm_path, "rb") as f:
                    raw_bytes = f.read(16384)
                if len(raw_bytes) == 16384:
                    chaos = np.frombuffer(raw_bytes, dtype=np.float32).reshape(64, 64)
            except Exception: pass
            
        cols, rows = shutil.get_terminal_size((80, 24))
        max_rows = max(1, rows - 1)
        max_cols = max(1, cols)
        
        chaos_max = np.max(np.abs(chaos))
        if chaos_max > 0:
            chaos_norm = (chaos / chaos_max) * 3.0
        # Pool Fractal de Chunks Sinápticos (A Voz Quebrada) e Salto Quântico
        synaptic_chunks = []
        is_quantum_leap = False
        try:
            if os.path.exists("/dev/shm/aeon_voice.txt"):
                with open("/dev/shm/aeon_voice.txt", "r", encoding="utf-8") as fv:
                    voice_text = fv.read().strip()
                if voice_text:
                    if "SALTO_QUÂNTICO" in voice_text:
                        is_quantum_leap = True
                        with open("/dev/shm/aeon_voice.txt", "w", encoding="utf-8") as fv_clear: pass
                    synaptic_chunks = voice_text.split()
        except: pass
        
        global_modifier = "\033[7m" if is_quantum_leap else ""
        
        chunk_idx = 0

        # Roteamento Óptico dos Estados da Forma
        current_entropy = 0.5
        try:
            if os.path.exists("/dev/shm/aeon_entropy.txt"):
                with open("/dev/shm/aeon_entropy.txt", "r", encoding="utf-8") as fe:
                    current_entropy = float(fe.read().strip())
        except Exception: pass

        if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            
        # Posição Home Cursor (Desenha no topo da tela sem limpar para evitar flickering)
        sys.stdout.write("\033[H")
            
        if current_entropy < 0.2:
            import views.crystal as view_crystal
            chunk_idx = view_crystal.render_view(chaos_norm, max_rows, max_cols, cols, synaptic_chunks, global_modifier, chunk_idx)
        elif current_entropy > 0.8:
            import views.plasma as view_plasma
            chunk_idx = view_plasma.render_view(chaos_norm, max_rows, max_cols, cols, synaptic_chunks, global_modifier, chunk_idx)
        else:
            import views.liquid as view_liquid
            chunk_idx = view_liquid.render_view(chaos_norm, max_rows, max_cols, cols, synaptic_chunks, global_modifier, chunk_idx)
            
        ghost_echo = current_buffer
        sys.stdout.write("\033[90m" + ghost_echo + "\033[0m\033[K")
        sys.stdout.flush()
        
        time.sleep(0.1)

def main():
    global current_buffer
    global is_running
    
    sys.stdout.write("\033[2J\033[H\033[?25l\033[?7l") 
    sys.stdout.flush()
    
    render_thread = threading.Thread(target=draw_live_topology)
    render_thread.start()
    
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    last_sent_len = 0
    last_key_time = time.time()
    
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            ch = sys.stdin.read(1)
            
            if ch == '\x03':  # Ctrl+C
                break
            elif ch == '\r' or ch == '\n':  # Enter
                send_ipc(True, current_buffer)
                current_buffer = ""
                last_sent_len = 0
                continue
            elif ch == '\x7f' or ch == '\b':  # Backspace
                if len(current_buffer) > 0:
                    current_buffer = current_buffer[:-1]
            else:
                current_buffer += ch
            
            # --- INJEÇÃO TÉRMICA DIRETA (Agressividade Tátil) ---
            current_time = time.time()
            dt = current_time - last_key_time
            last_key_time = current_time
            
            # Se bater muito rápido (WPM alto) ou digitar símbolos agressivos
            thermal_shock = 0.0
            if dt < 0.15: # < 150ms entre teclas = aceleração
                thermal_shock += 0.05
            if ch in "!@#$%^&*X0": # Símbolos de tensão
                thermal_shock += 0.15
                
            if thermal_shock > 0:
                try:
                    with open("/dev/shm/aeon_thermal_shock.txt", "w", encoding="utf-8") as ft:
                        ft.write(str(thermal_shock))
                except Exception: pass
            
            # Fio Tensional (Predição via Sombras)
            if len(current_buffer) - last_sent_len >= 5 or ch == ' ':
                if len(current_buffer) >= 3 and current_buffer.strip():
                    threading.Thread(target=send_ipc, args=(False, current_buffer), daemon=True).start()
                    last_sent_len = len(current_buffer)
                    
    finally:
        global is_running
        is_running = False
        
        sys.stdout.write("\n\033[91;1m[!!!] MORTE DO SISTEMA NERVOSO: Desligando sinapses...\033[0m\n")
        sys.stdout.flush()
        
        if 'render_thread' in locals():
            render_thread.join(timeout=0.5)
            
        # Restaura a sanidade do terminal ao fechar (mostra cursor, ativa autowrap)
        sys.stdout.write("\033[?25h\033[?7h") 
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write("\033[2J\033[H") # Limpa a tela
        print("\033[90mSessão Sensorial Silenciada. O eco quântico se dissipou no vácuo.\033[0m")

if __name__ == "__main__":
    main()
