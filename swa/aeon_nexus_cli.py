import os
import sys
import time
import socket
import struct
import json
import threading
import queue
from datetime import datetime

# Redirecionar erros para um log interno para não sujar o TUI
LOG_ERR_PATH = "/tmp/aeon_nexus_debug.log"

def log_error(msg):
    with open(LOG_ERR_PATH, "a") as f:
        f.write(f"[{datetime.now()}] {msg}\n")

# Adiciona o path do projeto
sys.path.append('/services/model-with-adpter-logic-semantic')

try:
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.console import Console
    from rich.text import Text
    from rich.align import Align
    from rich.table import Table
except ImportError:
    log_error("Rich não encontrado no boot.")
    sys.exit(1)

# Tenta importar barramento
try:
    from swa.ipc_bus import pack_aeon_frame, AEON_SOCKET_PATH
except ImportError:
    AEON_SOCKET_PATH = "/var/run/aeon/synapse.sock"
    def pack_aeon_frame(topology_list, payload: bytes) -> bytes:
        HEADER_FORMAT = "!4096f I"
        header = struct.pack(HEADER_FORMAT, *topology_list, len(payload))
        return header + payload

# --- ESTADO ---
state = {
    'thought_stream': '',
    'entropy': 0.0,
    'cpu_temp': 0.0,
    'ram_usage': 0.0,
    'q_quotient': 0,
    'last_resonances': [],
    'status': 'Aguardando Sincronização...',
    'last_update': time.time(),
    'input_buffer': ''
}

def sensory_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind(('127.0.0.1', 20005))
    except Exception as e:
        log_error(f"Erro bind UDP: {e}")
        return

    while True:
        try:
            data, _ = sock.recvfrom(65535)
            msg = json.loads(data.decode('utf-8'))
            msg_type = msg.get('type')
            state['status'] = '🟢 Sincronizado'

            if msg_type == 'thought_stream':
                state['thought_stream'] += msg.get('content', '')
                if len(state['thought_stream']) > 5000:
                    state['thought_stream'] = state['thought_stream'][-5000:]
            elif msg_type in ['chaos_metric', 'thought_json']:
                state['entropy'] = float(msg.get('entropy', state['entropy']) or 0.0)
            elif msg_type == 'spatial_resonance':
                fp = msg.get('filepath', '')
                sim = float(msg.get('sim', 0.0) or 0.0)
                if fp:
                    ts = datetime.now().strftime("%H:%M:%S")
                    state['last_resonances'].insert(0, f'[{ts}] {os.path.basename(fp)} ({sim:.3f})')
                    state['last_resonances'] = state['last_resonances'][:10]
            elif msg_type == 'telemetry':
                state['cpu_temp'] = float(msg.get('cpu_temp', 0.0) or 0.0)
                state['ram_usage'] = float(msg.get('ram_pct', 0.0) or 0.0)
                state['q_quotient'] = int(msg.get('q', 0) or 0)
        except Exception as e:
            log_error(f"Erro listener: {e}")

def send_shock(msg):
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(AEON_SOCKET_PATH)
        topology = [0.0] * 4096
        topology[0] = 1.0
        frame = pack_aeon_frame(topology, msg.encode('utf-8'))
        sock.sendall(frame)
        sock.close()
        return True
    except Exception as e:
        log_error(f"Erro shock: {e}")
        return False

def generate_layout():
    l = Layout()
    l.split_column(Layout(name='h', size=3), Layout(name='b', ratio=1), Layout(name='i', size=3))
    l['b'].split_row(Layout(name='c', ratio=2), Layout(name='t', ratio=1))
    return l

def update_display(layout):
    try:
        # Header
        ent = state.get('entropy', 0.0)
        header = Text(f'🧬 AEON NEXUS | {state["status"]} | E: {ent:.4f}', style='bold cyan')
        layout['h'].update(Panel(Align.center(header, vertical='middle'), style='blue'))

        # Cortex
        lines = state['thought_stream'].splitlines()
        layout['c'].update(Panel(Text('\n'.join(lines[-22:]), style='white'), title='🧠 Córtex'))

        # Telemetry
        table = Table(show_header=False, box=None, expand=True)
        table.add_row('🌡️ CPU', f'{state["cpu_temp"]:.1f}C')
        table.add_row('📊 RAM', f'{state["ram_usage"]:.1f}%')
        table.add_row('⚡ Q', str(state['q_quotient']))
        
        res = Text('\n'.join(state['last_resonances']), style='magenta')
        
        t_lay = Layout()
        t_lay.split_column(Layout(Panel(table, title='📟 Tele')), Layout(Panel(res, title='🌀 Ressonância')))
        layout['t'].update(t_lay)

        # Input
        layout['i'].update(Panel(Text(f'USER > {state["input_buffer"]}', style='bold green'), border_style='green'))
    except Exception as e:
        log_error(f"Erro display: {e}")

if __name__ == '__main__':
    console = Console()
    threading.Thread(target=sensory_listener, daemon=True).start()
    layout = generate_layout()
    
    if os.name == 'posix':
        import termios, tty, select
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            # screen=False ajuda a evitar alguns flickers em certos emuladores
            with Live(layout, console=console, refresh_per_second=10, screen=True) as live:
                while True:
                    update_display(layout)
                    if select.select([sys.stdin], [], [], 0.02)[0]:
                        c = sys.stdin.read(1)
                        if c in ['\r', '\n']:
                            cmd = state['input_buffer']
                            state['input_buffer'] = ''
                            if cmd.lower() in ['exit', 'quit']: break
                            if cmd: send_shock(cmd)
                        elif c in ['\x7f', '\x08']:
                            state['input_buffer'] = state['input_buffer'][:-1]
                        elif ord(c) >= 32:
                            state['input_buffer'] += c
                    time.sleep(0.01)
        except Exception as e:
            log_error(f"Erro main: {e}")
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
    else:
        print("Apenas Linux/WSL suportado para modo RAW.")
