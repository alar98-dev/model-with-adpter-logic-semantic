import socket
import json
import threading
import time
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.text import Text
from rich.align import Align

# Estado global (Compartilhado entre UDP e UI)
state = {
    "thought_stream": "",
    "entropy": 0.0,
    "degraded_count": 0,
    "anomalies": [],
    "resonances": [],
    "q_quotient": 256,
    "status": "Aguardando conexão...",
    "last_update": time.time()
}

UDP_IP = "127.0.0.1"
UDP_PORT = 20005

def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    
    while True:
        data, addr = sock.recvfrom(65535)
        try:
            msg = json.loads(data.decode('utf-8'))
            msg_type = msg.get("type")
            
            state["last_update"] = time.time()
            state["status"] = f"🟢 Ativo (SWA-CORE) - Último pacote: {time.strftime('%H:%M:%S')}"
            
            if msg_type == "thought_stream":
                state["thought_stream"] = msg.get("content", "")
                state["entropy"] = msg.get("entropy", state["entropy"])
            elif msg_type == "chaos_metric":
                state["degraded_count"] = msg.get("count", state["degraded_count"])
                token = msg.get("token", "")
                if token:
                    state["anomalies"].insert(0, f"[{time.strftime('%H:%M:%S')}] {token}")
                    state["anomalies"] = state["anomalies"][:15] # Keep last 15
                state["entropy"] = msg.get("entropy", state["entropy"])
            elif msg_type == "spatial_resonance":
                filepath = msg.get("filepath", "")
                sim = msg.get("sim", 0.0)
                if filepath:
                    state["resonances"].insert(0, f"[{time.strftime('%H:%M:%S')}] {filepath} (Res: {sim:.4f})")
                    state["resonances"] = state["resonances"][:10]
            elif msg_type == "telemetry":
                state["q_quotient"] = msg.get("q", state["q_quotient"])
                
        except Exception as e:
            pass

def generate_layout() -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    
    layout["main"].split_row(
        Layout(name="left_panel", ratio=2),
        Layout(name="right_panel", ratio=1)
    )
    
    layout["right_panel"].split_column(
        Layout(name="chaos", ratio=1),
        Layout(name="spatial", ratio=1)
    )
    return layout

def get_renderable(layout: Layout) -> Layout:
    # Cabeçalho
    header_text = Text(f"🧠 SWA HIPPOCAMPUS TERMINAL | {state['status']} | Entropia: {state['entropy']:.4f} | Q: {state['q_quotient']}", style="bold cyan")
    layout["header"].update(Panel(Align.center(header_text, vertical="middle"), style="blue"))
    
    # Pensamento Principal
    thought_panel = Panel(
        Text(state["thought_stream"], justify="left", style="white"),
        title="[bold green]Fluxo de Pensamento (Córtex)[/]",
        border_style="green"
    )
    layout["left_panel"].update(thought_panel)
    
    # Métrica de Caos
    chaos_lines = "\n".join(state["anomalies"]) if state["anomalies"] else "Nenhuma anomalia capturada."
    chaos_text = f"Tokens Degradados Atuais: {state['degraded_count']}\n\n{chaos_lines}"
    
    chaos_panel = Panel(
        Text(chaos_text, style="red"),
        title="[bold red]Métrica de Caos (Anomalias)[/]",
        border_style="red"
    )
    layout["chaos"].update(chaos_panel)
    
    # Ressonância Espacial (MMAP)
    spatial_lines = "\n".join(state["resonances"]) if state["resonances"] else "Nenhum esbarrão mmap detectado recentemente."
    spatial_panel = Panel(
        Text(spatial_lines, style="magenta"),
        title="[bold magenta]Ressonância Espacial (MMAP)[/]",
        border_style="magenta"
    )
    layout["spatial"].update(spatial_panel)
    
    # Rodapé
    footer_text = Text(f"Aeon V5 - {time.strftime('%Y-%m-%d %H:%M:%S')} | Monitoramento em Tempo Real", style="dim")
    layout["footer"].update(Panel(Align.center(footer_text, vertical="middle"), style="dim"))
    
    return layout

if __name__ == "__main__":
    t = threading.Thread(target=udp_listener, daemon=True)
    t.start()
    
    console = Console()
    layout = generate_layout()
    
    with Live(get_renderable(layout), console=console, refresh_per_second=4, screen=True) as live:
        while True:
            live.update(get_renderable(layout))
            time.sleep(0.25)
