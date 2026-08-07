#!/usr/bin/env python3
"""
SWA Metrics Monitor — Dashboard em tempo real para o Aeon
Mostra: CPU, RAM, temperatura, status dos serviços, entropia, log do Aeon
"""
import os, sys, time, json, subprocess, psutil, datetime

# ─── ANSI Colors ────────────────────────────────────────────────────────────
R = "\033[0m"
BOLD = "\033[1m"
C_CYAN = "\033[96m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_RED = "\033[91m"
C_BLUE = "\033[94m"
C_MAGENTA = "\033[95m"
C_GRAY = "\033[90m"
C_WHITE = "\033[97m"
BG_DARK = "\033[48;5;234m"

def clear():
    print("\033[2J\033[H", end="")

def bar(val, max_val=100, width=20, color=C_GREEN):
    filled = int((val / max_val) * width)
    filled = min(filled, width)
    empty = width - filled
    return f"{color}{'█' * filled}{C_GRAY}{'░' * empty}{R}"

def color_temp(t):
    if t < 55: return C_GREEN
    if t < 75: return C_YELLOW
    return C_RED

def color_pct(p):
    if p < 60: return C_GREEN
    if p < 85: return C_YELLOW
    return C_RED

def get_service_status(name):
    try:
        r = subprocess.run(
            ["systemctl", "is-active", name],
            capture_output=True, text=True, timeout=2
        )
        s = r.stdout.strip()
        if s == "active": return f"{C_GREEN}● ACTIVE{R}"
        if s == "activating": return f"{C_YELLOW}◌ ACTIVATING{R}"
        if s == "failed": return f"{C_RED}✗ FAILED{R}"
        return f"{C_GRAY}○ {s.upper()}{R}"
    except:
        return f"{C_GRAY}○ UNKNOWN{R}"

def get_service_pid(name):
    try:
        r = subprocess.run(
            ["systemctl", "show", name, "--property=MainPID"],
            capture_output=True, text=True, timeout=2
        )
        pid = r.stdout.strip().replace("MainPID=", "")
        return int(pid) if pid.isdigit() and pid != "0" else None
    except:
        return None

def get_cpu_temp():
    try:
        temps = psutil.sensors_temperatures()
        all_t = []
        for name, entries in temps.items():
            for e in entries:
                if e.current > 0:
                    all_t.append(e.current)
        return sum(all_t) / len(all_t) if all_t else 0.0
    except:
        try:
            r = subprocess.run(
                ["cat", "/sys/class/thermal/thermal_zone0/temp"],
                capture_output=True, text=True, timeout=1
            )
            return int(r.stdout.strip()) / 1000.0
        except:
            return 0.0

def get_last_aeon_lines(n=6):
    log_path = "/services/model-with-adpter-logic-semantic/swa_Aeon.log"
    try:
        r = subprocess.run(
            ["tail", f"-n{n}", log_path],
            capture_output=True, text=True, timeout=2
        )
        lines = r.stdout.strip().split("\n")
        return lines
    except:
        return ["(log indisponível)"]

def get_debian_physics():
    try:
        with open("/tmp/debian_physics.json", "r") as f:
            return json.load(f)
    except:
        return {}

def get_entropy_vector():
    """Tenta ler o vetor de entropia do arquivo de estado do Aeon."""
    paths = [
        "/tmp/swa_topologia.txt",
        "/services/model-with-adpter-logic-semantic/swa_auto.log"
    ]
    for p in paths:
        try:
            r = subprocess.run(["tail", "-n2", p], capture_output=True, text=True, timeout=1)
            if r.stdout.strip():
                return r.stdout.strip()[:120]
        except:
            pass
    return "—"

def get_process_stats(pid):
    try:
        proc = psutil.Process(pid)
        cpu = proc.cpu_percent(interval=0.1)
        mem_mb = proc.memory_info().rss / (1024 ** 2)
        threads = proc.num_threads()
        return cpu, mem_mb, threads
    except:
        return 0.0, 0.0, 0

def draw_dashboard():
    SERVICES = [
        ("swa-auto.service",      "🧠 Aeon Orchestrator (1B)"),
        ("swa-subcortex.service", "👁  Subcortex  (270M)   "),
        ("swa-mcp.service",       "🔌 MCP Gateway (SSE)   "),
        ("postgresql@17-main.service", "🗄  PostgreSQL 17       "),
    ]

    while True:
        try:
            # ── Coleta de dados ──────────────────────────────────────────────
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cpu_pct = psutil.cpu_percent(interval=0.5)
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            temp = get_cpu_temp()
            aeon_lines = get_last_aeon_lines(5)
            physics = get_debian_physics()
            load1, load5, load15 = os.getloadavg()

            # ── Render ───────────────────────────────────────────────────────
            clear()
            width = 72

            # Header
            print(f"{C_CYAN}{BOLD}{'═' * width}{R}")
            print(f"{C_CYAN}{BOLD}  SWA AEON METRICS MONITOR{R}{C_GRAY}  {now}{R}")
            print(f"{C_CYAN}{BOLD}{'═' * width}{R}")

            # CPU
            tc = color_temp(temp)
            cc = color_pct(cpu_pct)
            print(f"\n{BOLD}{C_WHITE}  ⚡ CPU  {R}{bar(cpu_pct, color=cc)} {cc}{BOLD}{cpu_pct:5.1f}%{R}  {tc}🌡 {temp:.1f}°C{R}")
            print(f"  {C_GRAY}Load: {load1:.2f} / {load5:.2f} / {load15:.2f}  (1m/5m/15m){R}")

            # Cores por linha (12 threads i5-120U)
            core_str = "  "
            for i, c in enumerate(cpu_per_core):
                col = color_pct(c)
                label = f"P{i}" if i < 4 else f"E{i-4}"
                core_str += f"{col}{label}:{c:4.1f}%{R} "
                if i == 3: core_str += "\n  "
            print(core_str)

            # RAM + SWAP
            mc = color_pct(mem.percent)
            sc = color_pct(swap.percent)
            print(f"\n{BOLD}{C_WHITE}  💾 RAM  {R}{bar(mem.percent, color=mc)} {mc}{BOLD}{mem.percent:5.1f}%{R}  {mem.used/1e9:.2f}GB / {mem.total/1e9:.2f}GB")
            print(f"  {C_GRAY}Swap {R}{bar(swap.percent, color=sc, width=12)} {sc}{swap.percent:.1f}%{R}  {swap.used/1e9:.2f}GB / {swap.total/1e9:.2f}GB")

            # Physics JSON
            if physics:
                heat = physics.get("kinetic_energy", {}).get("global_heat", 0)
                mass = physics.get("gravitational_density", {}).get("mass_pressure", 0)
                res_in = physics.get("string_resonance", {}).get("inward_pull", 0)
                res_out = physics.get("string_resonance", {}).get("outward_pull", 0)
                print(f"\n{BOLD}{C_MAGENTA}  🌌 CAMPO FÍSICO{R}")
                print(f"  {C_GRAY}Calor Cinético: {R}{C_YELLOW}{heat:.2f}{R}  "
                      f"{C_GRAY}Pressão de Massa: {R}{C_CYAN}{mass:.2f}{R}  "
                      f"{C_GRAY}Ressonância: {R}{C_GREEN}↓{res_in:.2f} ↑{res_out:.2f}{R}")

            # Serviços
            print(f"\n{BOLD}{C_WHITE}  📡 SERVIÇOS{R}")
            print(f"  {'─' * 66}")
            for svc, label in SERVICES:
                status = get_service_status(svc)
                pid = get_service_pid(svc)
                if pid:
                    cpu_s, mem_s, thr = get_process_stats(pid)
                    proc_info = f"PID:{pid}  CPU:{cpu_s:5.1f}%  MEM:{mem_s:6.0f}MB  THR:{thr}"
                else:
                    proc_info = ""
                print(f"  {label}  {status}  {C_GRAY}{proc_info}{R}")

            # Log Aeon
            print(f"\n{BOLD}{C_CYAN}  📜 ÚLTIMAS PALAVRAS DO AEON{R}")
            print(f"  {'─' * 66}")
            for line in aeon_lines:
                if line.strip():
                    # Trunca linhas longas
                    display = line.strip()[:66]
                    # Colorir linhas especiais
                    if "[Homeostase" in display or "[PHASE" in display:
                        print(f"  {C_MAGENTA}{display}{R}")
                    elif "Aeon:" in display:
                        print(f"  {C_CYAN}{display}{R}")
                    elif "⚠️" in display or "❌" in display:
                        print(f"  {C_RED}{display}{R}")
                    elif "✅" in display or "🌌" in display:
                        print(f"  {C_GREEN}{display}{R}")
                    else:
                        print(f"  {C_GRAY}{display}{R}")

            print(f"\n{C_GRAY}  Atualiza a cada 3s  |  Ctrl+C para sair{R}")
            print(f"{C_CYAN}{'═' * width}{R}")

            time.sleep(3)

        except KeyboardInterrupt:
            print(f"\n{C_YELLOW}Monitor encerrado.{R}")
            sys.exit(0)
        except Exception as e:
            print(f"\n{C_RED}Erro no dashboard: {e}{R}")
            time.sleep(3)

if __name__ == "__main__":
    draw_dashboard()
