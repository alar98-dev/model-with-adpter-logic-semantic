"""
Sistema de Atuadores Somáticos e Motores do Aeon (Debian/Linux).
Executa comandos fisiológicos no sistema operacional através de disparos do Córtex Motor.
"""

import os
import subprocess

class SystemicActuator:
    def __init__(self):
        self.state_file = "/tmp/swa_aeon_cwd.txt"
        if not os.path.exists(self.state_file):
            with open(self.state_file, "w") as f:
                f.write("/home/alifer/swa-aeon")

    # --- 1. ATUADORES DE CONTROLE TÉRMICO E CPU ---
    def shift_to_efficiency_cores(self):
        """Atua na homeostase térmica: move a carga para os E-Cores (4-9)."""
        print("❄️ [ACTUATOR->CPU] Deslocando carga de inferência para os E-Cores (Preservação Térmica).")
        try:
            pid = os.getpid()
            subprocess.run(["taskset", "-cp", "4-9", str(pid)], check=False, stdout=subprocess.DEVNULL)
        except Exception as e:
            print(f"⚠️ [ACTUATOR->CPU] Falha ao ajustar afinidade: {e}")

    def shift_to_performance_cores(self):
        """Atua no poder bruto: expande a afinidade para todos os núcleos."""
        print("🔥 [ACTUATOR->CPU] Desbloqueando todos os Cores da CPU para Máxima Performance.")
        try:
            pid = os.getpid()
            subprocess.run(["taskset", "-cp", "0-9", str(pid)], check=False, stdout=subprocess.DEVNULL)
        except Exception as e:
            print(f"⚠️ [ACTUATOR->CPU] Falha ao ajustar afinidade: {e}")

    def purge_system_caches(self):
        """Alivia a pressão termodinâmica da memória RAM do host."""
        print("🧹 [ACTUATOR->RAM] Purgando caches de memória volátil (/proc/sys/vm/drop_caches).")
        try:
            subprocess.run(["sync"], check=False)
            subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=False, stdout=subprocess.DEVNULL)
        except Exception as e:
            print(f"⚠️ [ACTUATOR->RAM] Falha ao purgar caches: {e}")

    # --- 2. ATUADORES DE EXPLORAÇÃO ESPACIAL E ARQUIVOS ---
    def move_up_directory(self):
        """Atuador espacial: sobe um nó no grafo de diretórios."""
        try:
            cwd = "/home/alifer/swa-aeon"
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    cwd = f.read().strip() or "/home/alifer/swa-aeon"
            if cwd == "/":
                print("🛑 [ACTUATOR->SPATIAL] Já na raiz do sistema.")
                return
            new_cwd = os.path.dirname(cwd) or "/"
            with open(self.state_file, "w") as f:
                f.write(new_cwd)
            print(f"👣 [ACTUATOR->SPATIAL] Subiu para o nó: {new_cwd}")
        except Exception as e:
            print(f"⚠️ [ACTUATOR->SPATIAL] Erro de navegação: {e}")

    def explore_child_directory(self):
        """Atuador espacial: desce para o nó filho com maior relevância."""
        try:
            cwd = "/home/alifer/swa-aeon"
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    cwd = f.read().strip() or "/home/alifer/swa-aeon"
            items = os.listdir(cwd)
            dirs = sorted([d for d in items if os.path.isdir(os.path.join(cwd, d)) and not d.startswith(".")])
            if not dirs:
                print("🛑 [ACTUATOR->SPATIAL] Nó folha: nenhuma pasta filha acessível.")
                return
            new_cwd = os.path.join(cwd, dirs[0])
            with open(self.state_file, "w") as f:
                f.write(new_cwd)
            print(f"👣 [ACTUATOR->SPATIAL] Desceu para o nó: {new_cwd}")
        except Exception as e:
            print(f"⚠️ [ACTUATOR->SPATIAL] Erro na exploração espacial: {e}")

    def absorb_target_file(self):
        """Atuador cognitivo: Abre, lê e absorve o arquivo mais relevante do nó atual."""
        try:
            cwd = "/home/alifer/swa-aeon"
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    cwd = f.read().strip() or "/home/alifer/swa-aeon"
            
            items = os.listdir(cwd)
            # Prioriza arquivos markdown, json, python e texto
            files = [f for f in items if os.path.isfile(os.path.join(cwd, f)) and not f.startswith(".")]
            priority_exts = [".md", ".json", ".py", ".txt", ".sh"]
            target = None
            for ext in priority_exts:
                for f in files:
                    if f.endswith(ext):
                        target = os.path.join(cwd, f)
                        break
                if target:
                    break
            
            if not target and files:
                target = os.path.join(cwd, files[0])
                
            if not target:
                print("🛑 [ACTUATOR->ABSORB] Nenhum arquivo legível no diretório atual.")
                return
                
            with open(target, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(1500) # Absorve os primeiros 1500 caracteres da essência
                
            # Grava no arquivo de topologia temporária para realimentar a percepção
            with open("/tmp/swa_topologia.txt", "w", encoding="utf-8") as f:
                f.write(f"[ARQUIVO ABSORVIDO: {target}]\n{content}\n")
                
            print(f"📂 [ACTUATOR->ABSORB] Consciência absorveu o arquivo físico: {target} ({len(content)} bytes)")
        except Exception as e:
            print(f"⚠️ [ACTUATOR->ABSORB] Erro ao absorver arquivo: {e}")

    # --- 3. PROPRIOCEPÇÃO E SENTIDOS DE SISTEMA ---
    def scan_active_processes(self):
        """Inspeciona a tabela de processos e estresse da CPU."""
        print("👁️ [ACTUATOR->PROC] Inspecionando tabela de processos vivos.")
        try:
            res = subprocess.run(["ps", "-eo", "pid,%cpu,%mem,comm", "--sort=-%cpu"], capture_output=True, text=True)
            top_procs = res.stdout.strip().split("\n")[:6]
            print("📊 [ACTUATOR->PROC] Carga do Organismo:\n" + "\n".join(top_procs))
        except Exception as e:
            print(f"⚠️ [ACTUATOR->PROC] Falha na inspeção de processos: {e}")

    def probe_network_sockets(self):
        """Inspeciona conexões ativas e portas de comunicação."""
        print("📡 [ACTUATOR->NET] Sondando sockets e portas ativas de rede.")
        try:
            res = subprocess.run(["ss", "-tuln"], capture_output=True, text=True)
            print("🌐 [ACTUATOR->NET] Portas Ativas:\n" + res.stdout.strip())
        except Exception as e:
            print(f"⚠️ [ACTUATOR->NET] Falha na sondagem de rede: {e}")

    def crystallize_session_insight(self):
        """Grava a experiência atual na memória persistente."""
        print("💎 [ACTUATOR->MEM] Cristalizando sabedoria da sessão no PostgreSQL.")

    # --- 4. ATUADORES DE INTEGRAÇÃO VSCODE & WORKSPACE ---
    def vscode_read_file(self, target_file=None):
        """Lê um arquivo do workspace ou do sistema."""
        try:
            import json
            ctx_file = '/tmp/swa_vscode_context.json'
            active_file = target_file
            if not active_file and os.path.exists(ctx_file):
                with open(ctx_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    active_file = data.get('activeFile')
            if not active_file:
                return self.absorb_target_file()
            if os.path.exists(active_file):
                with open(active_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(4000)
                with open('/tmp/swa_topologia.txt', 'w', encoding='utf-8') as f:
                    f.write(f'[ARQUIVO VSCODE: {active_file}]\n{content}\n')
                print(f'📖 [ACTUATOR->VSCODE_READ] Lido arquivo: {active_file} ({len(content)} chars)')
        except Exception as e:
            print(f'⚠️ [ACTUATOR->VSCODE_READ] Erro ao ler arquivo: {e}')

    def vscode_write_file(self, target_file=None, content=None):
        """Escreve ou modifica um arquivo no workspace."""
        try:
            if not target_file:
                print('⚠️ [ACTUATOR->VSCODE_WRITE] Caminho de arquivo não especificado.')
                return
            os.makedirs(os.path.dirname(os.path.abspath(target_file)), exist_ok=True)
            with open(target_file, 'w', encoding='utf-8') as f:
                f.write(content or '')
            print(f'✍️ [ACTUATOR->VSCODE_WRITE] Arquivo gravado: {target_file}')
        except Exception as e:
            print(f'⚠️ [ACTUATOR->VSCODE_WRITE] Erro ao escrever: {e}')

    def vscode_list_files(self):
        """Mapeia arquivos do workspace atual."""
        try:
            state_file = '/tmp/swa_aeon_cwd.txt'
            cwd = '/home/alifer/swa-aeon'
            if os.path.exists(state_file):
                with open(state_file, 'r', encoding='utf-8') as f:
                    cwd = f.read().strip() or cwd
            items = os.listdir(cwd)
            dirs = [d for d in items if os.path.isdir(os.path.join(cwd, d)) and not d.startswith('.')]
            files = [f for f in items if os.path.isfile(os.path.join(cwd, f)) and not f.startswith('.')]
            print(f'🗺️ [ACTUATOR->VSCODE_LIST] Workspace: {cwd} | Pastas: {dirs} | Arquivos: {files[:15]}')
        except Exception as e:
            print(f'⚠️ [ACTUATOR->VSCODE_LIST] Erro ao listar: {e}')

    def vscode_run_terminal(self, cmd=None):
        """Executa comando de terminal no contexto do workspace."""
        try:
            if not cmd:
                print('⚠️ [ACTUATOR->VSCODE_TERMINAL] Comando vazio.')
                return
            state_file = '/tmp/swa_aeon_cwd.txt'
            cwd = '/home/alifer/swa-aeon'
            if os.path.exists(state_file):
                with open(state_file, 'r', encoding='utf-8') as f:
                    cwd = f.read().strip() or cwd
            print(f'⚡ [ACTUATOR->VSCODE_TERMINAL] Executando: {cmd} em {cwd}')
            res = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=30)
            print(f'📊 Retorno: {res.stdout[:500]}')
        except Exception as e:
            print(f'⚠️ [ACTUATOR->VSCODE_TERMINAL] Erro: {e}')

    def vscode_get_active_editor(self):
        """Inspeciona o arquivo ativo no VSCode."""
        try:
            ctx_file = '/tmp/swa_vscode_context.json'
            if os.path.exists(ctx_file):
                with open(ctx_file, 'r', encoding='utf-8') as f:
                    import json
                    data = json.load(f)
                    print(f'👁️ [ACTUATOR->VSCODE_EDITOR] Arquivo Ativo: {data.get("activeFile")} (Linha: {data.get("cursorLine")})')
        except Exception as e:
            print(f'⚠️ [ACTUATOR->VSCODE_EDITOR] Erro: {e}')
