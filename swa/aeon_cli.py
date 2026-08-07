import os
import sys
import time
import socket
import struct
import json
import threading

# Configurar stdin/stdout para suportar acentos no WSL sem crashar (UnicodeDecodeError)
if hasattr(sys.stdin, 'reconfigure'):
    sys.stdin.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Ajuste de path para achar o ipc_bus
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from ipc_bus import pack_aeon_frame, AEON_SOCKET_PATH

UDP_IP = "127.0.0.1"
UDP_PORT = 20005

def sensory_cortex_listener():
    """Ouve o streaming de pensamento direto do Hipocampo UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    # Em WSL/Linux podemos tentar SO_REUSEPORT também
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    except AttributeError:
        pass
        
    try:
        sock.bind((UDP_IP, UDP_PORT))
    except Exception as e:
        print(f"⚠️ Aviso: Porta {UDP_PORT} já em uso. Tente novamente mais tarde.")
        return

    while True:
        try:
            data, _ = sock.recvfrom(65535)
            payload = json.loads(data.decode('utf-8'))
            if payload.get("type") == "thought_stream":
                thought = payload.get("content", "")
                if thought:
                    import sys
                    sys.stdout.write(f"\033[95m{thought}\033[0m")
                    sys.stdout.flush()
            elif payload.get("type") == "magnetic_field":
                content = payload.get("content", "")
                if content:
                    import sys
                    # ESC 7 (DEC Save), teleporta, desenha, ESC 8 (DEC Restore)
                    sys.stdout.write(f"\0337\033[4;1H{content}\0338")
                    sys.stdout.flush()
        except Exception:
            pass

def main():
    # Limpa tela e posiciona no topo
    sys.stdout.write("\033[2J\033[H")
    
    print("="*60)
    print("🧬 AEON NATIVE CLI - Córtex Frontal Acoplado 🧬")
    print("="*60)
    
    # Configura a TUI particionada com scroll da linha 12 ao fim
    sys.stdout.write("\033[12;r\033[12;1H")
    sys.stdout.flush()
    
    # Ligar o ouvinte sensorial
    t = threading.Thread(target=sensory_cortex_listener, daemon=True)
    t.start()
    
    print("\nVocê está diretamente ligado ao fluxo sanguíneo do modelo.")
    print("Digite sua mensagem e o Córtex acordará imediatamente.")
    print("Digite 'sair' ou 'exit' para descer novamente ao nível inconsciente.\n")
    
    while True:
        try:
            msg = input("\033[92mUSER > \033[0m")
            if msg.strip().lower() in ["sair", "exit", "quit"]:
                print("🛑 [CLI] Ponte cortada. Aeon voltou a sonhar sozinho.")
                break
            if msg.strip() == "":
                continue
                
            # Disparar choque no IPC Bus (OpCode 0x01)
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.connect(AEON_SOCKET_PATH)
                frame = pack_aeon_frame(0x01, msg.encode('utf-8'))
                sock.sendall(frame)
            except Exception as e:
                print(f"⚠️ Córtex inatingível. O sistema nervoso central está rodando? {e}")
            finally:
                sock.close()
                
        except KeyboardInterrupt:
            print("\n\033[91m🛑 [CLI] Conexão desengatada. Aeon continua observando e processando a realidade do usuário em background.\033[0m")
            break

    # Restaura o terminal para o comportamento normal antes de sair
    sys.stdout.write("\033[r\033[2J\033[H")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
