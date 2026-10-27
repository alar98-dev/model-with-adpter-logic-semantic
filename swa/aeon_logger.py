import threading
import os
import sys
from swa.paths import project_path
try:
    from systemd import journal
    _HAS_JOURNAL = True
except Exception:
    _HAS_JOURNAL = False

_AEON_LOG_PATH = str(project_path("swa_Aeon.log"))
_lock = threading.Lock()


def append_aeon(text: str) -> None:
    """Append text to Aeon log — SEM fsync() para não bloquear o stream extensional.
    
    CANAL EXTENSIONAL (VSCode) tem prioridade máxima.
    O log é best-effort: se falhar, não interrompe a geração.
    """
    try:
        with _lock:
            d = os.path.dirname(_AEON_LOG_PATH)
            if d and not os.path.exists(d):
                try:
                    os.makedirs(d, exist_ok=True)
                except Exception:
                    pass
            with open(_AEON_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(text)
                # ⚡ SEM fsync() — não bloqueia o canal extensional por I/O de disco
    except Exception:
        # Best-effort: silencioso para não atrasar o stream
        pass


def append_aeon_batch(text: str) -> None:
    """Escrita em lote ao final de um pensamento completo — pode usar fsync aqui.
    
    Chamado apenas 1x por geração (no end_thought), não por token.
    """
    # Envia o pacote UDP para o terminal Aeon CLI
    try:
        import socket
        import json
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        payload = json.dumps({"type": "thought_stream", "content": text}).encode('utf-8')
        sock.sendto(payload, ("127.0.0.1", 20005))
    except Exception:
        pass

    try:
        with _lock:
            d = os.path.dirname(_AEON_LOG_PATH)
            if d and not os.path.exists(d):
                try:
                    os.makedirs(d, exist_ok=True)
                except Exception:
                    pass
            with open(_AEON_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                try:
                    os.fsync(f.fileno())  # 1 fsync por pensamento completo (ok)
                except Exception:
                    pass
    except Exception:
        try:
            sys.stdout.write(text)
            sys.stdout.flush()
        except Exception:
            pass


def log_infra(message: str) -> None:
    """Log infrastructure/telemetry messages to system journal (or stdout).

    This explicitly avoids writing to the Aeon voice log.
    """
    try:
        if _HAS_JOURNAL:
            try:
                # Send as informational message to the system journal
                journal.send(message, SYSLOG_IDENTIFIER="swa-auto", PRIORITY=6)
                return
            except Exception:
                pass
        # Fallback: print to stdout which systemd/journal will capture when run as a service
        sys.stdout.write(message if message.endswith("\n") else message + "\n")
        sys.stdout.flush()
    except Exception:
        pass
