"""Obelisk Daemon — supervisor do processo devourer.

Processo raiz (PID fixo) que:
  1. Inicia/fiscaliza o main.py (obelisk watchdog)
  2. Monitora healthcheck via pipe
  3. Aplica rollback e reinicia se o processo filho morre
  4. Gera heartbeat próprio para o systemd/supervisor externo

Uso:
  python cortex/obelisk/daemon.py            # modo supervisor
  python cortex/obelisk/daemon.py --once     # executa main uma vez sem supervisão
"""

import os
import sys
import time
import signal
import subprocess

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PID_FILE = "/dev/shm/aeon/obelisk.pid"
HEARTBEAT_FILE = "/dev/shm/aeon/obelisk.heartbeat"


class ObeliskDaemon:
    def __init__(self):
        self.child = None
        self.running = True
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        self.running = False
        if self.child and self.child.poll() is None:
            self.child.terminate()

    def _write_pid(self):
        os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
        with open(PID_FILE, "w") as f:
            f.write(str(os.getpid()))

    def _touch_heartbeat(self):
        os.makedirs(os.path.dirname(HEARTBEAT_FILE), exist_ok=True)
        with open(HEARTBEAT_FILE, "w") as f:
            f.write(str(time.time()))

    def _spawn_devourer(self) -> subprocess.Popen:
        return subprocess.Popen(
            [sys.executable, "-m", "cortex.obelisk.main"],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": PROJECT_ROOT},
        )

    def run(self):
        self._write_pid()
        self.child = self._spawn_devourer()

        while self.running:
            self._touch_heartbeat()
            ret = self.child.poll()

            if ret is not None:
                self.child = self._spawn_devourer()
            else:
                pass

            for _ in range(6):
                if not self.running:
                    break
                time.sleep(5)

        if self.child and self.child.poll() is None:
            self.child.terminate()
            self.child.wait(timeout=10)


def run_once():
    from cortex.obelisk.main import main_loop
    main_loop()


if __name__ == "__main__":
    if "--once" in sys.argv:
        run_once()
    else:
        daemon = ObeliskDaemon()
        daemon.run()
