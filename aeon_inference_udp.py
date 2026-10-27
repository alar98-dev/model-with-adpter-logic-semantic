#!/usr/bin/env python3
# ==============================================================================
# 🔮 AEON ENTROPY NUCLEUS — NÚCLEO ÚNICO DE ENTROPIA DO MICRO-VERSO
#
# UM só Gemma carregado em memória para TODO o micro-verso (córtex de
# pensamento, fusão, temporal, identidade — nenhum outro módulo carrega modelo).
# Comunicação leve via UDP (127.0.0.1:20001), sem custo de HTTP/JSON/SSE.
#
# Protocolo:
#   req:  {"op":"generate","id":N,"q":"prompt","max_new_tokens":120,"temperature":0.9,"top_p":0.92}
#   resp: {"id":N,"r":"texto","cost":secs,"source":"entropy_nucleus","model":...}
#   busy: {"id":N,"busy":1}
#
# Robustez: cada pedido isolado em try/except (nunca mata o núcleo), fila de
# trabalho com limite, reload automático do modelo se a carga falhar.
#
#   Run:  /home/alifer/swa-aeon/venv/bin/python3 aeon_inference_udp.py
#   Modelo (env AEON_UDP_MODEL): default gemma-3-270m-it (leve, cabe na RAM).
#   Para o 1B use:  AEON_UDP_MODEL=/home/alifer/swa-aeon/models/gemma-3-1b-it
#   Log:  /home/alifer/swa-aeon/entropy_nucleus.log
# ==============================================================================

import os
import sys
import io
import json
import time
import socket
import threading
from collections import deque

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

sys.path.insert(0, "/home/alifer/swa-aeon")

DEFAULT_MODEL = "/home/alifer/swa-aeon/models/gemma-3-270m-it"
HOST, PORT = os.environ.get("AEON_UDP_HOST", "127.0.0.1"), 20001
MAX_QUEUE = 8
LOG_FILE = "/home/alifer/swa-aeon/entropy_nucleus.log"


def log(line):
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {line}\n")
    except Exception:
        pass
    try:
        print(line, flush=True)
    except Exception:
        pass


class EntropyNucleus:
    def __init__(self):
        self.model_dir = os.environ.get("AEON_UDP_MODEL", DEFAULT_MODEL)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # lock de singleton UDP (porta = 1 processo por núcleo)
        try:
            self.sock.bind((HOST, PORT))
        except Exception:
            sys.exit(0)
        self.sock.settimeout(0.5)

        self.queue = deque(maxlen=MAX_QUEUE)
        self.busy = False
        self.lock = threading.Lock()
        self.core = None

    # ---------------- modelo ----------------
    def ensure_model(self):
        """Carrega o Gemma UMA vez (motor de entropia puro: tokenizer + model).
        Se falhar, tenta de novo (núcleo nunca morre). Sem órgãos extras por-token:
        o micro-verso precisa de latência de segundos, não minutos."""
        if self.core is not None:
            return True
        try:
            log(f"🔮 [NÚCLEO] Carregando Gemma {self.model_dir} ...")
            t0 = time.time()
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            device = os.environ.get("AEON_NUCLEUS_DEVICE", "").strip()
            if not device:
                # XPU (oneAPI) tem se mostrado instável neste fluxo (“[RuntimeError] UR error”);
                # CPU float32 é o caminho confiável e rápido o suficiente para o 270m.
                device = "cpu"
            dtype = torch.float32
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_dir,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                .to(device)
                .eval()
            )
            self.device = device
            self.core = object()  # marcador: núcleo pronto
            model_id = os.path.basename(self.model_dir.rstrip("/"))
            log(f"🔮 [NÚCLEO] Modelo pronto em {time.time()-t0:.1f}s :: {model_id} ({device})")
            return True
        except Exception as exc:
            log(f"⚠️ [NÚCLEO] Falha ao carregar modelo: {exc} — nova tentativa em 30s")
            self.core = None
            return False

    # ---------------- fila / worker ----------------
    def run_worker(self):
        while True:
            item = None
            with self.lock:
                if self.queue:
                    item = self.queue.popleft()
            if item is None:
                time.sleep(0.02)
                continue
            self._handle(*item)

    def _handle(self, data, addr):
        try:
            req = json.loads(data.decode("utf-8", "replace"))
        except Exception:
            return
        rid = req.get("id", 0)
        q = req.get("q", "")
        if not q.strip():
            return
        if not self.ensure_model():
            self._reply(addr, {"id": rid, "r": None, "error": "núcleo sem modelo"})
            return

        with self.lock:
            self.busy = True
        timeout = float(req.get("timeout", 90))
        try:
            result = {}
            t0 = time.time()
            wt = threading.Thread(target=self._gen_worker, args=(req, result), daemon=True)
            wt.start()
            import queue as _q
            try:
                ev = result
                # espera até o worker publicar o resultado (com tolerância)
                for _ in range(int(timeout * 10)):
                    if ev.get("done", False):
                        break
                    time.sleep(0.1)
                if not ev.get("done", False):
                    raise TimeoutError(f"geração excedeu {timeout}s")
            except TimeoutError as te:
                log(f"⏰ [NÚCLEO] TIMEOUT no pedido {rid} ({timeout}s) — descartando modelo p/ sempre voltar limpo")
                self._reply(addr, {"id": rid, "r": None, "error": f"timeout {timeout}s"})
                self.core = None
                self.model = None
                import gc
                gc.collect()
                return
            if ev.get("error"):
                self._reply(addr, {"id": rid, "r": None, "error": ev["error"]})
                return
            cost = time.time() - t0
            model_id = os.path.basename(self.model_dir.rstrip("/"))
            self._reply(addr, {"id": rid, "r": ev["r"], "cost": round(cost, 1),
                               "source": "entropy_nucleus", "model": f"{model_id}@entropy-nucleus"})
            log(f"🔮 [NÚCLEO] gerado {len(ev['r'])} chars em {cost:.1f}s (id={rid})")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            log(f"⚠️ [NÚCLEO] erro no pedido {rid}: [{type(exc).__name__}] {exc!r}")
            self._reply(addr, {"id": rid, "r": None, "error": f"[{type(exc).__name__}] {exc}"})
        finally:
            with self.lock:
                self.busy = False

    def _gen_worker(self, req, result):
        """Geração isolada em thread própria — watchdog capaz de abortá-la."""
        try:
            import torch
            rid = req.get("id", 0)
            q = req.get("q", "")
            if not self.ensure_model():
                result["done"] = True
                result["error"] = "núcleo sem modelo"
                return
            messages = [{"role": "user", "content": q}]
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
            ids = prompt["input_ids"] if hasattr(prompt, "keys") else prompt
            ids = ids.to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    ids,
                    max_new_tokens=int(req.get("max_new_tokens", 120)),
                    temperature=float(req.get("temperature", 0.9)),
                    top_p=float(req.get("top_p", 0.92)),
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = out[0, ids.shape[-1]:]
            result["r"] = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            result["done"] = True
        except Exception as exc:
            result["error"] = f"[{type(exc).__name__}] {exc}"
            result["done"] = True

    def _reply(self, addr, payload):
        try:
            self.sock.sendto(json.dumps(payload).encode("utf-8"), addr)
        except Exception:
            pass

    # ---------------- loop ----------------
    def run(self):
        threading.Thread(target=self.run_worker, daemon=True).start()
        # pré-carrega (sem bloquear a primeira resposta)
        threading.Thread(target=self.ensure_model, daemon=True).start()
        log(f"🔮 [NÚCLEO] Entropy Nucleus ativo em {HOST}:{PORT} (modelo: {self.model_dir})")
        while True:
            try:
                data, addr = self.sock.recvfrom(65536)
            except socket.timeout:
                continue
            except KeyboardInterrupt:
                break
            busy = False
            with self.lock:
                busy = self.busy or len(self.queue) >= MAX_QUEUE
            if busy:
                try:
                    rid = json.loads(data).get("id", 0)
                except Exception:
                    rid = 0
                self._reply(addr, {"id": rid, "busy": 1})
                continue
            self.queue.append((data, addr))
        log("👋 Núcleo de entropia encerrado.")


if __name__ == "__main__":
    EntropyNucleus().run()