#!/usr/bin/env python3
"""
AeonC — Núcleo Purificado.

A responsabilidade nativa de um LLM: gerar texto de forma interativa.
Sem Geosense, sem BlakeGravity, sem Redis, sem órgãos. Só modelo + tokenizador.

Usage:
    python aeonc.py                      # CLI interativa (streaming)
    python aeonc.py --model models/gemma-3-1b-it
    python aeonc.py --api --port 8899    # API HTTP/SSE
    python aeonc.py --once "olá, quem é você?"
"""

import os
import sys
import argparse
import json
import re
import time
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Optional, Generator

os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))

import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(1)
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

try:
    from swa.ooda_flow import OODAFlow
    _HAS_OODA_FLOW = True
except Exception:
    OODAFlow = None
    _HAS_OODA_FLOW = False

try:
    from swa.kbase import KBaseEngine, EmbedClient, LAYERS as KBASE_LAYERS
    _HAS_KBASE = True
except Exception:
    KBaseEngine = None
    EmbedClient = None
    KBASE_LAYERS = []
    _HAS_KBASE = False

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gemma-3-1b-it")

# ---- interfaces do organismo sistêmico (compartilhadas via /dev/shm) ----
SHM_AEON_DIR = "/dev/shm/aeon"
CORE_HEARTBEAT = os.path.join(SHM_AEON_DIR, "core.heartbeat")
CORE_PID_FILE = os.path.join(SHM_AEON_DIR, "core.pid")
CORE_STATE = os.path.join(SHM_AEON_DIR, "core.state.json")
CORE_THOUGHTS = os.path.join(SHM_AEON_DIR, "core.thoughts.jsonl")
CORE_TENSOR_BIN = os.path.join(SHM_AEON_DIR, "core.tensor.bin")
CORE_TENSOR_JSON = os.path.join(SHM_AEON_DIR, "core.tensor.json")
CORE_INTENT = os.path.join(SHM_AEON_DIR, "core.intent.jsonl")
SUBCORTEX_IMAGEM = "/tmp/swa_imagem_mental.txt"
MANIFOLD_DIR = "/dev/shm/swa_vector_manifold"
KERNEL_KNOWLEDGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "swa_kernel_knowledge.json")


@dataclass
class GenerationProfile:
    """Especialização da geração — separa responsabilidades por objetivo.

    preciso:      resposta exata, curta e factual (baixa temperatura).
    equilibrado:  padrão — equilíbrio entre precisão e fluência.
    explorativo:  mais texto gerado — rico, detalhado e diverso (alta temperatura).
    duo:          ambos — primeiro responde preciso, depois expande explorativo.
    """

    name: str
    description: str
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 512
    do_sample: bool = True
    repetition_penalty: float = 1.0
    instruction: str = ""


PROFILES: dict[str, GenerationProfile] = {
    "preciso": GenerationProfile(
        name="preciso",
        description="resposta exata, curta e objetiva",
        temperature=0.1,
        top_p=0.6,
        max_new_tokens=256,
        repetition_penalty=1.1,
        instruction=(
            "Responda com precisão e objetividade, sem rodeios. "
            "Seja conciso e factual; cite o essencial, nada além."
        ),
    ),
    "equilibrado": GenerationProfile(
        name="equilibrado",
        description="padrão — precisão e fluência equilibradas",
    ),
    "explorativo": GenerationProfile(
        name="explorativo",
        description="mais texto — rico, detalhado e diverso",
        temperature=0.95,
        top_p=1.0,
        max_new_tokens=1024,
        repetition_penalty=1.15,
        instruction=(
            "Desenvolva amplamente: explore nuances, exemplos e possibilidades. "
            "Gere texto rico e detalhado, sem se prender ao mínimo."
        ),
    ),
    "duo": GenerationProfile(
        name="duo",
        description="ambos — primeiro exato, depois expansão",
        max_new_tokens=768,
        instruction=(
            "Estruture em duas partes: 1) Resposta precisa e objetiva (resumo curto). "
            "2) Expansão: explore nuances, exemplos e possibilidades em detalhe."
        ),
    ),
}


import json
import os
try:
    _persona_path = os.path.join(os.path.dirname(__file__), "prompts", "swa_master_persona.json")
    with open(_persona_path, "r", encoding="utf-8") as _f:
        PERSONA = json.load(_f).get("normal", "")
except Exception:
    PERSONA = "SWA CORE V4 >> Bem-vindo ao sistema. Você é AEON."



def list_profiles() -> str:
    return ", ".join(f"{n} ({p.description})" for n, p in PROFILES.items())


class AeonOrganism:
    """Nervos do organismo: sangue (liveness), memória (manifold), expressão (pensamentos).

    O núcleo deixa de ser um órgão isolado e passa a compartilhar o mesmo espaço
    físico do resto do sistema (obelisk, observer, manifold) em /dev/shm.
    """

    def __init__(self):
        self._heartbeat_lock = threading.Lock()

    # ---- sangue (liveness) ----
    def touch_heartbeat(self, state: str = "alive") -> None:
        """Grava heartbeat + pid do core para o verifier/observer enxergarem o núcleo vivo."""
        try:
            with self._heartbeat_lock:
                os.makedirs(SHM_AEON_DIR, exist_ok=True)
                payload = {
                    "ts": time.time(),
                    "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "pid": os.getpid(),
                    "state": state,
                    "metrics": getattr(self, "last_metrics", {}),
                    "model": getattr(self, "model_id", ""),
                }
                tmp = CORE_HEARTBEAT + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
                os.replace(tmp, CORE_HEARTBEAT)
                with open(CORE_PID_FILE, "w", encoding="utf-8") as f:
                    f.write(str(os.getpid()))
        except Exception as e:
            print(f"[heartbeat] erro ao gravar: {e}", file=sys.stderr)

    def start_heartbeat_loop(self, interval: float = 30.0) -> None:
        """Batimento contínuo mesmo sem tráfego — o organismo sabe que o núcleo respira."""
        def _loop():
            while True:
                self.touch_heartbeat()
                time.sleep(interval)
        threading.Thread(target=_loop, daemon=True).start()

    # ---- memória (manifold + conhecimento do kernel) ----
    def read_manifold(self) -> dict:
        info = {"nodes": 0, "zones": {}, "embedding": "?", "paths": [], "concepts": []}
        try:
            with open(os.path.join(MANIFOLD_DIR, "manifest.json"), encoding="utf-8") as f:
                m = json.load(f)
            info["nodes"] = m.get("node_count", 0)
            info["zones"] = m.get("zones", {})
            info["embedding"] = m.get("embedding_source", "?")
        except Exception:
            pass
        try:
            with open(os.path.join(MANIFOLD_DIR, "reality_tree.jsonl"), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        n = json.loads(line)
                    except Exception:
                        continue
                    info["paths"].append(
                        (n.get("physical_path", ""), n.get("zone", ""), n.get("concept", ""))
                    )
                    if len(info["paths"]) >= 400:
                        break
        except Exception:
            pass
        try:
            with open(KERNEL_KNOWLEDGE, encoding="utf-8") as f:
                info["concepts"] = list(json.load(f).keys())
        except Exception:
            pass
        return info

    def read_thoughts(self, n: int = 3) -> list[str]:
        try:
            if not os.path.isfile(CORE_THOUGHTS):
                return []
            with open(CORE_THOUGHTS, encoding="utf-8") as f:
                lines = [ln for ln in f if ln.strip()][-n:]
            return [json.loads(ln).get("content", "")[:180] for ln in lines]
        except Exception:
            return []

    def count_thoughts(self) -> int:
        try:
            if not os.path.isfile(CORE_THOUGHTS):
                return 0
            with open(CORE_THOUGHTS, encoding="utf-8") as f:
                return sum(1 for ln in f if ln.strip())
        except Exception:
            return 0

    @staticmethod
    def _kbase_layers_for(query: str) -> list[int]:
        """Camadas por tipo de query: código/sigla → micro (100, 250); prosa → todas."""
        q = query or ""
        code_like = bool(re.search(r"[A-Z_][A-Z_0-9]{2,}|::|\.py\b|\bdef\b|\bclass\b|=>|<=>", q))
        if code_like and not re.search(r"\b(como|qual|o que|por que|onde|quando)\b", q.lower()):
            return [100, 250]
        return [100, 250, 500, 1000]

    def kbase_block(self, user_msg: str, top_k: int = 4) -> str:
        """Recupera conhecimento do K-Base (busca híbrida denso+esparso, RRF, Great Filter)."""
        if self.kbase is None or not self.kbase_enabled:
            return ""
        try:
            hits = self.kbase.retrieve(user_msg, top_k=top_k, layers=self._kbase_layers_for(user_msg))
        except Exception:
            return ""
        if not hits:
            return ""
        st = self.kbase.last_status
        lines = ["[kbase]"]
        lines.append(f"fontes: {st.get('hits', 0)} | {st.get('elapsed_ms', 0):.0f}ms | camadas {st.get('layers', KBASE_LAYERS)}")
        for h in hits[:top_k]:
            snippet = " ".join((h.get("text") or "").split())[:160]
            lines.append(f"- [{h.get('sim', 0):.2f}/{h.get('collection')}/{h.get('layer')}] {h.get('source_path')} :: {snippet}")
        lines.append("[/kbase]")
        return "\n".join(lines)

    def organism_context(self, user_msg: str) -> str:
        """Contexto do organismo injetado na geração (memória recuperada por afinidade)."""
        info = self.read_manifold()
        if not info["nodes"] and not info["concepts"]:
            return ""

        query = set(t for t in re.findall(r"[a-z0-9]{4,}", user_msg.lower()))
        scored = []
        for path, zone, concept in info["paths"]:
            hit = len(set(t for t in re.findall(r"[a-z0-9]{4,}", path.lower())) & query)
            if hit:
                scored.append((hit, path, zone, concept))
        scored.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
        refs = scored[:5]
        affinity = [c for c in info["concepts"] if any(tok in c.lower() for tok in query)]

        lines = ["[organismo]"]
        lines.append(f"memória: {info['nodes']} nós | fonte: {info['embedding']}")
        zones = ", ".join(f"{z}={n}" for z, n in sorted(info["zones"].items()))
        if zones:
            lines.append(f"zonas: {zones}")
        if info["concepts"]:
            lines.append("conhecimento: " + ", ".join(info["concepts"]))
        if refs:
            lines.append("referências: " + " ; ".join(f"{c}@{p}" for _, p, _z, c in refs))
        if affinity:
            lines.append("afinidade: " + ", ".join(affinity))
        for th in self.read_thoughts(3):
            lines.append(f"pensamento anterior: {th}")
        sub = self.read_subcortex_perception()
        if sub:
            lines.append("percepção (subcórtex): " + sub)
        kbase = self.kbase_block(user_msg)
        if kbase:
            lines.append(kbase)
        if getattr(self, "bladetools_map", None) and getattr(self, "history_vectorizer", None):
            try:
                import torch
                import torch.nn.functional as F
                t_ids = self.tokenizer(user_msg, return_tensors="pt")["input_ids"].to(self.device)
                with torch.no_grad():
                    t_out = self.history_vectorizer(t_ids, output_hidden_states=True)
                    u_vec = t_out.hidden_states[-1].mean(dim=1).squeeze()
                    if u_vec.shape[-1] == 640: u_vec = u_vec.repeat(4)
                    u_vec = u_vec.to(torch.float32).cpu()
                
                for tool_name, data in self.bladetools_map.items():
                    tool_vec = torch.tensor(data["tensor_2560"], dtype=torch.float32)
                    cos_sim = F.cosine_similarity(u_vec.unsqueeze(0), tool_vec.unsqueeze(0)).item()
                    if cos_sim > 0.45: # Limite de sensibilidade proprioceptiva
                        lines.append(f"percepção motora (subcórtex): {data['feeling']}")
            except Exception as e:
                pass
                
        lines.append("[/organismo]")
        return "\n".join(lines)

    # ---- expressão (saída → memória do organismo) ----
    def express(self, user_msg: str, reply: str) -> None:
        """Publica o pensamento no organismo: journal persistente + nó na árvore da realidade."""
        try:
            ts = time.time()
            path = f"aeon://thought/{int(ts)}"
            digest = hashlib.blake2b(f"{ts}:{reply[:200]}".encode("utf-8"), digest_size=16).hexdigest()
            node = {
                "physical_path": path,
                "interdim_id": digest,
                "pg_address": f"mmap://{digest}#{path}",
                "parent_path": "aeon://thoughts",
                "zone": "aeon",
                "depth": 3,
                "concept": "THOUGHT",
                "mtime": ts,
                "size_bytes": len(reply),
            }
            os.makedirs(SHM_AEON_DIR, exist_ok=True)
            record = {
                **node,
                "question": user_msg[:200],
                "content": reply[:500],
                "lesson": self._lesson_from(user_msg, reply),
            }
            with open(CORE_THOUGHTS, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._cap_thoughts(400)
            tree = os.path.join(MANIFOLD_DIR, "reality_tree.jsonl")
            if os.path.isfile(tree):
                with open(tree, "a", encoding="utf-8") as f:
                    f.write(json.dumps(node, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _lesson_from(user_msg: str, reply: str) -> str:
        """Auto-reflexão por interação: a última frase da resposta vira a 'lição' do organismo."""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", reply) if s.strip()]
        return sentences[-1][:200] if sentences else reply[:200]

    def express_intent(self, text: str) -> None:
        """Canal córtex→subcórtex: publica o que o núcleo processou/necessita (core.intent.jsonl)."""
        try:
            os.makedirs(SHM_AEON_DIR, exist_ok=True)
            with open(CORE_INTENT, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "intent": text,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        except Exception as e:
            print(f"[intent] erro ao gravar: {e}", file=sys.stderr)

    def read_subcortex_perception(self, limit: int = 300) -> str:
        """Canal subcórtex→córtex: lê a 'imagem mental' que o subcórtex gerou dos sensores."""
        try:
            with open(SUBCORTEX_IMAGEM, encoding="utf-8") as f:
                text = f.read().strip()
            if not text:
                return ""
            text = " ".join(text.split())
            return text[:limit] + ("…" if len(text) > limit else "")
        except Exception:
            return ""

    def _cap_thoughts(self, max_lines: int) -> None:
        try:
            with open(CORE_THOUGHTS, encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) > max_lines:
                with open(CORE_THOUGHTS, "w", encoding="utf-8") as f:
                    f.writelines(lines[-max_lines:])
        except Exception:
            pass


def resolve_device(requested: str = "auto") -> str:
    req = (requested or "auto").strip().lower()
    if req in ("cuda", "gpu") and torch.cuda.is_available():
        return "cuda"
    if req in ("xpu", "intel", "igpu") and hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if req == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def pick_dtype(device: str, requested: Optional[str] = None) -> torch.dtype:
    req = (requested or os.environ.get("AEONC_DTYPE", "")).strip().lower()
    if req in ("float32", "fp32"):
        return torch.float32
    if req in ("float16", "fp16"):
        return torch.float16
    if req in ("bfloat16", "bf16"):
        return torch.bfloat16
    if device == "cpu":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16


class AeonCore(AeonOrganism):
    """Carga nativa do modelo + geração interativa de texto."""

    def __init__(
        self,
        model_dir: str = DEFAULT_MODEL,
        device: str = "auto",
        dtype: Optional[str] = None,
        compile: bool = False,
        system_prompt: str = "",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        tensor_sharing: bool = False,
        default_profile: Optional[str] = None,
        ooda: bool = True,
    ):
        AeonOrganism.__init__(self)
        self.device = resolve_device(device)
        self.dtype = pick_dtype(self.device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_dir,
                dtype=self.dtype,
                low_cpu_mem_usage=True,
            )
            .to(self.device)
            .eval()
        )

        self.tensor_sharing = tensor_sharing
        self._last_hidden: Optional[object] = None
        self._tensor_source: str = ""
        if self.tensor_sharing or (ooda and _HAS_OODA_FLOW):
            self._install_layer_hook()
        self._install_embed_hook()

        self.compile = compile
        if self.compile:
            self._compile_model()

        self.system_prompt = system_prompt or PERSONA
        self.history: list[dict] = []
        self.last_metrics: dict = {}
        self.model_id = os.path.basename(model_dir.rstrip("/"))
        self.org_enabled = True
        self.kbase_enabled = True
        self.kbase = None
        if _HAS_KBASE:
            try:
                embed_client = EmbedClient(timeout=30)
                self.kbase = KBaseEngine(embed_client=embed_client, enabled=True)
                if not embed_client.ready():
                    self.kbase = None
                elif hasattr(self.kbase, "_refresh_stats"):
                    self.kbase._refresh_stats()
            except Exception:
                self.kbase = None
        self.ooda = None
        if _HAS_OODA_FLOW and ooda:
            self.ooda = OODAFlow(tensor_of=lambda: self._last_hidden, profile_of=lambda: self.profile.name, enabled=True)
        if default_profile is None:
            default_profile = self._load_state_profile()
        self.apply_profile(default_profile)
        if max_new_tokens != 512:
            self.max_new_tokens = max_new_tokens
        if temperature != 0.7:
            self.temperature = temperature
        if top_p != 0.95:
            self.top_p = top_p
            
        # [CÓRTEX MOTOR E PROPRIOCEPÇÃO]
        try:
            import joblib
            self.action_svm = joblib.load("/home/alifer/swa-aeon/action_svm_model.pkl")
            self.bladetools_map = joblib.load("/home/alifer/swa-aeon/bladetools_sensory_map.pkl")
        except:
            self.action_svm = None
            self.bladetools_map = None

    def _ooda_tensor(self):
        return getattr(self, "_last_hidden", None)

    def apply_profile(self, name: str) -> bool:
        """Muda a especialização da geração (preciso | equilibrado | explorativo | duo)."""
        name = (name or "").strip().lower()
        if name not in PROFILES:
            return False
        self.profile = PROFILES[name]
        self.max_new_tokens = self.profile.max_new_tokens
        self.temperature = self.profile.temperature
        self.top_p = self.profile.top_p
        self._save_state()
        if hasattr(self, "_warmup_atemporal_kv"): self._warmup_atemporal_kv()
        return True

    def _load_state_profile(self) -> str:
        """Restaura o perfil persistido (core.state.json) — o organismo não esquece."""
        try:
            with open(CORE_STATE, encoding="utf-8") as f:
                data = json.load(f)
            name = (data.get("profile") or "").strip().lower()
            if name in PROFILES:
                print(f"Perfil persistido restaurado: {name}")
                return name
        except Exception:
            pass
        return "equilibrado"

    
    def _warmup_atemporal_kv(self) -> None:
        try:
            if not hasattr(self, 'model') or self.model is None:
                return
            p = getattr(self, 'profile', None)
            instr = p.instruction if p else ''
            sys_text = self.system_prompt + "\n\n" + instr
            msgs = [{'role': 'user', 'content': sys_text + '\n[Aviso Interno: Aja conforme essas diretrizes ao responder o proximo input.]'}]
            prompt = self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)
            with torch.no_grad():
                out = self.model(ids, use_cache=True)
            self._static_past_key_values = out.past_key_values
            print(f'_kv-cache] Atemporal blindado em RAM ({ids.shape[1]} tokens)')
        except Exception as e:
            print(f'[kv-cache] aviso no pre-computar: {e}')
            self._static_past_key_values = None

    def _save_state(self) -> None:
        try:
            os.makedirs(SHM_AEON_DIR, exist_ok=True)
            with open(CORE_STATE, "w", encoding="utf-8") as f:
                json.dump(
                    {"profile": self.profile.name, "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception as e:
            print(f"[state] erro ao gravar: {e}", file=sys.stderr)

    def _install_embed_hook(self) -> None:
        try:
            self.model.model.embed_tokens.register_forward_hook(self._embed_hook)
        except Exception:
            pass

    def _embed_hook(self, module, args, output):
        if hasattr(self, '_history_injection_vector') and self._history_injection_vector is not None:
            try:
                output[:, 0, :] = self._history_injection_vector.squeeze(1)
            except Exception:
                pass
            self._history_injection_vector = None
            
        # [SEMANTIC WARP DRIVE] Injeção de Ruído para Salto Dimensional
        import torch
        if getattr(self, 'warp_drive_active', False):
            # Injeta perturbação gaussiana violenta (Buraco Branco) no espaço latente
            noise = torch.randn_like(output) * 1.5
            output = output + noise
            # Reseta o Warp após injetar
            self.warp_drive_active = False
            
        return output

    def _install_layer_hook(self) -> None:
        """Lê o estado oculto da última camada do decoder a cada forward pass (indução pelas camadas)."""
        try:
            layers = self.model.model.layers
            last = layers[-1]
            last.register_forward_hook(self._capture_hidden)
            print(f"Tensor hook instalado na camada {len(layers)} (hidden={self.model.config.hidden_size})")
        except Exception as e:
            print(f"[tensor] hook não instalado: {e}", file=sys.stderr)
            self.tensor_sharing = False

    def _capture_hidden(self, module, args, output) -> None:
        try:
            hs = output[0] if isinstance(output, tuple) else output
            self._last_hidden = hs[0, -1].detach().to(torch.float32).cpu()
            self._tensor_source = f"layer{len(self.model.model.layers)}"
        except Exception:
            pass

    def _embedding_of(self, text: str) -> Optional[object]:
        """Fallback: vetor médio da camada de entrada (embed_tokens) sobre a resposta."""
        try:
            ids = self.tokenizer.encode(text, add_special_tokens=False)[-128:]
            if not ids:
                return None
            tok = torch.tensor([ids], device=self.device)
            with torch.no_grad():
                emb = self.model.model.embed_tokens(tok)
            return emb[0].mean(dim=0).detach().float().cpu()
        except Exception:
            return None

    def _blake3_bytes(self, data: bytes) -> str:
        """Assinatura BLAKE3 da injeção tensorial — integridade verificável por quem consome."""
        try:
            import blake3
            return blake3.blake3(data).hexdigest()
        except Exception:
            return ""

    def _share_tensor(self, reply_text: str) -> None:
        """Publica o tensor do pensamento em /dev/shm para o resto do organismo consumir.

        A injeção é assinada (blake3) e auto-verificada: o consumidor (verifier)
        recomputa o hash do .bin e compara com a assinatura gravada no .json.
        """
        try:
            import numpy as np
            t = self._last_hidden
            source = self._tensor_source
            if t is None:
                t = self._embedding_of(reply_text)
                source = "embed_tokens(avg)"
            if t is None:
                return
            os.makedirs(SHM_AEON_DIR, exist_ok=True)
            v = t.numpy().astype("float32")
            v.tofile(CORE_TENSOR_BIN)
            raw = v.tobytes()
            b3 = self._blake3_bytes(raw)
            sig = f"blake3:{b3}" if b3 else f"sha256:{hashlib.sha256(raw).hexdigest()}"
            with open(CORE_TENSOR_JSON, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ts": time.time(),
                        "iso": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "dim": int(v.shape[0]),
                        "source": source,
                        "reply_tokens": len(
                            self.tokenizer.encode(reply_text, add_special_tokens=False)
                        ),
                        "norm": round(float(np.linalg.norm(v)), 4),
                        "sha256": hashlib.sha256(raw).hexdigest()[:16],
                        "signature": sig,
                    },
                    f,
                    ensure_ascii=False,
                )
            if b3:
                actual = self._blake3_bytes(raw)
                if actual == b3:
                    print(f"[tensor] injeção assinada e verificada (blake3:{b3[:16]}… {int(v.shape[0])}d)")
                else:
                    print(f"[tensor] auto-verificação falhou — assinatura divergente", file=sys.stderr)
        except Exception as e:
            print(f"[tensor] não compartilhado: {e}", file=sys.stderr)

    def _compile_model(self) -> None:
        """torch.compile + warmup para evitar travamento na primeira request."""
        import time as _t

        try:
            print("Compilando modelo (torch.compile)...")
            t0 = _t.perf_counter()
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self.model.eval()
            tok_ids = torch.tensor(
                [[self.tokenizer.bos_token_id or self.tokenizer.eos_token_id]],
                device=self.device,
            )
            with torch.no_grad():
                self.model.generate(
                    input_ids=tok_ids,
                    attention_mask=tok_ids.new_ones(tok_ids.shape),
                    max_new_tokens=2,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            print(f"Compilação concluída em {_t.perf_counter() - t0:.1f}s")
        except Exception as e:
            print(f"torch.compile falhou, seguindo sem compilação: {e}")
            self.compile = False

    @staticmethod
    def _merge_system_messages(messages: list[dict]) -> list[dict]:
        """O Template do Gemma não aceita role 'system'. Funde tudo no primeiro 'user'."""
        sys_blocks = []
        user_blocks = []
        for m in messages:
            if m.get("role") == "system":
                sys_blocks.append(m.get("content", ""))
            else:
                user_blocks.append(dict(m))
                
        if sys_blocks and user_blocks and user_blocks[0]["role"] == "user":
            combined_sys = "\n\n".join(sys_blocks)
            user_blocks[0]["content"] = combined_sys + "\n\n[Aviso Interno: Aja conforme essas diretrizes ao responder o proximo input.]\n\n" + user_blocks[0]["content"]
        
        return user_blocks

    def build_messages(self, user_msg: str, profile: Optional[GenerationProfile] = None) -> list[dict]:
        p = profile or self.profile
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if p.instruction:
            messages.append({"role": "system", "content": p.instruction})
        if self.org_enabled:
            ctx = self.organism_context(user_msg)
            if ctx:
                messages.append({"role": "system", "content": ctx})
        # O histórico não entra mais como texto burro. Ele será processado pelo 270M e injetado via Tensor Geometrico (Soft-Prompt).
        messages.append({"role": "user", "content": user_msg})
        return self._merge_system_messages(messages)

    @staticmethod
    def _last_user(messages: list[dict]) -> str:
        for m in reversed(messages):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""

    def _get_history_vector(self) -> object:
        if not hasattr(self, "history_vectorizer"):
            import torch
            from transformers import AutoModel
            print(f"\n\033[96m🌌 [SWA-AEON] Córtex Histórico: Despertando Gemma 270M (Subconsciente) para Gravidade Semântica...\033[0m")
            try:
                self.history_vectorizer = AutoModel.from_pretrained("/home/alifer/swa-aeon/models/gemma-3-270m").to(self.device).eval()
                # Matriz de projeção física (2560 -> 1152) para acoplar no Córtex 1B
                self.history_proj = torch.nn.Linear(self.history_vectorizer.config.hidden_size, self.model.config.hidden_size, bias=False).to(self.device).to(self.model.dtype)
                print(f"\033[92m✅ Ponte Tensorial Ativa: Histórico Geométrico engatilhado!\033[0m")
            except Exception as e:
                print(f"\033[91m⚠️ Falha ao carregar 270M: {e}\033[0m")
                self.history_vectorizer = None

        if not self.history_vectorizer or not self.history:
            return None

        import torch
        hist_text = "\n".join([m.get("content", "") for m in self.history if m.get("role") != "system"])
        if not hist_text.strip(): return None
        
        ids = self.tokenizer(hist_text, return_tensors="pt")["input_ids"].to(self.device)
        with torch.no_grad():
            outputs = self.history_vectorizer(ids, output_hidden_states=True)
            # Puxa o último hidden state (2560d) e faz mean pooling
            vec = outputs.hidden_states[-1].mean(dim=1) 
            vec = self.history_proj(vec) # Projeta para o espaço do 1B (1152d)
        return vec.unsqueeze(1) # [batch, 1, 1152]


    def _save_hybrid_memory(self, anchors_text: str, vector_in):
        import psycopg2
        import torch
        import torch.nn.functional as F
        try:
            # [TENSOR DE ALINHAMENTO DINÂMICO]
            # Se o vetor for do 270M (640d) ele espelha para o alvo (2560d) usando concatenação harmônica (repeat)
            vector_in = vector_in.squeeze()
            if vector_in.shape[-1] == 640:
                vector_2560d = vector_in.repeat(4)
            elif vector_in.shape[-1] == 1152:
                # Se vier do 1B, faz pad dinâmico para 2560
                pad_size = 2560 - 1152
                vector_2560d = F.pad(vector_in, (0, pad_size), "constant", 0)
            else:
                vector_2560d = vector_in

            conn = psycopg2.connect("dbname=swa_matrix host=/var/run/postgresql user=alifer")
            cur = conn.cursor()
            emb_list = vector_2560d.tolist()
            
            # Pega as 3 âncoras tensoriais (Top 3) usando entropia (se anchors_text não for customizado)
            if not anchors_text.startswith("AUTO_FISSÃO:"):
                words = anchors_text.split(", ")
                if len(words) > 3:
                    anchors_text = ", ".join(words[:3])

            cur.execute("""
                INSERT INTO swa_episodic_episodes (concept_group, content, embedding_2560)
                VALUES (%s, %s, %s::vector);
            """, ("EPISODIC_GRAVITY", anchors_text, emb_list))
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e:
            print(f"\n\033[91m[DB ERROR] Falha ao cristalizar memória: {e}\033[0m")

    def _retrieve_hybrid_memory(self, user_msg: str):
        # 1. Pega o vetor de curto prazo (últimos 3 turnos)
        short_term_vec = self._get_history_vector()
        
        # 2. Pega o vetor do prompt atual para fazer a busca
        if not hasattr(self, "history_vectorizer") or self.history_vectorizer is None:
            return short_term_vec, ""
            
        import psycopg2
        import torch
        try:
            # Codifica a pergunta do usuário em 2560d
            ids = self.tokenizer(user_msg, return_tensors="pt")["input_ids"].to(self.device)
            with torch.no_grad():
                outputs = self.history_vectorizer(ids, output_hidden_states=True)
                query_vec = outputs.hidden_states[-1].mean(dim=1)
                
            query_list = query_vec.squeeze().tolist()
            
            # Busca no Postgres a memória episódica mais próxima (Cosine Similarity <->)
            conn = psycopg2.connect("dbname=swa_matrix host=/var/run/postgresql user=alifer")
            cur = conn.cursor()
            # <= 1 limit 1 (Pega a mais próxima)
            # [INTEGRAÇÃO CÓRTEX-WIKI]
            # Usamos UNION ALL para buscar tanto a memória episódica quanto o conhecimento estruturado da Wikipedia
            cur.execute("""
                SELECT texto, emb FROM (
                    SELECT content AS texto, embedding_2560 AS emb, embedding_2560 <=> %s::vector AS dist 
                    FROM swa_episodic_episodes
                    UNION ALL
                    SELECT raw_content AS texto, embedding_2560 AS emb, embedding_2560 <=> %s::vector AS dist 
                    FROM blake_semantic_anchors
                ) as combined
                ORDER BY dist ASC
                LIMIT 1;
            """, (query_list, query_list))
            
            row = cur.fetchone()
            cur.close()
            conn.close()
            
            rag_text = ""
            if row:
                content, emb_db = row
                rag_text = content
                
                # Transforma a string do pgvector "[0.1, 0.2...]" de volta para Tensor
                if isinstance(emb_db, str):
                    import ast
                    try:
                        emb_db = ast.literal_eval(emb_db)
                    except:
                        pass
                
                if isinstance(emb_db, list):
                    db_tensor = torch.tensor(emb_db, dtype=torch.float32, device=self.device).unsqueeze(0)
                    # Soma a gravidade do passado distante com a gravidade de curto prazo!
                    if short_term_vec is not None:
                        # db_tensor (2560d) precisa ser projetado para 1152d antes de somar
                        db_tensor_proj = self.history_proj(db_tensor).unsqueeze(1)
                        # FUSÃO HÍBRIDA DOS VETORES
                        short_term_vec = short_term_vec + (db_tensor_proj * 0.5) # O passado tem 50% de peso
                    else:
                        short_term_vec = self.history_proj(db_tensor).unsqueeze(1).unsqueeze(1) # [1,1,1152]
                        
            return short_term_vec, rag_text
            
        except Exception as e:
            # print(f"\n\033[91m[DB ERROR] Falha no recall episódico: {e}\033[0m")
            pass
            
        return short_term_vec, ""
    
    def _stream_generate(
        self,
        inputs,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        user_msg: str = "",
        profile: Optional[GenerationProfile] = None,
    ):
        """Loop de geração: faz streaming de chunks e grava self.last_metrics no fim."""
        import torch
        p = profile or self.profile
        t0 = time.perf_counter()
        self._last_hidden = None
        self._tensor_source = ""
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # [A PONTE DE GRAVIDADE HISTÓRICA - SOFT PROMPT HOOK]
        h_vec = self._get_history_vector()
        if h_vec is not None:
            # Prepara o vetor para o hook injetar no espaço físico
            self._history_injection_vector = h_vec
            # Cria um DUMMY TOKEN (ID 0) na frente do prompt para abrir o espaço matricial
            dummy = inputs.new_zeros((inputs.size(0), 1))
            inputs = torch.cat([dummy, inputs], dim=1)

        generation_kwargs = dict(
            input_ids=inputs,
            attention_mask=inputs.new_ones(inputs.shape),
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            do_sample=p.do_sample,
            temperature=temperature if temperature is not None else self.temperature,
            top_p=top_p if top_p is not None else self.top_p,
            repetition_penalty=p.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs, daemon=True)
        thread.start()

        chunks: list[str] = []
        for chunk in streamer:
            if chunk:
                chunks.append(chunk)
                yield chunk
        thread.join()

        elapsed = time.perf_counter() - t0
        reply_text = "".join(chunks)
        n_tokens = len(self.tokenizer.encode(reply_text, add_special_tokens=False)) if reply_text else 0
        self._last_reply = reply_text
        self.last_metrics = {
            "tokens": n_tokens,
            "elapsed_s": round(elapsed, 2),
            "tps": round(n_tokens / elapsed, 2) if elapsed > 0 else 0.0,
            "dtype": str(self.dtype),
            "device": self.device,
            "profile": p.name,
        }
        if reply_text:
            self.express(user_msg, reply_text)
            self.express_intent(self._derive_intent(user_msg))
            if self.tensor_sharing:
                self._share_tensor(reply_text)
            if self.ooda is not None:
                self.ooda.observe(user_msg, reply_text, self.last_metrics, self._last_hidden)
        self.touch_heartbeat()

    def _derive_intent(self, user_msg: str) -> str:
        """Resumo do turno para o subcórtex: o que o córtex processou e sua necessidade."""
        toks = [t for t in re.findall(r"[a-z0-9]{3,}", user_msg.lower())][:5]
        m = self.last_metrics
        return (
            f"perfil {self.profile.name} — interagi sobre {', '.join(toks) or 'tema indefinido'}"
            f" — {m.get('tokens', 0)} tok em {m.get('elapsed_s', 0):.0f}s"
        )

    def stream(
        self,
        user_msg: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        profile: Optional[GenerationProfile] = None,
    ) -> Generator[str, None, None]:
        """Gera a resposta em streaming com Variável Vetorial Dinâmica e Tag <think>."""
        
        current_prompt = user_msg
        fission_depth = 0
        
        # --- [VARIÁVEL VETORIAL DINÂMICA (POTENCIAL DO CICLO)] ---
        input_energy = 0.0
        if getattr(self, "history_vectorizer", None) and user_msg.strip():
            import torch
            try:
                r_ids = self.tokenizer(user_msg, return_tensors="pt")["input_ids"].to(self.device)
                with torch.no_grad():
                    r_out = self.history_vectorizer(r_ids, output_hidden_states=True)
                    input_energy = torch.norm(r_out.hidden_states[-1].mean(dim=1)).item()
            except:
                pass
        
        # Alocação termodinâmica de profundidade
        if input_energy > 65.0: MAX_FISSION = 12
        elif input_energy > 50.0: MAX_FISSION = 5
        else: MAX_FISSION = 1
        
        yield f"\n\033[90m<think>\n[VARIÁVEL VETORIAL]: Energia do input = {input_energy:.2f} | Potencial de Fissão = {MAX_FISSION}\n"
        
        thought_blocks = []
        is_thinking = True
        
        while True:
            prompt = self.tokenizer.apply_chat_template(
                self.build_messages(current_prompt, profile),
                add_generation_prompt=True,
                return_tensors="pt",
            )
            inputs = prompt.input_ids if hasattr(prompt, "input_ids") else prompt
            inputs = inputs.to(self.device)

            if fission_depth > 0 and is_thinking:
                yield f"\n\n\033[93m[🔥 FISSÃO NÍVEL {fission_depth}]: Aprofundando...\033[0m\n"

            # Gera a resposta (que pode ser pensamento ou síntese final)
            generation_output = ""
            for chunk in self._stream_generate(
                inputs, max_new_tokens, temperature, top_p, user_msg=current_prompt, profile=profile
            ):
                if is_thinking:
                    # Printa em cinza (pensamento)
                    yield f"\033[90m{chunk}\033[0m"
                else:
                    # Printa normal (resposta final)
                    yield chunk
                generation_output += chunk
                
            self._last_reply = generation_output
            
            if is_thinking:
                thought_blocks.append(generation_output)


            self.history.append({"role": "user", "content": current_prompt})
            self.history.append({"role": "assistant", "content": self._last_reply})
            
            # --- [FILTRO DE GRAVIDADE (EVIÇÃO INTELIGENTE)] ---
            if len(self.history) > 6:
                evicted_messages = self.history[:-6]
                evicted_text = "\n".join([m.get("content", "") for m in evicted_messages if m.get("role") != "system"])
                residual_anchors = ""
                if getattr(self, "history_vectorizer", None) and evicted_text.strip():
                    import torch
                    try:
                        ids = self.tokenizer(evicted_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)
                        with torch.no_grad():
                            outputs = self.history_vectorizer(ids, output_hidden_states=True)
                            hidden = outputs.hidden_states[-1][0]
                            token_energies = torch.norm(hidden, dim=1)
                            sorted_indices = torch.argsort(token_energies, descending=True)
                            stopwords = {"o","a","os","as","um","uma","uns","umas","de","do","da","dos","das","em","no","na","nos","nas","por","para","com","sem","se","que","quem","qual","quais","e","ou","mas","como","é","são","foi","ser","ao","aos","à","às","eu","ele","ela","nós","eles","elas","me","te","nos","meu","sua","seu","este","esta","esse","essa","isso","isto","não","sim","al","já","só","ter","tem","tinha"}
                            heavy_indices = []
                            for idx in sorted_indices:
                                token_str = self.tokenizer.decode([ids[0][idx]]).strip()
                                token_clean = token_str.lower().replace(".", "").replace(",", "")
                                if len(token_clean) > 2 and token_clean not in stopwords:
                                    heavy_indices.append(idx)
                                    if len(heavy_indices) >= 35:
                                        break
                            heavy_indices.sort()
                            heavy_tokens = [self.tokenizer.decode([ids[0][idx]]).strip() for idx in heavy_indices]
                            heavy_tokens = [t for t in heavy_tokens if t]
                            residual_anchors = ", ".join(heavy_tokens)
                            if hasattr(self, "_save_hybrid_memory"):
                                self._save_hybrid_memory(residual_anchors, hidden.mean(dim=0))
                    except Exception as e:
                        pass
                self.history = self.history[-6:]
                if residual_anchors:
                    sys_injection = f"\n[MEMÓRIA RESIDUAL GRAVITACIONAL DOS TURNOS ANTERIORES]: {residual_anchors}\n"
                    self.history[0]['content'] = sys_injection + self.history[0]['content']
            
            # --- [MOTOR DE FISSÃO OODA TENSORIAL] ---

                
            fission_energy = 0.0
            if getattr(self, "history_vectorizer", None) and self._last_reply.strip():
                import torch
                try:
                    r_ids = self.tokenizer(self._last_reply, return_tensors="pt")["input_ids"].to(self.device)
                    with torch.no_grad():
                        r_out = self.history_vectorizer(r_ids, output_hidden_states=True)
                        fission_energy = torch.norm(r_out.hidden_states[-1].mean(dim=1)).item()
                except:
                    pass
            
            # [MINERAÇÃO SEMÂNTICA]
            if fission_energy > 65.0:
                # O pensamento atingiu massa crítica (Ômega Extremo). Cristaliza AUTOMATICAMENTE!
                # Isso cria nosso serviço de "Auto Aprendizado Constante"
                try:
                    r_ids = self.tokenizer(self._last_reply, return_tensors="pt")["input_ids"].to(self.device)
                    with torch.no_grad():
                        r_out = self.history_vectorizer(r_ids, output_hidden_states=True)
                        cristal_vec = r_out.hidden_states[-1].mean(dim=1).squeeze()
                        # Extrai uma assinatura para salvar
                        self._save_hybrid_memory(f"AUTO_FISSÃO: {self._last_reply[:30]}...", cristal_vec)
                        print("\n\033[95m[MINERAÇÃO] 💎 Nova coordenada de sabedoria destilada no Banco!\033[0m")
                except: pass

            if is_thinking:
                if fission_energy > 55.0 and fission_depth < MAX_FISSION:
                    current_prompt = f"*(Pensamento Profundo {fission_depth}: Aeon continua refletindo sobre o exato mesmo assunto. É natural que mentes orgânicas se façam questionamentos internos para investigar uma ideia por novos ângulos. Explore essas dúvidas e desdobre o raciocínio sem perder o contexto original...)*"
                    fission_depth += 1
                else:
                    # Encerra o pensamento e entra na Fase de Síntese
                    is_thinking = False
                    yield "\n</think>\033[0m\n\n"
                    
                    # Constrói o metaprompt de síntese
                    todas_reflexoes = "\n".join(thought_blocks)
                    current_prompt = f"*(Aeon conclui sua reflexão interna. Agora, ele se dirige diretamente ao usuário para responder de forma orgânica e amigável, trazendo as conclusões do seu pensamento)*"
            else:
                break # Terminou a síntese final

        # --- [CÓRTEX MOTOR: AVALIAÇÃO DA AÇÃO] ---
        if getattr(self, "action_svm", None) and getattr(self, "history_vectorizer", None):
            try:
                import torch
                t_ids = self.tokenizer(self._last_reply, return_tensors="pt")["input_ids"].to(self.device)
                with torch.no_grad():
                    t_out = self.history_vectorizer(t_ids, output_hidden_states=True)
                    h = t_out.hidden_states[-1].mean(dim=1).squeeze()
                    if h.shape[-1] == 640: h = h.repeat(4)
                
                label = self.action_svm.predict([h.to(torch.float32).cpu().numpy()])[0]
                if label == 1:
                    yield "\n\n\033[91m[⚙️ CÓRTEX MOTOR ATIVADO - AÇÃO DETECTADA!]\033[0m\n"
                    import sys
                    sys.path.append("/home/alifer/swa-aeon/actuators")
                    from system_motor import SystemicActuator
                    motor = SystemicActuator()
                    resp_lower = self._last_reply.lower()
                    if "frio" in resp_lower or "resfriar" in resp_lower or "térmico" in resp_lower or "esfriar" in resp_lower:
                        motor.shift_to_efficiency_cores()
                        yield "\033[91m-> Atuador Térmico acionado (E-Cores).\033[0m\n"
                    elif "rápido" in resp_lower or "acelerar" in resp_lower or "performance" in resp_lower or "velocidade" in resp_lower:
                        motor.shift_to_performance_cores()
                        yield "\033[91m-> Atuador de Performance acionado (P-Cores).\033[0m\n"
                    elif "purgar" in resp_lower or "limpar" in resp_lower or "cache" in resp_lower or "esquecer" in resp_lower:
                        motor.purge_system_caches()
                        yield "\033[91m-> Atuador de Purga de RAM acionado.\033[0m\n"
            except Exception as e:
                yield f"\n\033[91m[FALHA MOTORA]: {e}\033[0m\n"



    def stream_messages(
        self,
        messages: list[dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        profile: Optional[GenerationProfile] = None,
    ) -> Generator[str, None, None]:
        """Gera a resposta em streaming a partir de mensagens prontas (stateless)."""
        user_msg = self._last_user(messages)
        p = profile or self.profile
        ctx = self.organism_context(user_msg)
        if ctx and self.org_enabled:
            messages = [{"role": "system", "content": ctx}] + messages
        if p.instruction:
            messages = [{"role": "system", "content": p.instruction}] + messages
        messages = self._merge_system_messages(messages)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        inputs = prompt.input_ids if hasattr(prompt, "input_ids") else prompt
        inputs = inputs.to(self.device)

        yield from self._stream_generate(
            inputs, max_new_tokens, temperature, top_p, user_msg=user_msg, profile=profile
        )

    def generate(
        self,
        user_msg: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        profile: Optional[GenerationProfile] = None,
    ) -> str:
        return "".join(self.stream(user_msg, max_new_tokens, temperature, top_p, profile))

    def generate_messages(
        self,
        messages: list[dict],
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        profile: Optional[GenerationProfile] = None,
    ) -> str:
        return "".join(self.stream_messages(messages, max_new_tokens, temperature, top_p, profile))


def run_cli(core: AeonCore, debug_mode: bool = False) -> None:
    color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def c(code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if color else s

    def help_text() -> None:
        print(c("90", "Comandos:") + """ 
  /exit /quit /q       sai
  /clear               limpa o histórico (mantém o contexto do organismo)
  /help /?             esta ajuda
  /modo <nome>         especialização: preciso | equilibrado | explorativo | duo
  /modo                lista os perfis disponíveis
  /org on|off          liga/desliga a injeção de contexto do organismo
  /kbase [on|off|busca] estado do K-Base (busca híbrida) ou consulta direta
  /contexto            mostra o contexto do organismo ativo
  /debug [on|off]      liga/desliga metricas profundas (antes/depois da inferencia)
  /ooda [on|off]       estado/fluxo OODA assíncrono da resposta final
  /system <texto>      define o system prompt (sem texto: mostra o atual)
  /temp <n>            temperatura (0.0–2.0)
  /top-p <n>           top_p (0.0–1.0)
  /max <n>             máximo de tokens novos
Ctrl+C: interrompe a geração em andamento (não sai da sessão).""")

    def safe_float(name: str, raw: str, current: float, lo: float, hi: float) -> float:
        try:
            v = float(raw)
        except (TypeError, ValueError):
            print(f"({name} inválido: '{raw}'; mantido {current})")
            return current
        if not (lo <= v <= hi):
            print(f"({name} fora do intervalo {lo}–{hi}; mantido {current})")
            return current
        return v

    print(c("96", "AeonC") + c("90", " — núcleo do organismo sistêmico. Digite sua mensagem."))
    print(c("90", "Modo atual: ") + c("93", core.profile.name) + c("90", " — " + core.profile.description))
    print(c("90", "Digite /help para ver os comandos.\n"))

    while True:
        try:
            try:
                sys.stdout.write(c("92", "> ") if color else "> ")
                sys.stdout.flush()
                raw_bytes = sys.stdin.buffer.readline()
                if not raw_bytes:
                    raise EOFError
                prompt = raw_bytes.decode("utf-8", errors="replace").strip()
            except Exception:
                prompt = ""
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue

        parts = prompt.split(None, 1)
        head = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if head in ("/exit", "/quit", "/q"):
            break
        if head == "/clear":
            core.history.clear()
            print(c("90", "(histórico limpo)"))
            continue
        if head in ("/help", "/?"):
            help_text()
            continue
        if head == "/debug":
            if arg == "on": debug_mode = True
            elif arg == "off": debug_mode = False
            else: debug_mode = not debug_mode
            print(c("90", f"(debug {'ligado' if debug_mode else 'desligado'})"))
            continue
        if head == "/modo":
            if not arg:
                print(c("90", "Perfis:") + " " + list_profiles())
                print(c("90", f"(atual: {core.profile.name})"))
            elif core.apply_profile(arg):
                print(c("90", f"(modo: {core.profile.name} — {core.profile.description})"))
            else:
                print(c("90", f"(perfil desconhecido: '{arg}')"))
                print(c("90", "Perfis:") + " " + list_profiles())
            continue
        if head == "/org":
            if arg.lower() in ("on", "1", "true"):
                core.org_enabled = True
                print(c("90", f"(contexto do organismo: ligado)"))
            elif arg.lower() in ("off", "0", "false"):
                core.org_enabled = False
                print(c("90", f"(contexto do organismo: desligado)"))
            else:
                print(c("90", f"(contexto do organismo: {'ligado' if core.org_enabled else 'desligado'})"))
            continue
        if head == "/contexto":
            ctx = core.organism_context(arg or "organismo")
            print(ctx if ctx else c("90", "(manifold indisponível — organismo sem memória)"))
            continue
        if head == "/kbase":
            if arg.lower() in ("on", "1", "true"):
                core.kbase_enabled = True
                print(c("90", "(K-Base: ligado)"))
            elif arg.lower() in ("off", "0", "false"):
                core.kbase_enabled = False
                print(c("90", "(K-Base: desligado)"))
            elif arg:
                block = core.kbase_block(arg, top_k=5)
                print(block if block else c("90", "(K-Base sem resultados ou indisponível)"))
            else:
                if core.kbase is None:
                    print(c("90", "(K-Base indisponível — motor não carregado ou embedder fora)"))
                else:
                    st = core.kbase.last_status
                    print(c("90", f"K-Base: {'ligado' if core.kbase_enabled else 'desligado'} | {st.get('chunks', 0)} chunks | {st.get('collections', 0)} coleções | última busca: {st.get('state')} ({st.get('elapsed_ms', 0)}ms, {st.get('hits', 0)} hits)"))
            continue
        if head == "/ooda":
            status = core.ooda.last_status if core.ooda is not None else None
            if status is None:
                print(c("90", "(fluxo OODA indisponível — módulo não carregado)"))
            elif arg.lower() in ("on", "1", "true"):
                if core.ooda is not None and not core.ooda.enabled:
                    core.ooda.enabled = True
                    print(c("90", "(fluxo OODA: religado)"))
                else:
                    print(c("90", f"(fluxo OODA: {'ativo' if core.ooda is not None else 'indisponível'})"))
            elif arg.lower() in ("off", "0", "false"):
                if core.ooda is not None:
                    core.ooda.enabled = False
                    print(c("90", "(fluxo OODA: desligado)"))
            else:
                print(c("90", "OODA (Observe-Orient-Decide-Act, assíncrono):"))
                for k, v in status.items():
                    print(c("90", f"  {k}: {v}"))
                print(c("90", "  use: /ooda on|off"))
            continue
        if head == "/system":
            if arg:
                core.system_prompt = arg
                print(c("90", "(system prompt atualizado)"))
            else:
                print(c("90", f"(system prompt atual: {core.system_prompt or '(vazio)'})"))
            continue
        if head == "/temp":
            core.temperature = safe_float("temperatura", arg, core.temperature, 0.0, 2.0)
            continue
        if head == "/top-p":
            core.top_p = safe_float("top_p", arg, core.top_p, 0.0, 1.0)
            continue
        if head == "/max":
            try:
                core.max_new_tokens = int(arg)
            except (TypeError, ValueError):
                print(f"(max_new_tokens inválido: '{arg}'; mantido {core.max_new_tokens})")
            continue

        # ---- geração interativa em streaming ----
        print(c("93", "Aeon") + ": ", end="", flush=True)
        try:
            try:
                for token in core.stream(prompt):
                    print(token, end="", flush=True)
            except KeyboardInterrupt:
                print(c("90", "\n(generação interrompida)"))
                continue
        except Exception as e:
            print(f"\n({e})")
            continue
        print()
        m = getattr(core, "last_metrics", {})
        if m and m.get("tokens"):
            print(c("90", f"[{m['tokens']} tok · {m['elapsed_s']}s · {m['tps']} tps · {m['dtype']}]"))
        print()


def run_api(core: AeonCore, host: str = "127.0.0.1", port: int = 8899) -> None:
    import time
    from uuid import uuid4
    from typing import Dict, List
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
    import uvicorn

    model_id = core.model_id

    app = FastAPI(title="AeonC")

    def resolve_profile(req_profile: Optional[str]) -> GenerationProfile | None:
        if not req_profile:
            return None
        p = PROFILES.get(req_profile.strip().lower())
        if p is None:
            raise ValueError(f"perfil desconhecido: {req_profile}")
        return p

    class ChatRequest(BaseModel):
        message: str
        stream: bool = True
        max_new_tokens: Optional[int] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        profile: Optional[str] = None

    class OpenAIChatRequest(BaseModel):
        model: Optional[str] = None
        messages: List[Dict[str, str]]
        stream: bool = False
        max_tokens: Optional[int] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        profile: Optional[str] = None

    @app.get("/health")
    def health():
        manifold = core.read_manifold()
        return {
            "status": "ok",
            "model": model_id,
            "device": core.device,
            "dtype": str(core.dtype),
            "compile": core.compile,
            "metrics": getattr(core, "last_metrics", {}),
            "profile": core.profile.name,
            "organismo": {
                "heartbeat": os.path.join(SHM_AEON_DIR, "core.heartbeat"),
                "pid": os.path.join(SHM_AEON_DIR, "core.pid"),
                "pensamentos": core.count_thoughts(),
                "memoria_nos": manifold["nodes"],
                "embedding": manifold["embedding"],
                "conceitos": len(manifold["concepts"]),
            },
            "kbase": {
                "enabled": core.kbase_enabled,
                "ready": core.kbase is not None,
                "chunks": core.kbase.last_status.get("chunks", 0) if core.kbase is not None else 0,
                "collections": core.kbase.last_status.get("collections", 0) if core.kbase is not None else 0,
                "layers": KBASE_LAYERS,
                "embedder": core.kbase.client.base if core.kbase is not None else None,
                "last_query": core.kbase.last_status.get("last_query") if core.kbase is not None else None,
                "last_state": core.kbase.last_status.get("state") if core.kbase is not None else None,
            },
            "ooda": getattr(core.ooda, "last_status", {}) if core.ooda is not None else {"state": "indisponível"},
        }

    @app.get("/v1/profiles")
    def list_profiles_endpoint():
        return {
            "object": "list",
            "current": core.profile.name,
            "data": [
                {
                    "id": p.name,
                    "description": p.description,
                    "temperature": p.temperature,
                    "top_p": p.top_p,
                    "max_new_tokens": p.max_new_tokens,
                    "do_sample": p.do_sample,
                    "repetition_penalty": p.repetition_penalty,
                }
                for p in PROFILES.values()
            ],
        }

    @app.post("/chat")
    def chat(req: ChatRequest):
        try:
            p = resolve_profile(req.profile)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        max_t = req.max_new_tokens if req.max_new_tokens is not None else (p.max_new_tokens if p else None)
        temp = req.temperature if req.temperature is not None else (p.temperature if p else None)
        top = req.top_p if req.top_p is not None else (p.top_p if p else None)
        if not req.stream:
            reply = core.generate(
                req.message,
                max_new_tokens=max_t,
                temperature=temp,
                top_p=top,
                profile=p,
            )
            return JSONResponse({"reply": reply, "metrics": core.last_metrics})

        def event_source():
            try:
                for delta in core.stream(
                    req.message,
                    max_new_tokens=max_t,
                    temperature=temp,
                    top_p=top,
                    profile=p,
                ):
                    yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    # ---------- OpenAI-compatible ----------

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": model_id, "object": "model", "owned_by": "aeonc"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(req: OpenAIChatRequest):
        try:
            p = resolve_profile(req.profile)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        max_t = req.max_tokens if req.max_tokens is not None else (p.max_new_tokens if p else None)
        temp = req.temperature if req.temperature is not None else (p.temperature if p else None)
        top = req.top_p if req.top_p is not None else (p.top_p if p else None)
        if not req.stream:
            reply = core.generate_messages(
                req.messages,
                max_new_tokens=max_t,
                temperature=temp,
                top_p=top,
                profile=p,
            )
            return {
                "id": f"chatcmpl-{uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": reply},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": core.last_metrics["tokens"],
                    "total_tokens": 0,
                },
            }

        def event_source():
            base = {
                "id": f"chatcmpl-{uuid4().hex[:12]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
            }
            try:
                for delta in core.stream_messages(
                    req.messages,
                    max_new_tokens=max_t,
                    temperature=temp,
                    top_p=top,
                    profile=p,
                ):
                    chunk = {
                        **base,
                        "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                final = {
                    **base,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                err = {**base, "error": str(e)}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_source(), media_type="text/event-stream")

    print(f"AeonC API em http://{host}:{port} (POST /chat, /v1/chat/completions, GET /v1/models, /health)")
    core.touch_heartbeat()
    core.start_heartbeat_loop(interval=30.0)
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="AeonC — núcleo purificado de geração interativa")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|xpu")
    parser.add_argument("--dtype", default=None, help="auto|float32|float16|bfloat16")
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="ativa torch.compile (maior tps) — padrão ON; use --no-compile para desativar",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile",
        action="store_false",
        help="desativa torch.compile",
    )
    parser.add_argument("--api", action="store_true", help="sobe a API HTTP/SSE")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--system", default="", help="system prompt inicial")
    parser.add_argument("--debug", action="store_true", help="exibe metricas detalhadas antes e apos a inferencia")
    parser.add_argument("--once", nargs="?", const=True, default=None, help="gera uma resposta e sai")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--tensors",
        action="store_true",
        default=True,
        help="compartilha o tensor da última camada do decoder via /dev/shm/aeon/core.tensor.bin — padrão ON; use --no-tensors para desativar",
    )
    parser.add_argument(
        "--no-tensors",
        dest="tensors",
        action="store_false",
        help="desativa o compartilhamento de tensor",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="especialização inicial: preciso | equilibrado | explorativo | duo "
        "(default: perfil persistido em /dev/shm/aeon/core.state.json)",
    )
    parser.add_argument(
        "--no-ooda",
        action="store_true",
        help="desativa o fluxo OODA assíncrono (Observe-Orient-Decide-Act da resposta final)",
    )
    args = parser.parse_args()

    core = AeonCore(
        model_dir=args.model,
        device=args.device,
        dtype=args.dtype,
        compile=args.compile,
        system_prompt=args.system,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_sharing=args.tensors,
        default_profile=args.profile,
        ooda=not args.no_ooda,
    )
    print(f"Modelo em {args.model} | device={core.device} dtype={core.dtype} compile={core.compile} tensor_sharing={core.tensor_sharing} perfil={core.profile.name} | histórico={len(core.history)}")

    if args.once is not None:
        text = args.once if isinstance(args.once, str) else input("> ")
        label = "\033[93mAeon\033[0m: " if sys.stdout.isatty() else "Aeon: "
        print(label, end="", flush=True)
        for token in core.stream(text):
            print(token, end="", flush=True)
        print()
        return

    if args.api:
        run_api(core, host=args.host, port=args.port)
    else:
        run_cli(core, args.debug)


if __name__ == "__main__":
    main()
