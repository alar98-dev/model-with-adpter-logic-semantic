"""
SWA Hybrid Steering Plugin — Pluga o olho do polvo no gerador de voz do runtime.

Exposição: attach_hybrid_steering(runtime) → adiciona runtime.hybrid_stream(...)
e um callback de OODA que injeta a decisão híbrida.

Comportamento (todo guardado, nunca derruba o runtime):
  - Gera steering vector (1,1,model_dim) via HybridInferenceAdapter
    (k-base 270M → 2560D, diff relativo, bandit+markov, heurística).
  - Se o bandit decide CUT/CLIMAX (alto impacto) → usa stream_with_prefix
    com o steering como v_prefix (geração híbrida ancorada na k-base).
  - Caso contrário → delega para o chat_stream padrão (sem overhead).
  - Ou explicitamente: runtime.hybrid_stream(prompt, force=True)
    força o caminho híbrido sempre que quiser.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, Optional

# ------------------------------------------------------------------ helpers
def _model_dim_of(runtime) -> int:
    try:
        eng = runtime.gemma_voice if hasattr(runtime, "gemma_voice") else None
        if eng is not None and hasattr(eng, "embeddings"):
            return int(eng.embeddings.weight.shape[-1])
    except Exception:
        pass
    # Gemma-3-1B: hidden_size=1152 (MODEL_DIM)
    return 1152


def _get_hybrid(runtime) -> Any:
    """Lazy singleton do adaptador híbrido no runtime."""
    adapter = getattr(runtime, "_hybrid_adapter", None)
    if adapter is None:
        try:
            import sys
            e_path = os.environ.get(
                "AEON_ENGINE_PATH",
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aeon-engine"),
            )
            # Tenta caminho dentro do repo swa-aeon (submódulo) e a árvore aeon-engine
            for p in [
                e_path,
                "/home/alifer/aeon-engine",
                "/services/model-with-adpter-logic-semantic/aeon-engine",
            ]:
                if os.path.isdir(p):
                    if p not in sys.path:
                        sys.path.insert(0, p)
                    break
            from hybrid.hybrid_inference import HybridInferenceAdapter
            dim = _model_dim_of(runtime)
            adapter = HybridInferenceAdapter(model_dim=dim)
            runtime._hybrid_adapter = adapter
        except Exception as e:
            print(f"⚠️ [HYBRID] Adapter indisponível: {e}")
            runtime._hybrid_adapter = False
            return None
    return adapter if adapter is not False else None


def _hybrid_should_steer(runtime, text: str) -> bool:
    """Decisão bandit: usar steering (CUT/CLIMAX) ou streaming normal."""
    try:
        adapter = _get_hybrid(runtime)
        if adapter is None:
            return False
        pred = adapter.decide(text, z_acoustic=0.6, duration=30)
        action = pred.get("action", "STREAM")
        state = pred.get("state", "CONTEXT")
        # Gera com steering quando há alta intenção (corte/clímax)
        return action == "CUT" or state == "CLIMAX"
    except Exception:
        return False


def hybrid_generate(
    runtime,
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.7,
    anchor_intent: Optional[str] = None,
    force: bool = False,
    conversation_key: Optional[str] = None,
) -> Iterator[str]:
    """Gera texto híbrido: steering 270M + heuristic + streaming Gemma-1B.

    Se force ou o bandit decidir alto impacto, usa stream_with_prefix com o
    v_prefix do steering do polvo; senão delega ao chat_stream padrão.
    `conversation_key` (quando disponível) vira a chave relacional da onda na
    metarealidade — cada turno do polvo registra um "rei" na Tábua dos Reinados.
    """
    try:
        adapter = _get_hybrid(runtime)
        gemma = runtime.gemma_voice if hasattr(runtime, "gemma_voice") else None

        if adapter is not None and gemma is not None and (force or _hybrid_should_steer(runtime, prompt)):
            try:
                import torch
                # REPIO: enriquece a intenção com o diff relativo da k-base (se houver)
                diff_context = ""
                repio = getattr(runtime, "_repio", None)
                if repio is not None:
                    try:
                        hits = repio.recall(prompt, k=3)
                        if hits:
                            diff_context = " | ".join(
                                (h.get("content") or "")[:200] for h in hits
                            )[:1500]
                    except Exception:
                        pass

                # [SWA] Contexto da metarealidade funde com o diff relativo da k-base
                # (2ª âncora de realidade: memória externa → realidade interna).
                try:
                    meta_org = getattr(runtime, "metareality", None)
                    if meta_org is not None and hasattr(meta_org, "context"):
                        meta_txt = meta_org.context(user_msg=prompt, max_chars=400)
                        if meta_txt:
                            diff_context = (
                                (diff_context + " | " + meta_txt) if diff_context else meta_txt
                            )[:1500]
                except Exception:
                    pass

                v_prefix = adapter.build_steering(
                    prompt,
                    model_dim=_model_dim_of(runtime),
                    z_acoustic=0.6,
                    duration=30,
                    anchor_intent=anchor_intent,
                    context=diff_context or None,
                ).to(device=gemma.device, dtype=gemma.embeddings.weight.dtype)

                # [SWA] Macro-prefixo v2: empilha o vetor de realidade da metarealidade
                # como 2ª âncora (reality_index/emotion/metamatter). Só acopla se houver
                # sinal real do subcórtex — vector() retorna None na ausência.
                try:
                    meta_org = getattr(runtime, "metareality", None)
                    meta_org_vec = getattr(meta_org, "vector", None)
                    if meta_org_vec is not None:
                        rv = meta_org_vec(model_dim=_model_dim_of(runtime))
                        if rv is not None:
                            # [TÁBUA] âncora adaptativa: drift alto do ciclo →
                            # amplia o peso da realidade no macro-prefixo.
                            try:
                                alpha = float(
                                    getattr(meta_org, "cycle_weight", lambda: 1.0)()
                                )
                            except Exception:
                                alpha = 1.0
                            rv = rv * alpha
                            v_prefix = torch.cat(
                                [v_prefix, rv.to(device=v_prefix.device, dtype=v_prefix.dtype)],
                                dim=1,
                            )
                            print(
                                f"🔗 [HYBRID] v_prefix macro: k-base + realidade x{alpha:.2f}",
                                flush=True,
                            )
                except Exception as _rv_err:
                    print(f"⚠️ [HYBRID] Fusão realidade no prefixo falhou: {_rv_err}", flush=True)

                # stream_with_prefix retorna só texto (iterável de str), não tuplas
                stream = gemma.stream_with_prefix(
                    v_prefix=v_prefix,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                # normalize como o chat_stream (tuplas são metadados)
                reply_parts: list = []
                for t in stream:
                    if isinstance(t, tuple):
                        continue
                    if t:
                        reply_parts.append(t)
                        yield t
                # Auto-memória: persistir prompt + resposta na k-base (REPIO)
                # só no caminho steering (conteúdo de alto impacto).
                try:
                    hybrid_ingest(runtime, prompt, concept="OBSERVE")
                    reply = "".join(reply_parts)[:2000]
                    if reply:
                        hybrid_ingest(runtime, reply, concept="RESPONSE")
                except Exception:
                    pass
                _observe_turn(runtime, prompt, "".join(reply_parts), conversation_key)
                return
            except Exception as e:
                print(f"⚠️ [HYBRID] Steering falhou ({e}); fallback chat_stream.")
                runtime._hybrid_fallback_reset = True

        # Fallback: chat_stream padrão (zero mudança de comportamento)
        if gemma is not None:
            fb_parts: list = []
            for t in gemma.chat_stream(
                user_msg=prompt,
                system_msg="",
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            ):
                if t:
                    fb_parts.append(t)
                    yield t
            _observe_turn(runtime, prompt, "".join(fb_parts), conversation_key)
            return
        # Último recurso: não há modelo → nada
        return
    except Exception as e:
        print(f"⚠️ [HYBRID] Erro ao gerar ({e}); vazio.")
        return


# ------------------------------------------------------------------ ingest
def hybrid_ingest(runtime, text: str, concept: Optional[str] = None, source: str = "voice") -> bool:
    """Auto-memória: persiste um texto como âncora na k-base (REPIO).

    Best-effort, nunca lança. Só grava se PG disponível e texto não-vazio.
    """
    if not text or not text.strip():
        return False
    repio = getattr(runtime, "_repio", None)
    if repio is None:
        return False
    try:
        return bool(repio.ingest(text.strip()[:2000], concept=concept, source=source))
    except Exception:
        return False


def _observe_turn(runtime, user_msg: str, reply: str, conversation_key: Optional[str] = None) -> None:
    """Registra o turno na metarealidade (evento + onda relacional).

    Best-effort, nunca lança: o órgão pode estar ausente (metarealidade
    desligada) sem afetar a geração híbrida. Com `conversation_key`, `observe`
    também grava a onda relacional (Tábua dos Reinados); sem ela, só o evento
    `metareality`.
    """
    try:
        org = getattr(runtime, "metareality", None)
        observe = getattr(org, "observe", None)
        if observe is None:
            return
        observe(user_msg, reply or "", wave=0, conversation_key=conversation_key)
    except Exception:
        pass


# ------------------------------------------------------------------ ooda
def register_hybrid_ooda(runtime, ooda_flow) -> bool:
    """Placa a decisão do polvo no ciclo OODA existente (não-bloqueante).

    - Publica o diagnóstico octopus (action/state/intensity) no state do OODA.
    - Expõe runtime._hybrid_decision para telemetria/consciência do runtime.
    Guardado: sem órgão OODA ou sem adaptador, apenas retorna False.
    """
    try:
        if ooda_flow is None or not hasattr(ooda_flow, "on_cycle"):
            return False
        adapter = getattr(runtime, "_hybrid_adapter", None)

        def _inject(state):
            try:
                # Tenta extrair alguma semântica corrente do ciclo OODA.
                seed = ""
                for attr in ("bnf_tag", "decision", "raw_tokens"):
                    v = getattr(state, attr, None)
                    if isinstance(v, str) and v.strip():
                        seed = v
                        break
                pred = (
                    adapter.decide(seed or "contexto geral", z_acoustic=0.5, duration=30)
                    if adapter is not None
                    else {}
                )
                runtime._hybrid_decision = pred
                try:
                    setattr(state, "hybrid", pred)
                except Exception:
                    pass
            except Exception:
                runtime._hybrid_decision = {}

        # injeta direto como callback extra do órgão OODA
        ooda_flow.on_cycle(_inject)
        return True
    except Exception:
        return False


# ------------------------------------------------------------------ attach
def attach_hybrid_steering(runtime) -> bool:
    """Adiciona runtime.hybrid_stream e placa o steering no OODA.

    Retorna True se o adaptador carregou (k-base 270M ativa), False se só
    estilo fallback. Nunca lança exceção.
    """
    try:
        runtime.hybrid_stream = lambda prompt, **kw: hybrid_generate(runtime, prompt, **kw)
        adapter = _get_hybrid(runtime)
        attached = adapter is not None

        # REPIO (k-base PG): recall relativo + ingestão de memória contínua.
        # Guardado — sem PG a engine apenas degrada para o diff local.
        try:
            from hybrid.repio_engine import get_repio
            runtime._repio = get_repio(adapter)
        except Exception:
            runtime._repio = None
        runtime._repio_available = bool(getattr(runtime, "_repio", None) is not None)

        # Placa um status visível para telemetria
        try:
            status = adapter.status() if adapter else {}
            if runtime._repio is not None:
                status.update(runtime._repio.status())
            runtime._hybrid_status = status
        except Exception:
            runtime._hybrid_status = {}
        return attached
    except Exception:
        runtime.hybrid_stream = lambda prompt, **kw: (
            (y for y in [""]) if hasattr(runtime, "gemma_voice") and runtime.gemma_voice is None else iter(())
        )
        return False


def hybrid_status(runtime) -> Dict[str, Any]:
    return getattr(runtime, "_hybrid_status", {})
