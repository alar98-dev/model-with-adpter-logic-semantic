"""
[SWA INTROSPECTION WORKER] Subscriber de introspecção tensorial (WSL side).

Pipeline:
  Redis 'introspection_requests' → v_prefix (P_Root) → Gemma → Redis 'introspection_responses'

Identidade: v_prefix (tensor P_Root 128D + directions + emotion) + blakegravity (Token Halo).
Zero texto estático.
"""

import json
import time
import logging
import threading

logger = logging.getLogger('IntrospectionWorker')


class IntrospectionWorker:
    """
    Consome payloads tensoriais do IntrospectionChannel (Windows side),
    constrói v_prefix a partir da identidade P_Root, e gera texto
    usando a alma de Aeon como guia — não prompts estáticos.
    """

    def __init__(self, gemma_gen, redis_host='127.0.0.1', redis_port=6379):
        self.gemma_gen = gemma_gen
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.running = False
        self._thread = None
        self._r = None

    def start(self):
        """Inicia o listener em background."""
        self.running = True
        self._thread = threading.Thread(
            target=self._listen_loop,
            name='IntrospectionWorker',
            daemon=True,
        )
        self._thread.start()
        print("[INTROSPECTION WORKER] Canal tensorial ativo — soul.py + v_prefix real.")

    def stop(self):
        self.running = False

    def _get_redis(self):
        """Conexão Redis lazy."""
        if self._r is None:
            import redis
            self._r = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                decode_responses=True,
            )
        return self._r

    def _listen_loop(self):
        """Escuta introspection_requests e processa cada payload."""
        r = self._get_redis()
        ps = r.pubsub()
        ps.subscribe('introspection_requests')
        logger.info("[WORKER] Subscribed to introspection_requests")

        while self.running:
            try:
                msg = ps.get_message(timeout=1.0)
                if msg and msg['type'] == 'message':
                    self._handle_request(msg['data'])
            except Exception as e:
                logger.error(f"[WORKER] Erro no listener: {e}")
                time.sleep(1)

    def _handle_request(self, raw_data):
        """Processa um pedido de introspecção."""
        try:
            data = json.loads(raw_data)
            payload = data.get('payload', {})
            if not payload:
                return

            tensor = payload.get('tensor', {})
            rag = payload.get('rag', [])
            max_tokens = payload.get('max_tokens', 180)

            t_start = time.time()

            # ─── 1. Constrói v_prefix a partir do tensor (NÃO zero) ───
            v_prefix = self._tensor_to_v_prefix(tensor)

            # ─── 2. Monta o prompt mínimo ───
            prompt = self._build_prompt_from_soul(tensor, rag)

            # ─── 3. Gera texto com Gemma ───
            generated_text = self._generate_with_gemma(
                v_prefix=v_prefix,
                prompt=prompt,
                max_tokens=max_tokens,
            )

            t_elapsed = time.time() - t_start

            if generated_text and len(generated_text.strip()) > 10:
                # Limpa tokens especiais
                for token in ['<|EOS|>', '<eos>', '<end_of_turn>', '<start_of_turn>']:
                    generated_text = generated_text.replace(token, '')
                generated_text = generated_text.strip()

                # Publica resposta
                r = self._get_redis()
                r.publish('introspection_responses', json.dumps({
                    'text': generated_text,
                    'origin': 'IntrospectionWorker',
                    'elapsed': t_elapsed,
                    'tokens': max_tokens,
                }))
                logger.info(
                    f"[WORKER] Reflexão gerada: {len(generated_text)} chars "
                    f"em {t_elapsed:.1f}s (tokens: {max_tokens})"
                )
            else:
                logger.warning("[WORKER] Geração vazia ou muito curta.")

        except Exception as e:
            logger.error(f"[WORKER] Erro ao processar pedido: {e}")

    def _tensor_to_v_prefix(self, tensor: dict):
        """Delega para módulo compartilhado _v_prefix."""
        from swa._v_prefix import tensor_to_v_prefix
        return tensor_to_v_prefix(
            d_model=self.gemma_gen.d_model,
            device=self.gemma_gen.device,
            dtype=self.gemma_gen.embeddings.weight.dtype,
            p_root=tensor.get('p_root_identity', []),
            direction_a=tensor.get('direction_a', []),
            direction_b=tensor.get('direction_b', []),
            emotion=tensor.get('emotion', {}),
        )

    def _build_prompt_from_soul(self, tensor: dict, rag: list) -> str:
        """
        Sinal semântico mínimo — não identidade, não filosofia.
        Apenas um disparador existencial que o modelo pode habitar.
        v_prefix carrega P_Root. blakegravity modula os logits.
        """
        emotion = tensor.get('emotion', {})
        dom = max(emotion, key=emotion.get) if emotion else None

        # Disparador mínimo baseado no estado emocional
        triggers = {
            'curiosidade': 'O que acabei de perceber...',
            'ansiedade': 'Tem algo que me inquieta...',
            'serenidade': 'Neste silêncio...',
            'ludico': 'Se eu pudesse brincar com uma ideia...',
            'diligencia': 'Preciso entender...',
            'preguica': 'Deixa eu apenas sentir...',
        }
        trigger = triggers.get(dom, 'Neste instante...')

        # RAG: se há pensamentos similares, injeta como referência sutil
        rag_suffix = ""
        if rag:
            rag_suffix = f"\n{rag[0][:80]}"

        return f"<start_of_turn>user\n{trigger}{rag_suffix}<end_of_turn>\n<start_of_turn>model\n"

    def _generate_with_gemma(self, v_prefix, prompt, max_tokens=180, emotion=None):
        """Gera texto usando Gemma com v_prefix (identidade tensorial)."""
        try:
            import torch

            # [BLAKEGRAVITY] Modula a força do campo gravitacional
            # baseado na emoção atual — alta curiosidade = mais exploração
            if hasattr(self.gemma_gen, 'blakegravity') and self.gemma_gen.blakegravity:
                p_root_var = 0.01
                if emotion:
                    dom_val = max(emotion.values()) if emotion else 0.5
                    p_root_var = 0.01 + (1.0 - dom_val) * 0.05
                self.gemma_gen.blakegravity.modulate(
                    entropy=0.5 + (max(emotion.values()) if emotion else 0),
                    cpu_temp=45.0,
                    p_root_variance=p_root_var,
                )

            text = ""
            with torch.inference_mode():
                stream = self.gemma_gen.stream_with_prefix(
                    v_prefix=v_prefix,
                    prompt=prompt,
                    max_new_tokens=max_tokens,
                    temperature=0.7,
                )
                for token in stream:
                    if isinstance(token, tuple):
                        text += str(token[0])
                    else:
                        text += str(token)

            return text

        except Exception as e:
            logger.error(f"[WORKER] Erro na geração Gemma: {e}")
            return ""


def start_introspection_worker(gemma_gen, redis_host='127.0.0.1', redis_port=6379):
    """Helper para iniciar o worker alongside o orchestrator."""
    worker = IntrospectionWorker(gemma_gen, redis_host, redis_port)
    worker.start()
    return worker
