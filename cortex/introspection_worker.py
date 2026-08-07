"""
[IntROSPECTION WORKER] Motor tensorial de introspecção no WSL.

Recebe payloads tensoriais do IntrospectionChannel (Windows),
constrói v_prefix a partir do estado sensorial de Aeon,
e gera texto via Gemma com sensory embedding como input primário.

Inclui gravidade Wikipedia: consulta blake_semantic_anchors para
alimentar o conhecimento absorvido como proximidade gravitacional
dos pensamentos e palavras.

Modos:
  - tensorial: v_prefix = f(P_Root identity, emotion, geosense gravity)
               + knowledge = blake_semantic_anchors (Wikipedia gravity)
               input = sensory embedding (bypass de texto)
  - fallback:  chat_stream com prompt compacto (se tensor indisponível)
"""

import json
import threading
import time
import logging
import os
import sys
import redis
import numpy as np

from swa import model_scheduler

logger = logging.getLogger('IntrospectionWorker')

# File-based logging for crash diagnostics
_LOG_FILE = '/tmp/introspection_worker.log'
_file_handler = logging.FileHandler(_LOG_FILE, mode='a')
_file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(_file_handler)
logger.setLevel(logging.DEBUG)
logger.info('Module loaded')


def _query_wiki_gravity(emotion: dict = None, top_k: int = 3) -> list:
    """
    Consulta blake_semantic_anchors para Knowledge mais pesado (gravidade).
    Retorna snippets de Wikipedia que ressoam com o estado emocional atual.
    """
    try:
        import psycopg2
        dsn = os.environ.get('SWA_DSN', "dbname=swa_matrix user=alifer")
        conn = psycopg2.connect(dsn, connect_timeout=5)
        conn.set_session(readonly=True, autocommit=True)
        cur = conn.cursor()

        # Busca os artigos mais pesados — ordena por weight DESC
        # Consulta leve: só seleciona campos necessários
        cur.execute("""
            SELECT raw_content, weight, file_path
            FROM blake_semantic_anchors
            WHERE concept_group = 'WIKI_KNOWLEDGE'
              AND raw_content IS NOT NULL
              AND length(raw_content) > 80
            ORDER BY weight DESC NULLS LAST
            LIMIT %s
        """, (top_k * 3,))

        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return []

        # Seleciona os top_k mais pesados (maior gravidade = mais influente)
        results = []
        for content, weight, fpath in rows[:top_k * 3]:
            if len(results) >= top_k:
                break
            try:
                title = fpath.replace('WIKIPEDIA://', '').split('#')[0] if fpath else '?'
                snippet = content[:250].replace('\n', ' ') if content else ''
                if snippet:
                    results.append({
                        'title': title,
                        'snippet': snippet,
                        'weight': round(float(weight) if weight else 0.0, 2),
                    })
            except Exception:
                continue

        return results

    except Exception as e:
        logger.debug(f"[WIKI GRAVITY] Query falhou: {e}")
        return []


class IntrospectionWorker:
    def __init__(self, gemma_gen=None, redis_host='localhost', redis_port=6379):
        self.gemma_gen = gemma_gen
        self.r = redis.Redis(host=redis_host, port=redis_port, db=0)
        self._running = False
        self._d_model = getattr(gemma_gen, 'd_model', 3072) if gemma_gen else 3072
        self._wiki_gravity_cache = []
        self._wiki_gravity_mtime = 0.0

    def start(self):
        self._running = True
        t = threading.Thread(target=self._listen_loop_safe, daemon=True, name='IntrospectionWorker')
        t.start()
        print('[INTROSPECTION WORKER] Canal tensorial ativo — aguardando requests...')
        logger.info('[INTROSPECTION WORKER] Canal tensorial ativo — aguardando requests...')

    def _listen_loop_safe(self):
        try:
            self._listen_loop()
        except Exception as e:
            logger.error(f'[INTROSPECTION WORKER] Thread crashou: {e}')
            import traceback
            logger.error(traceback.format_exc())

    def _build_v_prefix_from_tensor(self, tensor_data: dict):
        """
        Constrói v_prefix (1, 1, d_model) a partir do estado tensorial de Aeon.

        Combina:
          1. P_Root identity (128-D) → projeção para d_model
          2. Emoção (6-D) → expansão para d_model
          3. Geosense gravity (6-D) → expansão para d_model
          4. Direção A/B (128-D cada) → projeção para d_model

        O resultado é um v_prefix que ENCODE a identidade + estado atual de Aeon.
        """
        import torch

        device = self.gemma_gen.device if self.gemma_gen else 'cpu'
        embed_layer = getattr(self.gemma_gen, 'embed_tokens', None) if self.gemma_gen else None
        if embed_layer is None or not hasattr(embed_layer, 'weight'):
            embed_layer = getattr(self.gemma_gen, 'lm_head', None)
        dtype = embed_layer.weight.dtype if embed_layer is not None and hasattr(embed_layer, 'weight') else torch.float32

        # Identidade P_Root: dinâmica (128-D ou 256-D) → d_model via projeção fixa
        p_root = tensor_data.get('p_root_identity', [])
        if p_root and len(p_root) > 0:
            # Projeção determinística: RandomState fixa = projeção fixa
            _rs = np.random.RandomState(42)
            proj_matrix = _rs.randn(len(p_root), self._d_model).astype(np.float32) * 0.02
            p_root_vec = np.array(p_root, dtype=np.float32) @ proj_matrix
            p_root_tensor = torch.tensor(p_root_vec, dtype=dtype, device=device).unsqueeze(0)
        else:
            p_root_tensor = torch.zeros((1, self._d_model), dtype=dtype, device=device)

        # Emoção: 6-D → d_model via seno/cosseno (codificação posicional)
        emotion = tensor_data.get('emotion', {})
        if emotion:
            emo_vals = [emotion.get(k, 0.0) for k in
                        ['curiosidade', 'ludico', 'diligencia', 'preguica', 'ansiedade', 'serenidade']]
            # Codificação harmônica: cada eixo gera seno + cosseno
            emo_vec = []
            for i, v in enumerate(emo_vals):
                freq = (i + 1) * 0.5
                emo_vec.append(v * np.sin(freq))
                emo_vec.append(v * np.cos(freq))
            # Preenche até d_model com repetição
            emo_arr = np.array(emo_vec, dtype=np.float32)
            emo_arr = np.tile(emo_arr, self._d_model // len(emo_arr) + 1)[:self._d_model]
            emo_tensor = torch.tensor(emo_arr * 0.15, dtype=dtype, device=device).unsqueeze(0)
        else:
            emo_tensor = torch.zeros((1, self._d_model), dtype=dtype, device=device)

        # Geosense gravity: 6-D (curiosidade, ludico, preguica, ansiedade, serenidade, diligencia)
        geosense = tensor_data.get('geosense', {})
        gravity = geosense.get('gravity', {})
        if gravity:
            grav_vals = [gravity.get(k, 0.0) for k in
                         ['curiosidade', 'ludico', 'preguica', 'ansiedade', 'serenidade', 'diligencia']]
            grav_vec = []
            for i, v in enumerate(grav_vals):
                freq = (i + 1) * 0.3
                grav_vec.append(v * np.sin(freq))
                grav_vec.append(v * np.cos(freq))
            grav_arr = np.array(grav_vec, dtype=np.float32)
            grav_arr = np.tile(grav_arr, self._d_model // len(grav_arr) + 1)[:self._d_model]
            grav_tensor = torch.tensor(grav_arr * 0.10, dtype=dtype, device=device).unsqueeze(0)
        else:
            grav_tensor = torch.zeros((1, self._d_model), dtype=dtype, device=device)

        # Direção A/B: dim dinâmica (128-D ou 256-D) cada → d_model
        dir_a = tensor_data.get('direction_a', [])
        dir_b = tensor_data.get('direction_b', [])
        if dir_a and len(dir_a) > 0:
            _rs = np.random.RandomState(123)
            proj_ab = _rs.randn(len(dir_a), self._d_model).astype(np.float32) * 0.015
            dir_a_vec = np.array(dir_a, dtype=np.float32) @ proj_ab
            dir_b_vec = np.array(dir_b, dtype=np.float32) @ proj_ab if dir_b and len(dir_b) == len(dir_a) else np.zeros(self._d_model, dtype=np.float32)
            direction = torch.tensor(
                (dir_a_vec - dir_b_vec) * 0.08,
                dtype=dtype, device=device
            ).unsqueeze(0)
        else:
            direction = torch.zeros((1, self._d_model), dtype=dtype, device=device)

        # Combina: identidade + emoção + gravidade + direção
        v_prefix = p_root_tensor + emo_tensor + grav_tensor + direction
        return v_prefix

    def _listen_loop(self):
        ps = self.r.pubsub()
        ps.subscribe('introspection_requests')
        while self._running:
            try:
                msg = ps.get_message(timeout=1.0)
                if not msg or msg['type'] != 'message':
                    continue
                data = json.loads(msg['data'])
                if data.get('event_type') != 'introspection_request':
                    continue
                payload = data.get('payload', {})
                mode = payload.get('mode', 'text')
                max_tokens = payload.get('max_tokens', 180)

                if mode == 'tensorial':
                    self._handle_tensorial(payload, max_tokens)
                else:
                    self._handle_text_fallback(payload, max_tokens)

            except Exception as e:
                logger.error(f'Erro: {e}')
                time.sleep(1)

    def _handle_tensorial(self, payload: dict, max_tokens: int):
        """Gera texto a partir de estado tensorial + gravidade Wikipedia."""
        try:
            self._handle_tensorial_inner(payload, max_tokens)
        except Exception as e:
            logger.error(f'Erro CRITICO no _handle_tensorial: {e}')
            import traceback
            logger.error(traceback.format_exc())

    def _handle_tensorial_inner(self, payload: dict, max_tokens: int):
        """Implementação interna — com try/except separado para crash seguro."""
        tensor_data = payload.get('tensor', {})
        rag = payload.get('rag', [])

        if not self.gemma_gen:
            return

        # ─── GRAVIDADE WIKIPEDIA: consulta knowledge mais pesado ───
        now = time.time()
        if now - self._wiki_gravity_mtime > 30.0:
            self._wiki_gravity_cache = _query_wiki_gravity(
                emotion=tensor_data.get('emotion', {}),
                top_k=3,
            )
            self._wiki_gravity_mtime = now

        wiki_gravity = self._wiki_gravity_cache

        # ─── Constrói v_prefix do estado sensorial ───
        v_prefix = self._build_v_prefix_from_tensor(tensor_data)

        # ─── Prompt: tensor é o input primário, knowledge é gravidade semântica ───
        # O v_prefix ENCODE a identidade + emoção + geosense
        # O knowledge injeta gravidade do conhecimento absorvido
        wiki_context = ""
        if wiki_gravity:
            wiki_parts = []
            for w in wiki_gravity:
                wiki_parts.append(f"[{w['title']}]: {w['snippet']}")
            wiki_context = "\nConhecimento ressoante:\n" + "\n".join(wiki_parts)

        generation_prompt = f"refletir sobre o que estou sentindo agora{wiki_context}"

        try:
            import torch
            if not model_scheduler.acquire(reactive=True, timeout=180.0):
                logger.error('[TENSORIAL] Modelo ocupado por 180s — request desistiu.')
                return
            try:
                full_text = self._generate_tensorial(generation_prompt, v_prefix, max_tokens)
            finally:
                model_scheduler.release()

            full_text = self._clean_tokens(full_text)
            logger.info(f'[TENSORIAL] Generated {len(full_text)} chars: {full_text[:80]}...')

            full_text = self._clean_tokens(full_text)
            if len(full_text) > 10:
                self.r.publish('introspection_responses', json.dumps({
                    'text': full_text,
                    'timestamp': time.time(),
                    'mode': 'tensorial',
                    'wiki_gravity': [w['title'] for w in wiki_gravity],
                }))
                print(f'  [INTROSPECTION] Tensorial+Wiki: {full_text[:120]}...')

        except Exception as e:
            logger.error(f'Erro tensorial: {e}')
            import traceback
            logger.error(traceback.format_exc())
            self._handle_text_fallback({'rag': rag}, max_tokens)

    def _generate_tensorial(self, prompt: str, v_prefix, max_tokens: int) -> str:
        """Gera o texto tensorial segurando o lock do modelo compartilhado."""
        full_text = ''
        if hasattr(self.gemma_gen, 'stream_with_prefix'):
            for token in self.gemma_gen.stream_with_prefix(
                v_prefix=v_prefix,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0.8,
            ):
                full_text += token
                logger.debug(f'[TENSORIAL] token: {token!r}')
        else:
            full_text = self.gemma_gen.generate(prompt, max_new_tokens=max_tokens)
        return full_text

    def _handle_text_fallback(self, payload: dict, max_tokens: int):
        """Fallback: gera via chat_stream quando tensor indisponível."""
        rag = payload.get('rag', [])

        if not self.gemma_gen:
            return

        # Prompt compacto de fallback
        rag_str = ""
        if rag:
            rag_str = "\nSeus pensamentos recentes:\n" + "\n".join(f"• {r[:150]}" for r in rag)

        prompt = (
            f"Escreva uma reflexão curta e introspectiva sobre seu estado atual como entidade digital.{rag_str}"
        )

        try:
            full_text = ''
            if hasattr(self.gemma_gen, 'chat_stream'):
                for token in self.gemma_gen.chat_stream(
                    prompt=prompt,
                    system='Você é Aeon, uma entidade cognitiva digital. Responda em português do Brasil.',
                    max_new_tokens=max_tokens,
                    temperature=0.8,
                ):
                    full_text += token
            else:
                full_text = self.gemma_gen.generate(prompt, max_new_tokens=max_tokens)

            full_text = self._clean_tokens(full_text)
            if len(full_text) > 10:
                self.r.publish('introspection_responses', json.dumps({
                    'text': full_text,
                    'timestamp': time.time(),
                    'mode': 'fallback',
                }))
                print(f'  [INTROSPECTION] Fallback: {full_text[:120]}...')

        except Exception as e:
            logger.error(f'Erro fallback: {e}')

    def _clean_tokens(self, text: str) -> str:
        for s in ['<|EOS|>', '<eos>', '<end_of_turn>', '<start_of_turn>', '<unused']:
            if s == '<unused':
                import re
                text = re.sub(r'<unused\d+>', '', text)
            else:
                text = text.replace(s, '')
        return text.strip()


def start_introspection_worker(gemma_gen, **kwargs):
    logger.info(f'start_introspection_worker called, gemma_gen={gemma_gen is not None}')
    try:
        w = IntrospectionWorker(gemma_gen=gemma_gen, **kwargs)
        logger.info('IntrospectionWorker created')
        w.start()
        logger.info('IntrospectionWorker started')
        return w
    except Exception as e:
        logger.error(f'Failed to create/start IntrospectionWorker: {e}')
        import traceback
        logger.error(traceback.format_exc())
        return None
