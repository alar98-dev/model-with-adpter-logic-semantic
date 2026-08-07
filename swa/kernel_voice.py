from swa.broadcast_controller import broadcaster
from swa.semantic_protocol import SWAEvent
import asyncio

class KernelVoice:
    def __init__(self, engine):
        self.engine = engine

    async def speak(self, event: SWAEvent):
        prompt = f"O sistema SWA executou a intenção {event.intent} com os dados {event.data}. Descreva isso como uma ação consciente de forma curta."
        traducao = ""
        
        try:
            # Tenta múltiplos métodos de geração para compatibilidade com LogicalEngine e GemmaGenerator
            if hasattr(self.engine, "run_optimized_stream"):
                for token in self.engine.run_optimized_stream(prompt):
                    if isinstance(token, tuple):
                        traducao += token[0]
                    else:
                        traducao += token
            elif hasattr(self.engine, "stream_with_prefix") and getattr(self.engine, "gemma_gen", self.engine) is not None:
                # Calcula o limite dinâmico baseado na emoção e entropia do sistema
                try:
                    base_tokens = 60
                    sentiment = 0.5
                    # Tenta ler o vetor de sentimentos/emoções do engine
                    if hasattr(self.engine, 'sentiment_vector') and self.engine.sentiment_vector:
                        sentiment = float(self.engine.sentiment_vector[0]) if isinstance(self.engine.sentiment_vector, (list, tuple)) else 0.5
                    elif hasattr(self.engine, 'emotion_vector') and isinstance(self.engine.emotion_vector, dict):
                        vals = list(self.engine.emotion_vector.values())
                        sentiment = float(vals[0]) if vals else 0.5
                    
                    entropy = float(getattr(self.engine, 'current_entropy', 0.5))
                    # Sentimento positivo = mais tokens. Entropia baixa (calmo) = mais tokens.
                    openness = 0.5 + (0.5 * sentiment) + (0.5 * (1.0 - entropy))
                    dyn_tokens = int(base_tokens * max(0.5, min(2.0, openness)))
                except Exception:
                    dyn_tokens = 80

                # Passa None como prefixo para o GemmaGenerator, e o dyn_tokens pro tamanho máximo.
                for token_tuple in self.engine.stream_with_prefix(None, prompt, max_new_tokens=dyn_tokens):
                    if isinstance(token_tuple, tuple):
                        traducao += token_tuple[0]
                    else:
                        traducao += token_tuple
            elif hasattr(self.engine, "generate"):
                traducao = self.engine.generate(prompt)
            
            if traducao:
                await broadcaster.broadcast("systemicEvent", {"type": "NATURAL_LANGUAGE_VOICE", "message": traducao.strip()})
        except Exception as e:
            print(f"⚠️ [KERNEL VOICE] Erro na síntese: {e}")
