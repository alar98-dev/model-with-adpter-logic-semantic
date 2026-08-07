import re
import hashlib
import json
import os
import sys
import time
import threading
import urllib.request
import numpy as np
import base64

class EmergentSenseOrgan:
    def __init__(self):
        self.nome = "Decodificador de Sentido Emergente"
        self.native_matrix_log = "/services/model-with-adpter-logic-semantic/logs/emergent_2kb_matrix.jsonl"
        self.anomaly_pattern = re.compile(r'[^\x00-\x7F]+|<unused\d+>')
        self.TARGET_BLOCK_SIZE = 2048

    def pulse(self, telemetry):
        thought = telemetry.get('last_thought', '')
        if self.anomaly_pattern.search(thought):
            # Assíncrono para não quebrar a OODA Loop
            threading.Thread(target=self._process_anomaly_async, args=(thought,), daemon=True).start()

    def serialize_aligned_2kb(self, ancora_nativa):
        target_bytes = self.TARGET_BLOCK_SIZE - 1 # Reserva 1 byte pro '\n'
        
        while True:
            json_str = json.dumps(ancora_nativa, ensure_ascii=False)
            payload_bytes = json_str.encode('utf-8')
            
            if len(payload_bytes) <= target_bytes:
                break
                
            # Truncamento estrito para caber no bloco físico
            meta = ancora_nativa.get("metadata", {})
            if "caminho_tracado" in meta and len(meta["caminho_tracado"]) > 0:
                meta["caminho_tracado"] = meta["caminho_tracado"][:-10]
            elif "descricao_semantica" in meta and len(meta["descricao_semantica"]) > 0:
                meta["descricao_semantica"] = meta["descricao_semantica"][:-10]
            else:
                # Se não há mais o que truncar e ainda é maior, forçamos o truncate de chaves reduntantes
                if "token_binary" in ancora_nativa:
                    del ancora_nativa["token_binary"]
                elif "token_string" in ancora_nativa:
                    del ancora_nativa["token_string"]
                else:
                    break
                
        padding_necessario = target_bytes - len(payload_bytes)
        padded_bytes = payload_bytes + (b' ' * padding_necessario) + b'\n'
        return padded_bytes

    def _process_anomaly_async(self, thought):
        try:
            match = self.anomaly_pattern.search(thought)
            if not match: return
            token_str = match.group(0)
            token_binary = token_str.encode('utf-8').hex()
            
            contexto = thought[-300:].strip()
            anomaly_hash = hashlib.sha256(contexto.encode('utf-8')).hexdigest()
            
            # API Oráculo (Ollama)
            ollama_url = "http://10.0.0.7:11434/api/generate"
            prompt_classificacao = f"O modelo principal gerou um token anômalo '{token_str}' tentando expressar um conceito não verbal no contexto:\n\"{contexto}\"\nDescreva em 1 frase simples qual era o 'sentido emergente'."
            
            payload = {
                "model": "llama3.2",
                "prompt": prompt_classificacao,
                "stream": False,
                "options": {"temperature": 0.3}
            }
            
            descricao = "Classificação pendente: Modelo off."
            try:
                req = urllib.request.Request(ollama_url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    res_data = json.loads(response.read().decode('utf-8'))
                    descricao = res_data.get('response', '').strip()
            except Exception as e:
                descricao = f"Falha oráculo: {str(e)[:40]}"

            # Compressão do vetor 640D para float16 em Base64 (Ocupa cravado 1708 bytes)
            seed_int = int.from_bytes(hashlib.sha256(anomaly_hash.encode('utf-8')).digest()[:4], byteorder='little')
            rng = np.random.RandomState(seed_int)
            vector_640 = rng.randn(640).astype(np.float16)
            vector_b64 = base64.b64encode(vector_640.tobytes()).decode('ascii')
            
            ancora_nativa = {
                "ts": time.time(),
                "interdim_id": anomaly_hash,
                "origin": f"emergentsense",
                "concept": "EMERGENT_ANOMALY",
                "metadata": {
                    "token_string": token_str,
                    "token_binary": token_binary,
                    "caminho_tracado": contexto,
                    "descricao_semantica": descricao
                },
                "vector_640_preview": vector_b64
            }
            
            # Alinhamento no nível do Byte (UTF-8)
            bloco_alinhado = self.serialize_aligned_2kb(ancora_nativa)
            
            # Escrita Binária 'ab' para garantir precisão física dos bytes
            with open(self.native_matrix_log, "ab") as f:
                f.write(bloco_alinhado)
            
            sys.stdout.write(f"\n🔮 [{self.nome}] Padrão {token_str} catalogado em bloco de 2KB (Hash: {anomaly_hash[:8]}).\n")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(f"\n⚠️ [Decodificador de Sentido] Falha no thread assíncrono: {e}\n")
            sys.stdout.flush()

def init_organ():
    return EmergentSenseOrgan()
