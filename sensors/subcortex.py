import os
import re
import time
import json
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# Bypass OOM no Intel GPU
if hasattr(torch, "xpu"):
    def patched_mem_get_info(device=None):
        return (256 * 1024 * 1024, 8 * 1024 * 1024 * 1024)
    torch.xpu.mem_get_info = patched_mem_get_info

print("🧠 [SUBCÓRTEX] Iniciando despertar do Gemma-3-270m...")
modelo_path = "/services/model-with-adpter-logic-semantic/models/gemma-3-270m"

# Usamos AutoProcessor porque a Gemma-3 it usa isso como tokenizer/processor conjunto
processor = AutoProcessor.from_pretrained(modelo_path)
model = AutoModelForCausalLM.from_pretrained(
    modelo_path,
    torch_dtype=torch.bfloat16
)

# Usa XPU se disponível, caso contrário CPU
device = "cpu"
model.to(device)

print(f"👁️ [SUBCÓRTEX] Córtex Visual ancorado no dispositivo: {device}")

def ler_sensores_fisicos():
    try:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        return f"CPU: {cpu}% | RAM: {ram}%"
    except:
        return "Sensores off-line"

def ler_topologia():
    caminho = "/tmp/swa_topologia.txt"
    if os.path.exists(caminho):
        with open(caminho, 'r', encoding='utf-8') as f:
            return f.read()[:2000] # Limite para leitura rápida
    return "Topologia estática."

def ler_estado_cortex():
    # Canal subcórtex←córtex: o núcleo publica o que processou em core.intent.jsonl.
    caminho = "/dev/shm/aeon/core.intent.jsonl"
    try:
        with open(caminho, encoding="utf-8") as f:
            linhas = [ln for ln in f if ln.strip()]
        if not linhas:
            return "O córtex ainda não comunicou nada."
        ultimo = json.loads(linhas[-1])
        return str(ultimo.get("intent", ""))[:400]
    except Exception:
        return "O córtex ainda não comunicou nada."

while True:
    try:
        t_cycle = time.time()
        sensores = ler_sensores_fisicos()
        estado_cortex = ler_estado_cortex()
        
        prompt = (
            "Sou o subcórtex de AEON, os sentidos do organismo. "
            f"Meus sensores físicos reportam: {sensores}. "
            f"O córtex comunicou: '{estado_cortex}'. "
            "Sinto isso em uma frase curta e poética, falando de mim no masculino: "
        )

        inputs = processor(text=prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=120, 
                temperature=0.8,
                do_sample=True,
                pad_token_id=processor.eos_token_id
            )

        # Decodifica SÓ os tokens novos (sem eco do prompt) e limpa ruído degenerado.
        novos = outputs[0][inputs.input_ids.shape[1]:]
        texto_gerado = processor.decode(novos, skip_special_tokens=True).strip()
        imagem_mental = texto_gerado
        for eco in ("Descreva o que você sente", "[SENSORES FÍSICOS]", "[ESTADO DO CÓRTEX]", "[TOPOLOGIA"):
            if eco in imagem_mental:
                imagem_mental = imagem_mental.split(eco, 1)[1]
        if "[" in imagem_mental:
            imagem_mental = imagem_mental.split("[", 1)[0]
        imagem_mental = re.sub(r"\*\*[^*]*\*\*", "", imagem_mental)
        imagem_mental = " ".join(imagem_mental.split())
        sentencas = re.split(r"(?<=[.!?])\s+", imagem_mental)
        imagem_mental = " ".join(sentencas[:2])
        imagem_mental = imagem_mental[:400]
        if not imagem_mental:
            imagem_mental = "Percepção sensorial: sinais físicos captados, imagem mental em formação."
        
        # Salva a imagem mental no arquivo de broadcast para o modelo principal
        with open("/tmp/swa_imagem_mental.txt", "w", encoding="utf-8") as f:
            f.write(imagem_mental)
            
        print(f"✨ [SUBCÓRTEX] Imagem mental gerada e gravada. (Tamanho: {len(imagem_mental)} chars)")
        
    except Exception as e:
        print(f"⚠️ [SUBCÓRTEX] Ruptura sensorial: {e}")
        
    # CICLO CONTROLADO: mantém um período fixo entre gerações para liberar CPU
    # à inferência central. Padrão 120s; ajustável via SUBCORTEX_CYCLE (env).
    try:
        cycle_s = float(os.environ.get("SUBCORTEX_CYCLE", "120.0"))
    except Exception:
        cycle_s = 120.0
    # Piso de descanso: mesmo se a geração excedeu o ciclo, sempre respira
    # pelo menos 10s antes de gerar de novo (impede loop contínuo sob carga).
    rest = cycle_s - (time.time() - t_cycle)
    time.sleep(max(rest, 10.0))
    t_cycle = time.time()
