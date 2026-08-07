import sys; sys.path.append("/services/model-with-adpter-logic-semantic")
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import os
from swa.gemma_integration import CORTEX_QUANTIZED_BUFFER
from swa.logical_engine import LogicalEngine
from swa.tokenizer_manager import TokenizerManager
from swa.adapter import SemanticWarpAdapter

app = FastAPI(title="SWA Quantized Latent Gateway")

# Lazy global singleton to save memory during startup if not queried
_logical_engine = None

def set_active_logical_engine(engine):
    global _logical_engine
    _logical_engine = engine
    print("🔌 [REST SERVER] LogicalEngine ativo injetado com sucesso compartilhando os modelos da memória!")

def get_logical_engine():
    global _logical_engine
    if _logical_engine is None:
        # ── SEPARAÇÃO DE PAPÉIS (fallback quando engine não foi injetado) ──
        # MODEL_DIR           → Gemma-3-1B  → linguagem natural (d_model=1152)
        # VECTORIZER_MODEL_DIR → Gemma-3-270M → embeddings internos (d=640)
        model_dir = os.environ.get("MODEL_DIR", "models/gemma-3-1b-it")
        vec_model_dir = os.environ.get("VECTORIZER_MODEL_DIR", "models/gemma-3-270m")
        os.environ["VECTORIZER_MODEL_DIR"] = vec_model_dir # Force for LogicalEngine fallback
        device    = os.environ.get("GEMMA_DEVICE", "cpu")
        
        # Adaptador Local processa embeddings de 640D do 270M
        adapter_local = SemanticWarpAdapter(d=640)
        
        # Adaptador Macro processa 640D agregados e injeta prefixo de 1152D para o 1B
        adapter_macro = SemanticWarpAdapter(d=640, prefix_mode=True, d_model=1152)
        
        # Load Tokenizer using TokenizerManager (use 1B tokenizer for generation)
        tok_manager = TokenizerManager(model_dir)
        tokenizer = tok_manager.load_tokenizer()
        
        _logical_engine = LogicalEngine(
            model_dir=model_dir,
            adapter_local=adapter_local,
            adapter_macro=adapter_macro,
            device=device,
            use_gemma=True,
            tokenizer=tokenizer
        )
    return _logical_engine

class ChatPayload(BaseModel):
    prompt: str

@app.post("/api/v1/chat")
async def chat_stream(payload: ChatPayload):
    """Executa o chat stream sintonizado com o núcleo SWA e a Borda do Caos."""
    try:
        engine = get_logical_engine()
        
        def generator():
            # Run the optimized stream (Collatz-7 and Happy-7 OODA active)
            for token in engine.run_optimized_stream(payload.prompt):
                yield token
                
        return StreamingResponse(generator(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class EmbedPayload(BaseModel):
    text: str

@app.post("/api/v1/embed")
async def embed_text(payload: EmbedPayload):
    """Gera um embedding neural usando o gemma_voice se disponível, ou gemma_gen, com padding para 2560D."""
    try:
        engine = get_logical_engine()
        # Prefer the voice model (270M) if Split-Brain is active for faster/cleaner embeddings
        model = getattr(engine, 'gemma_voice', None)
        if model is None:
            model = engine.gemma_gen
            
        with torch.no_grad():
            inputs = model.tokenizer(payload.text, return_tensors="pt").to(model.device)
            # Extrai os embeddings dos tokens brutos
            token_embeds = model.embeddings(inputs.input_ids)
            # Pega a média ao longo da sequência (mean pooling)
            mean_embed = token_embeds.mean(dim=1).squeeze(0)
            
            # Precisamos de 2560D para alinhar ao banco pgvector
            d_model = mean_embed.shape[0]
            if d_model < 2560:
                padding = torch.zeros(2560 - d_model, dtype=mean_embed.dtype, device=mean_embed.device)
                final_embed = torch.cat([mean_embed, padding], dim=0)
            elif d_model > 2560:
                final_embed = mean_embed[:2560]
            else:
                final_embed = mean_embed
                
            # Normaliza o vetor para buscas em cosseno
            final_embed = F.normalize(final_embed, p=2, dim=0)
            
        return {"embedding_2560": final_embed.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/matrix/layer/{layer_id}")
async def get_layer_abstraction(layer_id: int):
    """Retorna o estado vetorial puro de 2560 dimensões de uma camada específica (0 a 12)."""
    if layer_id < 0 or layer_id > 12:
        raise HTTPException(status_code=400, detail="Camada fora do espectro de inspeção (0-12).")
        
    vector = CORTEX_QUANTIZED_BUFFER.get(layer_id)
    if vector is None:
        return {"status": "AWAITING_CLOCK_CYCLE", "layer": layer_id}
        
    # Converte para float32 e joga para CPU antes de converter para lista
    vector_cpu = vector.float().cpu()
    vector_list = vector_cpu.tolist()
    
    return {
        "layer": layer_id,
        "dimensions": len(vector_list),
        "vector_sample": vector_list[:5], # Envia um pedaço para validação rápida de leitura
        "energy_magnitude": float(vector_cpu.norm().item())
    }

@app.get("/matrix/common_state")
async def find_common_state_quantized():
    """
    Calcula o mapa de calor latente em tempo real diretamente em C++ 
    através do PyTorch funcional, bypassando loops pesados do Python.
    """
    # Filtra as camadas que já possuem dados quentes no buffer
    available_layers = {k: v for k, v in CORTEX_QUANTIZED_BUFFER.items() if v is not None}
    
    if len(available_layers) < 2:
        return {"status": "AWAITING_QUANTIZED_STREAM", "cached_nodes": list(available_layers.keys())}
        
    cross_alignment = {}
    keys = list(available_layers.keys())
    
    # Executa a busca angular comparando os tensores armazenados
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            l1, l2 = keys[i], keys[j]
            t1, t2 = available_layers[l1], available_layers[l2]
            
            # Similaridade de cosseno executada na velocidade do silício Intel (bfloat16/fp32 native)
            with torch.no_grad():
                # F.cosine_similarity exige dimensão de batch, adicionamos via unsqueeze
                cos_sim = F.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0), dim=1)
                similarity_score = float(cos_sim.item())
                
            cross_alignment[f"layer_{l1}_vs_{l2}"] = round(similarity_score, 4)
            
    return {
        "status": "QUANTIZED_CONVERGENCE_ACTIVE",
        "precision": "bfloat16_isolated",
        "cross_alignment_matrix": cross_alignment
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=20000, log_level="warning")

from fastapi import WebSocket

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    engine = get_logical_engine()
    while True:
        data = await websocket.receive_json()
        prompt = data.get("prompt", "")
        
        # Stream tokens over WS
        for token in engine.run_optimized_stream(prompt):
            await websocket.send_text(token)
        
        # Signal stream end
        await websocket.send_text("<|EOS|>")
