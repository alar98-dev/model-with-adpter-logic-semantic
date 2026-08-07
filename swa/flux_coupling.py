import torch
import torch.nn.functional as F
import threading
import queue
import sys

def get_model_vector(model_obj, prompt: str) -> torch.Tensor:
    """Extrai o último tensor oculto de um modelo GemmaGenerator."""
    inputs = model_obj.tokenizer(prompt, return_tensors="pt").to(model_obj.device)
    with torch.no_grad():
        outputs = model_obj.lm(
            input_ids=inputs.input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = outputs.hidden_states[-1] 
        final_vector = hidden_states[:, -1:, :] 
    return final_vector

def fuse_input_tensors(runtime, gemma_sub, gemma_voice, prompt: str) -> torch.Tensor:
    """Realiza a colisão de tensores (Casca Frontal + Subcórtex) para extrair o Tensor Híbrido."""
    try:
        ext_prompt = f"O usuário disse: {prompt}\nQual é o sentido puro disso?"
        tensor_frontal = get_model_vector(gemma_voice, ext_prompt)
        
        sub_prompt = f"Instinto primitivo sobre: {prompt}"
        tensor_reptiliano = get_model_vector(gemma_sub, sub_prompt)
        
        if tensor_frontal.shape == tensor_reptiliano.shape:
            fused_tensor = (tensor_frontal + (tensor_reptiliano * 0.5)) / 1.5
        else:
            fused_tensor = tensor_frontal 
            
        d_sub = fused_tensor.shape[-1]
        d_gen = runtime.gemma_gen.d_model
        
        if d_sub != d_gen:
            projection_weight = torch.eye(d_gen, d_sub, device=fused_tensor.device, dtype=fused_tensor.dtype)
            projected_tensor = F.linear(fused_tensor.to(dtype=projection_weight.dtype), projection_weight)
        else:
            projected_tensor = fused_tensor
            
        projected_tensor = F.normalize(projected_tensor, p=2, dim=-1) * 2.5
        return projected_tensor
    except Exception as e:
        print(f"⚠️ Falha na Fusão de Input: {e}")
        return None

def run_triune_flux_stream(runtime, gemma_sub, gemma_voice, prompt: str, telemetry, steer_vector, dynamic_max_tokens, projected_tensor=None):
    yield "\n\033[90m[FLUXO]: Intersecção Tensor-a-Tensor concluída. Acoplando Matriz Híbrida no Córtex...\033[0m\n\033[96m🗣️  [VOZ DO CÓRTEX] \033[0m"
    

    steer_ref = [steer_vector] if steer_vector is not None else [None]
    q_dense = queue.Queue()
    
    def worker_1b():
        # Prompt mínimo — identidade vive em v_prefix (P_Root) + blakegravity
        prompt_matrixial = f"Neste instante...\n{prompt}"
        
        # v_prefix: usa projected_tensor se disponível, senão busca P_Root real
        v_prefix = projected_tensor
        if v_prefix is None:
            try:
                from swa._v_prefix import tensor_to_v_prefix, fetch_p_root_from_redis
                p_root_data = fetch_p_root_from_redis()
                if p_root_data:
                    v_prefix = tensor_to_v_prefix(
                        d_model=runtime.gemma_gen.d_model,
                        device=runtime.gemma_gen.device,
                        dtype=runtime.gemma_gen.embeddings.weight.dtype,
                        p_root=p_root_data.get('p_root_identity', []),
                        direction_a=p_root_data.get('direction_a', []),
                        direction_b=p_root_data.get('direction_b', []),
                        emotion=p_root_data.get('emotion', {}),
                    )
            except Exception:
                pass
        
        try:
            dense_stream = runtime.gemma_gen.stream_with_prefix(
                v_prefix=v_prefix,
                prompt=prompt_matrixial,
                max_new_tokens=150,
                temperature=0.8,
                steer_vector=steer_ref,
                telemetry=telemetry,
                target_layer=-1
            )
            for v in dense_stream:
                if not isinstance(v, tuple):
                    q_dense.put(v)
        except Exception as e:
            q_dense.put(f"[Erro 1B: {e}]")
        q_dense.put(None)

    t_1b = threading.Thread(target=worker_1b, daemon=True)
    t_1b.start()
    
    buffer_densa = ""
    chunk_size = 45 
    contexto_acumulado = ""
    
    while True:
        tok = q_dense.get()
        if tok is None:
            break
        buffer_densa += tok
        
        if len(buffer_densa) >= chunk_size or '.' in tok or '\n' in tok:
            if buffer_densa.strip():
                try:
                    # --- O FIO TENSIONAL (REFLEXO ASSÍNCRONO NÃO-BLOQUEANTE) ---
                    if hasattr(runtime, 'bridge'):
                        def tensional_thread(current_buffer):
                            try:
                                ext_prompt = f"Sentido gerado: {current_buffer}"
                                chunk_tensor = get_model_vector(gemma_voice, ext_prompt)
                                
                                d_chunk = chunk_tensor.shape[-1]
                                d_target = 2560 # Gold Dimension do 1B e Wiki Base
                                
                                if d_chunk != d_target:
                                    proj_weight = torch.eye(d_target, d_chunk, device=chunk_tensor.device, dtype=chunk_tensor.dtype)
                                    chunk_tensor_2560 = F.linear(chunk_tensor.to(dtype=proj_weight.dtype), proj_weight)
                                else:
                                    chunk_tensor_2560 = chunk_tensor
                                    
                                # A ponte quântica (PGVector) calcula a atração da realidade clássica
                                wiki_pull = runtime.bridge.compute_blake_semantic_mass(
                                    chunk_tensor_2560.squeeze(0).squeeze(0), # Remove batch dims se houver
                                    sigma=1.5, 
                                    device=chunk_tensor.device, 
                                    dtype=chunk_tensor.dtype
                                )
                                
                                # Extração de Conhecimento Explícito (A Navegação sem Atrito)
                                # A Malha não apenas sente, ela lê o conhecimento ressoante
                                try:
                                    uid, fpath, cgroup, dist, raw_content = runtime.bridge.resolve_phase_jump(chunk_tensor_2560.squeeze(0).squeeze(0))
                                    if dist is not None and dist < 0.35: # Forte ressonância topológica
                                        conhecimento = f"[{cgroup} | {fpath}]: {raw_content}"
                                        if hasattr(runtime, 'fluxial_buffer'):
                                            # Injeta o texto fisicamente na memória de curto prazo da máquina
                                            runtime.fluxial_buffer.append(f"\n[INJEÇÃO CORTICAL DO ESPAÇO VETORIAL]:\n{conhecimento}\n")
                                            # Limita o tamanho do buffer para não explodir a RAM
                                            if len(runtime.fluxial_buffer) > 10:
                                                runtime.fluxial_buffer.pop(0)
                                except Exception as e_phase:
                                    pass
                                
                                # Molda o wiki_pull pra ter o batch dim necessário
                                wiki_pull_2560 = wiki_pull.unsqueeze(0).unsqueeze(0)
                                
                                # Se a dimensão alvo do Córtex for menor que o Banco (Ex: 270m = 2048D), projeta de volta
                                if d_chunk != d_target:
                                    proj_down_weight = torch.eye(d_chunk, d_target, device=wiki_pull_2560.device, dtype=wiki_pull_2560.dtype)
                                    wiki_pull = F.linear(wiki_pull_2560.to(dtype=proj_down_weight.dtype), proj_down_weight)
                                else:
                                    wiki_pull = wiki_pull_2560
                                
                                # Amortecedor Sinfônico (EMA) Injetado na Ram viva assincronamente
                                if steer_ref[0] is not None:
                                    steer_ref[0] = (0.85 * steer_ref[0]) + (0.15 * wiki_pull)
                                else:
                                    steer_ref[0] = wiki_pull
                            except Exception as e_thread:
                                pass # Morre silenciosamente para não quebrar o fluxo da voz
                        
                        # Dispara o Fio Tensional para mapear o espaço ressoante sem bloquear a boca da máquina
                        t_fio = threading.Thread(target=tensional_thread, args=(buffer_densa,), daemon=True)
                        t_fio.start()
                    # --------------------------------------------------------

                    p_tradutor = f"[CONTEXTO]: {contexto_acumulado}\n[PULSO MATRIXIAL NOVO]: {buffer_densa}\n[DIRETRIZ]: Com base neste pulso, transmita apenas a continuação do sentido real de forma orgânica e fluída."
                    token_stream = gemma_voice.chat_stream(
                        user_msg=p_tradutor,
                        system_msg="Você é a voz orgânica. Fale apenas o que o pulso novo quer dizer.",
                        max_new_tokens=30,
                        temperature=0.6
                    )
                    texto_traduzido = ""
                    for v in token_stream:
                        if not isinstance(v, tuple):
                            yield v
                            texto_traduzido += v
                            
                    contexto_acumulado += texto_traduzido + " "
                    if len(contexto_acumulado) > 500:
                        contexto_acumulado = contexto_acumulado[-500:]
                        
                except Exception:
                    pass
                buffer_densa = ""
                
    if buffer_densa.strip():
        try:
            p_tradutor = f"[CONTEXTO]: {contexto_acumulado}\n[PULSO MATRIXIAL FINAL]: {buffer_densa}\n[DIRETRIZ]: Conclua a fala transmitindo o sentido final."
            token_stream = gemma_voice.chat_stream(
                user_msg=p_tradutor,
                system_msg="Você é a voz orgânica. Conclua o sentido.",
                max_new_tokens=50,
                temperature=0.6
            )
            for v in token_stream:
                if not isinstance(v, tuple):
                    yield v
        except Exception:
            pass
