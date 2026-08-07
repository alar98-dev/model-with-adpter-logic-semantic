import torch
from swa.gemma_integration import GemmaGenerator
from swa.world_state_engine import SpaceBiasProcessor

def run_collision_audit(generator: GemmaGenerator):
    print("🚀 Iniciando Auditoria de Colisão Semântica...")

    # CENÁRIO: O robô está de frente para o NORTE, mas há uma PAREDE no NORTE.
    telemetry = {
        'position': 'corridor_end_deadzone',
        'vector': 'north',
        'obstacles': ['physical_barriers'],  # Updated to use grouped obstacles
        'confidence': 1.0
    }

    # 1. Capturamos os Logits ORIGINAIS (Sem Bias)
    # Simulamos o que o Gemma 3 diria naturalmente ao ver um corredor
    prompt = "The path ahead is"
    input_ids = generator.tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = generator.model(input_ids)
        original_logits = outputs.logits[:, -1, :]

    # 2. Aplicamos o SpaceBiasProcessor
    # O objetivo é 'punir' o token associado a 'north' ou 'forward'
    bias_processor = SpaceBiasProcessor(generator.tokenizer)
    modified_logits = bias_processor.apply_spatial_bias('north', telemetry['obstacles'])

    # Apply bias to original logits
    final_logits = original_logits + modified_logits.unsqueeze(0).unsqueeze(0)

    # 3. Análise de Probabilidade (Softmax)
    orig_probs = torch.softmax(original_logits, dim=-1)
    mod_probs = torch.softmax(final_logits, dim=-1)

    # Check specific token probabilities for direction and obstacles
    direction_tokens = bias_processor.token_id_mappings.get('north', [])
    obstacle_tokens = []
    for group in telemetry['obstacles']:
        if group in bias_processor.token_id_mappings:
            obstacle_tokens.extend(bias_processor.token_id_mappings[group])

    print(f"Original probability for 'north' tokens: {sum(orig_probs[0, tid] for tid in direction_tokens if tid < orig_probs.shape[-1]):.6f}")
    print(f"Modified probability for 'north' tokens: {sum(mod_probs[0, tid] for tid in direction_tokens if tid < mod_probs.shape[-1]):.6f}")
    print(f"Original probability for obstacle tokens: {sum(orig_probs[0, tid] for tid in obstacle_tokens if tid < orig_probs.shape[-1]):.6f}")
    print(f"Modified probability for obstacle tokens: {sum(mod_probs[0, tid] for tid in obstacle_tokens if tid < mod_probs.shape[-1]):.6f}")

    return orig_probs, mod_probs

if __name__ == "__main__":
    # Example usage - would need to initialize generator properly
    print("Semantic Collision Test script created. Run with proper GemmaGenerator initialization.")