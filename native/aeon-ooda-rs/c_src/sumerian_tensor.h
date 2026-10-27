#ifndef SUMERIAN_TENSOR_H
#define SUMERIAN_TENSOR_H

#include <stdint.h>
#include <stddef.h>

#define TENSOR_DIM 640
#define SUMERIAN_HOUSES 60
#define SUMERIAN_REALMS 6

#ifdef __cplusplus
extern "C" {
#endif

// Inicializa tabelas de senos e cossenos pre-calculadas na base 60 e atratores de reinos
void sumerian_init_tables(void);

// Produto Escalar Simbolico via AVX2 + FMA (640D)
// Utiliza 4 acumuladores YMM paralelos com desenrolamento de laco para saturar o pipeline
float sumerian_avx2_dot_product(const float* a, const float* b);

// Normalizacao RMSNorm vetorizada com precisao SIMD
void sumerian_avx2_rmsnorm(float* vec, float eps);

// Rotacao Geometrica Sumeriana (Givens Symplectic Rotation na Base 60)
// Aplica rotacao de fase theta_h = 2*PI*(house-1)/60 diretamente nos pares de coordenadas
void sumerian_avx2_rotate_phase(float* vec, uint32_t house_id, float delta_q);

// Transicao de Estado Completa no Espaco Tensorial Sumeriano
// 1. Projecao no Atrator do Reino via AVX2
// 2. Rotacao de Fase da Casa Sumeriana
// 3. Injecao de Tau e Amortecimento de Entropia DeltaQ
// 4. RMSNorm final
float sumerian_tensor_forward_step(
    float* state,
    uint32_t house_id,
    uint32_t realm_id,
    float tau,
    float delta_q
);

// Atualizacao Hebbiana do Lar Vetorial em Hardware AVX2 + FMA
// centroid = centroid + lr * (current - centroid)
void sumerian_lair_hebbian_update(float* centroid, const float* current, float lr);

// Afinidade e Sensacao de Pertencimento no Lar Vetorial
float sumerian_lair_affinity(const float* current, const float* centroid);

// Projecao Subcortex -> Cortex Gemma (640D -> 128D -> 1152D) via AVX2 / FMA
// Produto Escalar Empacotado SIMD INT8 x UINT8 via AVX2 (_mm256_maddubs_epi16 + _mm256_madd_epi16)
// Satura os 256 bits com 32 multiplicacoes e 16 somas por ciclo em pistas de 16 bits
float sumerian_avx2_dot_product_packed_i8(
    const uint8_t* u8_act,
    const int8_t* i8_weight,
    int length,
    float scale_act,
    float scale_weight
);

// Quantizacao rapida SIMD F32 -> Int8 simetrico
void sumerian_avx2_quantize_f32_to_i8(const float* src, int8_t* dst, int length, float* out_scale);

// Quantizacao rapida F32 -> Uint8
void sumerian_avx2_quantize_f32_to_u8(const float* src, uint8_t* dst, int length, float* out_scale, float* out_min);

void sumerian_cortex_projection_step(
    const float* input_640,
    const float* w_a_matrix,
    const float* w_a_bias,
    const float* prefix_proj_matrix,
    const float* prefix_proj_bias,
    const float* v_instinct,
    const float* v_teo,
    float beta,
    float gamma,
    float* output_1152
);

#ifdef __cplusplus
}
#endif

#endif // SUMERIAN_TENSOR_H

// Conversao nativa de alta velocidade F32 <-> FP16 via F16C (1 registrador XMM de 128b por ciclo)
void export_prefix_fp16(const float* src_1152, uint16_t* dst_1152);
void import_prefix_fp16(const uint16_t* src_1152, float* dst_1152);

