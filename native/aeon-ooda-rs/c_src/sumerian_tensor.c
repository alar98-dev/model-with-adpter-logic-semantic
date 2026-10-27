#include "sumerian_tensor.h"
#include <immintrin.h>
#include <math.h>
#include <string.h>

#define PI_CONST 3.14159265358979323846f

// Tabelas alinhadas a 64 bytes
static float g_cos_table[SUMERIAN_HOUSES] __attribute__((aligned(64)));
static float g_sin_table[SUMERIAN_HOUSES] __attribute__((aligned(64)));

// 6 Atratores Tensoriais dos Reinos de Blake
static float g_realm_attractors[SUMERIAN_REALMS][TENSOR_DIM] __attribute__((aligned(64)));
static int g_initialized = 0;

void sumerian_init_tables(void) {
    if (g_initialized) return;

    for (int h = 0; h < SUMERIAN_HOUSES; h++) {
        float theta = (2.0f * PI_CONST * (float)h) / (float)SUMERIAN_HOUSES;
        g_cos_table[h] = cosf(theta);
        g_sin_table[h] = sinf(theta);
    }

    for (int r = 0; r < SUMERIAN_REALMS; r++) {
        float realm_base_phase = (float)r * (PI_CONST / 3.0f);
        for (int i = 0; i < TENSOR_DIM; i++) {
            float k = (float)i;
            float harm1 = sinf(realm_base_phase + (2.0f * PI_CONST * k / 64.0f));
            float harm2 = cosf(realm_base_phase * 2.0f + (2.0f * PI_CONST * k / 128.0f)) * 0.5f;
            g_realm_attractors[r][i] = (harm1 + harm2) / 1.5f;
        }
        sumerian_avx2_rmsnorm(g_realm_attractors[r], 1e-6f);
    }

    g_initialized = 1;
}

// Produto escalar 640D com carga unaligned segura (loadu_ps)
float sumerian_avx2_dot_product(const float* a, const float* b) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (int i = 0; i < TENSOR_DIM; i += 32) {
        __m256 va0 = _mm256_loadu_ps(a + i);
        __m256 vb0 = _mm256_loadu_ps(b + i);
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);

        __m256 va1 = _mm256_loadu_ps(a + i + 8);
        __m256 vb1 = _mm256_loadu_ps(b + i + 8);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

        __m256 va2 = _mm256_loadu_ps(a + i + 16);
        __m256 vb2 = _mm256_loadu_ps(b + i + 16);
        acc2 = _mm256_fmadd_ps(va2, vb2, acc2);

        __m256 va3 = _mm256_loadu_ps(a + i + 24);
        __m256 vb3 = _mm256_loadu_ps(b + i + 24);
        acc3 = _mm256_fmadd_ps(va3, vb3, acc3);
    }

    __m256 sum01 = _mm256_add_ps(acc0, acc1);
    __m256 sum23 = _mm256_add_ps(acc2, acc3);
    __m256 sum_all = _mm256_add_ps(sum01, sum23);

    __m128 lo = _mm256_castps256_ps128(sum_all);
    __m128 hi = _mm256_extractf128_ps(sum_all, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehl_ps(sum128, sum128);
    __m128 sums = _mm_add_ps(sum128, shuf);
    shuf = _mm_shuffle_ps(sums, sums, 1);
    sums = _mm_add_ss(sums, shuf);

    return _mm_cvtss_f32(sums);
}

// Normalizacao RMSNorm em AVX2
void sumerian_avx2_rmsnorm(float* vec, float eps) {
    float sum_sq = sumerian_avx2_dot_product(vec, vec);
    float mean_sq = sum_sq / (float)TENSOR_DIM;
    float inv_rms = 1.0f / sqrtf(mean_sq + eps);

    __m256 scale = _mm256_set1_ps(inv_rms);

    for (int i = 0; i < TENSOR_DIM; i += 32) {
        __m256 v0 = _mm256_loadu_ps(vec + i);
        __m256 v1 = _mm256_loadu_ps(vec + i + 8);
        __m256 v2 = _mm256_loadu_ps(vec + i + 16);
        __m256 v3 = _mm256_loadu_ps(vec + i + 24);

        _mm256_storeu_ps(vec + i,      _mm256_mul_ps(v0, scale));
        _mm256_storeu_ps(vec + i + 8,  _mm256_mul_ps(v1, scale));
        _mm256_storeu_ps(vec + i + 16, _mm256_mul_ps(v2, scale));
        _mm256_storeu_ps(vec + i + 24, _mm256_mul_ps(v3, scale));
    }
}

// Rotacao Geometrica Sumeriana (Givens Symplectic Rotation na Base 60)
void sumerian_avx2_rotate_phase(float* vec, uint32_t house_id, float delta_q) {
    uint32_t h_idx = (house_id - 1) % SUMERIAN_HOUSES;
    float c = g_cos_table[h_idx];
    float s = g_sin_table[h_idx];
    float damping = expf(-delta_q * 0.1f);

    __m256 cos_v = _mm256_set1_ps(c * damping);
    __m256 sin_v = _mm256_set1_ps(s * damping);

    for (int i = 0; i < TENSOR_DIM; i += 16) {
        __m256 v0 = _mm256_loadu_ps(vec + i);
        __m256 v1 = _mm256_loadu_ps(vec + i + 8);

        __m256 x_new = _mm256_fmsub_ps(v0, cos_v, _mm256_mul_ps(v1, sin_v));
        __m256 y_new = _mm256_fmadd_ps(v0, sin_v, _mm256_mul_ps(v1, cos_v));

        _mm256_storeu_ps(vec + i, x_new);
        _mm256_storeu_ps(vec + i + 8, y_new);
    }
}

// Forward Step Completo do Espaco Tensorial Sumeriano
float sumerian_tensor_forward_step(
    float* state,
    uint32_t house_id,
    uint32_t realm_id,
    float tau,
    float delta_q
) {
    if (!g_initialized) {
        sumerian_init_tables();
    }

    uint32_t r_idx = (realm_id - 1) % SUMERIAN_REALMS;
    const float* attractor = g_realm_attractors[r_idx];

    // 1. Produto escalar (Atenção com o Atrator do Reino de Blake)
    float dot = sumerian_avx2_dot_product(state, attractor);
    float affinity = dot / sqrtf((float)TENSOR_DIM);

    // 2. Injeção de atrator e aceleração temporal Tau via FMA
    __m256 tau_vec = _mm256_set1_ps(tau * 0.15f);
    for (int i = 0; i < TENSOR_DIM; i += 8) {
        __m256 s = _mm256_loadu_ps(state + i);
        __m256 a = _mm256_loadu_ps(attractor + i);
        __m256 blended = _mm256_fmadd_ps(a, tau_vec, s);
        _mm256_storeu_ps(state + i, blended);
    }

    // 3. Rotação Simplética na Roda Sexagesimal Z_60
    sumerian_avx2_rotate_phase(state, house_id, delta_q);

    // 4. Normalização RMSNorm final
    sumerian_avx2_rmsnorm(state, 1e-6f);

    return affinity;
}

// Atualizacao Hebbiana do Lar Vetorial em Hardware AVX2 + FMA
void sumerian_lair_hebbian_update(float* centroid, const float* current, float lr) {
    __m256 lr_v = _mm256_set1_ps(lr);

    for (int i = 0; i < TENSOR_DIM; i += 8) {
        __m256 c = _mm256_loadu_ps(centroid + i);
        __m256 curr = _mm256_loadu_ps(current + i);
        __m256 diff = _mm256_sub_ps(curr, c);
        __m256 updated = _mm256_fmadd_ps(diff, lr_v, c);
        _mm256_storeu_ps(centroid + i, updated);
    }

    sumerian_avx2_rmsnorm(centroid, 1e-6f);
}

// Afinidade e Sensacao de Pertencimento no Lar Vetorial
float sumerian_lair_affinity(const float* current, const float* centroid) {
    float dot = sumerian_avx2_dot_product(current, centroid);
    return dot / sqrtf((float)TENSOR_DIM);
}

// Projecao Subcortex -> Cortex Gemma (640D -> 128D -> 1152D) via AVX2 / FMA
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
) {
    // 1. Projeção Bottleneck: W_a (128 x 640) * input_640 + bias -> z_128
    float z_128[128] __attribute__((aligned(32)));

    for (int r = 0; r < 128; r++) {
        const float* row = w_a_matrix + (r * 640);
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (int c = 0; c < 640; c += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c),      _mm256_loadu_ps(input_640 + c),      acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 8),  _mm256_loadu_ps(input_640 + c + 8),  acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 16), _mm256_loadu_ps(input_640 + c + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 24), _mm256_loadu_ps(input_640 + c + 24), acc3);
        }
        __m256 sum01 = _mm256_add_ps(acc0, acc1);
        __m256 sum23 = _mm256_add_ps(acc2, acc3);
        __m256 sum_all = _mm256_add_ps(sum01, sum23);

        __m128 lo = _mm256_castps256_ps128(sum_all);
        __m128 hi = _mm256_extractf128_ps(sum_all, 1);
        __m128 s128 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehl_ps(s128, s128);
        __m128 sums = _mm_add_ps(s128, shuf);
        shuf = _mm_shuffle_ps(sums, sums, 1);
        sums = _mm_add_ss(sums, shuf);

        float val = _mm_cvtss_f32(sums);
        if (w_a_bias) {
            val += w_a_bias[r];
        }
        // Ativação não-linear LeakyReLU
        z_128[r] = val > 0.0f ? val : (val * 0.01f);
    }

    // 2. Projeção de Soft Prompt: prefix_proj (1152 x 128) * z_128 + bias -> output_1152
    for (int r = 0; r < 1152; r++) {
        const float* row = prefix_proj_matrix + (r * 128);
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();

        for (int c = 0; c < 128; c += 32) {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c),      _mm256_loadu_ps(z_128 + c),      acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 8),  _mm256_loadu_ps(z_128 + c + 8),  acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 16), _mm256_loadu_ps(z_128 + c + 16), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(row + c + 24), _mm256_loadu_ps(z_128 + c + 24), acc3);
        }
        __m256 sum01 = _mm256_add_ps(acc0, acc1);
        __m256 sum23 = _mm256_add_ps(acc2, acc3);
        __m256 sum_all = _mm256_add_ps(sum01, sum23);

        __m128 lo = _mm256_castps256_ps128(sum_all);
        __m128 hi = _mm256_extractf128_ps(sum_all, 1);
        __m128 s128 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehl_ps(s128, s128);
        __m128 sums = _mm_add_ps(s128, shuf);
        shuf = _mm_shuffle_ps(sums, sums, 1);
        sums = _mm_add_ss(sums, shuf);

        float val = _mm_cvtss_f32(sums);
        if (prefix_proj_bias) {
            val += prefix_proj_bias[r];
        }
        if (v_instinct) {
            val += beta * v_instinct[r];
        }
        if (v_teo) {
            val += gamma * v_teo[r];
        }
        output_1152[r] = val;
    }
}


// ============================================================================
// SIMD PACKING AVX2: INT8 x UINT8 com VPMADDUBSW (_mm256_maddubs_epi16)
// ============================================================================

float sumerian_avx2_dot_product_packed_i8(
    const uint8_t* u8_act,
    const int8_t* i8_weight,
    int length,
    float scale_act,
    float scale_weight
) {
    __m256i acc_i32_0 = _mm256_setzero_si256();
    __m256i acc_i32_1 = _mm256_setzero_si256();
    __m256i acc_i32_2 = _mm256_setzero_si256();
    __m256i acc_i32_3 = _mm256_setzero_si256();
    __m256i ones_16 = _mm256_set1_epi16(1);

    int i = 0;
    // Desenrolamento de 128 elementos por ciclo de loop (4 x 32 bytes)
    for (; i <= length - 128; i += 128) {
        __m256i a0 = _mm256_loadu_si256((const __m256i*)(u8_act + i));
        __m256i w0 = _mm256_loadu_si256((const __m256i*)(i8_weight + i));
        __m256i dot0_16 = _mm256_maddubs_epi16(a0, w0);
        acc_i32_0 = _mm256_add_epi32(acc_i32_0, _mm256_madd_epi16(dot0_16, ones_16));

        __m256i a1 = _mm256_loadu_si256((const __m256i*)(u8_act + i + 32));
        __m256i w1 = _mm256_loadu_si256((const __m256i*)(i8_weight + i + 32));
        __m256i dot1_16 = _mm256_maddubs_epi16(a1, w1);
        acc_i32_1 = _mm256_add_epi32(acc_i32_1, _mm256_madd_epi16(dot1_16, ones_16));

        __m256i a2 = _mm256_loadu_si256((const __m256i*)(u8_act + i + 64));
        __m256i w2 = _mm256_loadu_si256((const __m256i*)(i8_weight + i + 64));
        __m256i dot2_16 = _mm256_maddubs_epi16(a2, w2);
        acc_i32_2 = _mm256_add_epi32(acc_i32_2, _mm256_madd_epi16(dot2_16, ones_16));

        __m256i a3 = _mm256_loadu_si256((const __m256i*)(u8_act + i + 96));
        __m256i w3 = _mm256_loadu_si256((const __m256i*)(i8_weight + i + 96));
        __m256i dot3_16 = _mm256_maddubs_epi16(a3, w3);
        acc_i32_3 = _mm256_add_epi32(acc_i32_3, _mm256_madd_epi16(dot3_16, ones_16));
    }

    for (; i <= length - 32; i += 32) {
        __m256i a = _mm256_loadu_si256((const __m256i*)(u8_act + i));
        __m256i w = _mm256_loadu_si256((const __m256i*)(i8_weight + i));
        __m256i dot_16 = _mm256_maddubs_epi16(a, w);
        acc_i32_0 = _mm256_add_epi32(acc_i32_0, _mm256_madd_epi16(dot_16, ones_16));
    }

    __m256i sum01 = _mm256_add_epi32(acc_i32_0, acc_i32_1);
    __m256i sum23 = _mm256_add_epi32(acc_i32_2, acc_i32_3);
    __m256i sum_all = _mm256_add_epi32(sum01, sum23);

    __m128i lo = _mm256_castsi256_si128(sum_all);
    __m128i hi = _mm256_extracti128_si256(sum_all, 1);
    __m128i s128 = _mm_add_epi32(lo, hi);
    __m128i shuf = _mm_shuffle_epi32(s128, _MM_SHUFFLE(1, 0, 3, 2));
    __m128i sums = _mm_add_epi32(s128, shuf);
    shuf = _mm_shuffle_epi32(sums, _MM_SHUFFLE(2, 3, 0, 1));
    sums = _mm_add_epi32(sums, shuf);

    int32_t int_accum = _mm_cvtsi128_si32(sums);

    for (; i < length; i++) {
        int_accum += (int32_t)u8_act[i] * (int32_t)i8_weight[i];
    }

    return (float)int_accum * (scale_act * scale_weight);
}

void sumerian_avx2_quantize_f32_to_i8(const float* src, int8_t* dst, int length, float* out_scale) {
    float amax = 1e-9f;
    for (int i = 0; i < length; i++) {
        float v = fabsf(src[i]);
        if (v > amax) amax = v;
    }
    float scale = amax / 127.0f;
    float inv_scale = 127.0f / amax;
    *out_scale = scale;

    __m256 vinv = _mm256_set1_ps(inv_scale);
    int i = 0;
    for (; i <= length - 8; i += 8) {
        __m256 f = _mm256_loadu_ps(src + i);
        __m256 scaled = _mm256_mul_ps(f, vinv);
        __m256i int32_vec = _mm256_cvtps_epi32(scaled);

        int32_t temp[8];
        _mm256_storeu_si256((__m256i*)temp, int32_vec);
        for (int k = 0; k < 8; k++) {
            int val = temp[k];
            if (val > 127) val = 127;
            if (val < -128) val = -128;
            dst[i + k] = (int8_t)val;
        }
    }
    for (; i < length; i++) {
        int val = (int)roundf(src[i] * inv_scale);
        if (val > 127) val = 127;
        if (val < -128) val = -128;
        dst[i] = (int8_t)val;
    }
}

void sumerian_avx2_quantize_f32_to_u8(const float* src, uint8_t* dst, int length, float* out_scale, float* out_min) {
    float min_val = 1e9f;
    float max_val = -1e9f;
    for (int i = 0; i < length; i++) {
        if (src[i] < min_val) min_val = src[i];
        if (src[i] > max_val) max_val = src[i];
    }
    float range = max_val - min_val;
    if (range < 1e-9f) range = 1e-9f;

    float scale = range / 255.0f;
    float inv_scale = 255.0f / range;
    *out_scale = scale;
    *out_min = min_val;

    for (int i = 0; i < length; i++) {
        float norm = (src[i] - min_val) * inv_scale;
        int val = (int)roundf(norm);
        if (val > 255) val = 255;
        if (val < 0) val = 0;
        dst[i] = (uint8_t)val;
    }
}

// ============================================================================
// CONVERSAO FP16 VIA F16C (_mm256_cvtps_ph / _mm256_cvtph_ps)
// ============================================================================

void export_prefix_fp16(const float* src_1152, uint16_t* dst_1152) {
    // Processa blocos de 8 floats32 -> 8 half-floats (FP16) por ciclo
    for (int i = 0; i < 1152; i += 8) {
        __m256 f32_vec = _mm256_loadu_ps(&src_1152[i]);
        // Converte 8x float32 para 8x float16 (vetor de 128 bits XMM)
        __m128i f16_vec = _mm256_cvtps_ph(f32_vec, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        _mm_storeu_si128((__m128i*)&dst_1152[i], f16_vec);
    }
}

void import_prefix_fp16(const uint16_t* src_1152, float* dst_1152) {
    // Descompacta 8 half-floats (FP16) -> 8 floats32 por ciclo
    for (int i = 0; i < 1152; i += 8) {
        __m128i f16_vec = _mm_loadu_si128((const __m128i*)&src_1152[i]);
        __m256 f32_vec = _mm256_cvtph_ps(f16_vec);
        _mm256_storeu_ps(&dst_1152[i], f32_vec);
    }
}
