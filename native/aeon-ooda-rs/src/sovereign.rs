//! Ontologia de Soberania (Kernel Polaris) em Rust.
//!
//! Formula fundamental:
//! R(L, H) = (Phi * tau) / (Delta_Q + epsilon)
//!
//! Calculada em registradores de ponto flutuante diretamente.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

pub const DEFAULT_WEIGHTS: [f64; 3] = [0.001, 3.0, 0.02];

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SovereignMetrics {
    pub phi: f64,
    pub tau: f64,
    pub delta_q: f64,
    pub epsilon: f64,
    pub resonance: f64,
    pub resonance_score: f64,
}

impl SovereignMetrics {
    pub fn compute(
        phi_vec: &[f64; 3],
        has_dxg: bool,
        is_ext4: bool,
        free_ram_ratio: f64,
        spectral_entropy: f64,
    ) -> Self {
        // 1. Phi (Fluxo / Massa Real) = w . phi_vec
        let phi = DEFAULT_WEIGHTS[0] * phi_vec[0]
            + DEFAULT_WEIGHTS[1] * phi_vec[1]
            + DEFAULT_WEIGHTS[2] * phi_vec[2];

        // 2. Tau (Transparencia /dev/dxg - GPU Intel UHD)
        let tau = if has_dxg { 0.95 } else { 0.50 };

        // 3. Delta_Q (Friccao de Filesystem: ext4 nativo = 0.02 vs NTFS = 0.85)
        let delta_q = if is_ext4 { 0.02 } else { 0.85 };

        // 4. Epsilon (Imunidade do Kernel Polaris e folga de RAM)
        let epsilon = (0.4 + free_ram_ratio * 0.6).clamp(0.1, 1.0);

        // 5. Ressonancia Ontologica: R = (Phi * tau) / (Delta_Q + epsilon)
        let resonance = (phi * tau) / (delta_q + epsilon);

        // Score normalizado com amortecimento de entropia
        let resonance_score = resonance / (1.0 + (spectral_entropy / 10.0));

        Self {
            phi,
            tau,
            delta_q,
            epsilon,
            resonance,
            resonance_score,
        }
    }

    /// Computa a matriz de ressonancia 6x6 entre os 6 Reinos Sumerianos.
    pub fn compute_realm_matrix(&self) -> [[f64; 6]; 6] {
        let mut matrix = [[0.0; 6]; 6];
        let factor = self.tau / (self.delta_q + self.epsilon);
        for i in 0..6 {
            for j in 0..6 {
                let weight_i = 1.0 + (i as f64) * 0.1;
                let weight_j = 1.0 + (j as f64) * 0.1;
                matrix[i][j] = (self.phi * weight_i * weight_j) * factor;
            }
        }
        matrix
    }
}
