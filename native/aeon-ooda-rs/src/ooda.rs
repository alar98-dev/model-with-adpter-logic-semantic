//! Motor OODA de Alta Frequencia (Observe - Orient - Decide - Act) em Rust.
//!
//! Submilissegundo, zero-copy, e perfeitamente sincronizado com o anel Z_60.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::archetypes::House;
use crate::shm::{OuroborosData, advance_ouroboros_step, read_phase, write_phase_atomic};
use crate::sovereign::SovereignMetrics;
use crate::tensor_channel::TensorPacket;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OodaTickResult {
    pub previous_house_id: u8,
    pub current_house_id: u8,
    pub realm_name: String,
    pub symbol: String,
    pub archetype: String,
    pub sanskrit_sumerian: String,
    pub action: String,
    pub phase_deg: u16,
    pub theta: f64,
    pub metrics: SovereignMetrics,
    pub ouroboros: OuroborosData,
    pub execution_us: u128,
}

pub struct OodaEngine {
    redis_client: Option<redis::Client>,
}

impl OodaEngine {
    pub fn new() -> Self {
        let redis_client = redis::Client::open("redis://127.0.0.1:6379/").ok();
        Self { redis_client }
    }

    /// Executa um ciclo completo OODA em tempo submilissegundo.
    pub fn tick(&mut self) -> Result<OodaTickResult, String> {
        let t0 = std::time::Instant::now();
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_secs_f64();

        // 1. OBSERVE (Afericao sensorial e leitura de estado)
        let (prev_id, next_house) = match read_phase() {
            Some(phase) => {
                let current = House::get(phase.house_id);
                (phase.house_id, current.next())
            }
            None => (60, House::get(1)),
        };

        // Telemetria fisica
        let has_dxg = Path::new("/dev/dxg").exists();
        let is_ext4 = !Path::new("/mnt/c").exists()
            || !std::env::current_dir()
                .map(|p| p.to_string_lossy().contains("/mnt/c"))
                .unwrap_or(false);
        let free_ram_ratio = Self::read_free_ram_ratio();

        // 2. ORIENT (Calculo da Ontologia de Soberania e Alinhamento Tensorial)
        let phi_vec = [12.5, 0.45, 0.35]; // [densidade_tokens, shm_ratio, cpu_ratio]
        let spectral_entropy = 0.42;
        let metrics =
            SovereignMetrics::compute(&phi_vec, has_dxg, is_ext4, free_ram_ratio, spectral_entropy);

        // 3. DECIDE (Transicao no anel discreto Z_60)
        let is_closure = next_house.is_ouroboros_closure();

        // 4. ACT (Gravacao atomica em /dev/shm e sinalizacao sinaptica)
        write_phase_atomic(next_house, &metrics, now)
            .map_err(|e| format!("Erro ao gravar SHM phase: {}", e))?;

        let ouro = advance_ouroboros_step(is_closure, now)
            .map_err(|e| format!("Erro ao atualizar Ouroboros: {}", e))?;

        // 4b. CANAL TENSORIAL NATIVO C/Rust COM INFERENCIA PURA AVX2/FMA E LAR VETORIAL
        let (geo_affinity, lair_affinity, latent_vector) =
            crate::sumerian_pure_tensor::step_global_tensor(
                next_house.id as u32,
                next_house.realm as u32,
                metrics.tau as f32,
                metrics.delta_q as f32,
            );

        let tensor_pkt = TensorPacket::with_vector(
            ouro.total_houses_navigated,
            next_house.id as u32,
            next_house.realm as u32,
            next_house.theta as f32,
            metrics.delta_q as f32,
            metrics.resonance_score as f32,
            metrics.resonance as f32 + (geo_affinity * 0.05) + (lair_affinity * 0.05),
            latent_vector,
        );
        let _ = tensor_pkt.write_to_shm();

        // Pulso no canal Redis se disponivel
        if let Some(ref client) = self.redis_client {
            if let Ok(mut con) = client.get_connection() {
                let payload = serde_json::json!({
                    "event": "OODA_PULSE_Z60",
                    "house_id": next_house.id,
                    "realm_id": next_house.realm as u8,
                    "archetype": next_house.name,
                    "resonance": metrics.resonance,
                    "ouroboros_cycle": ouro.great_cycle_id,
                    "ts": now
                });
                let _: Result<(), _> = redis::cmd("PUBLISH")
                    .arg("aeon:v6:cortex:act")
                    .arg(payload.to_string())
                    .query(&mut con);
            }
        }

        let elapsed = t0.elapsed().as_micros();

        Ok(OodaTickResult {
            previous_house_id: prev_id,
            current_house_id: next_house.id,
            realm_name: next_house.realm.name().to_string(),
            symbol: next_house.symbol.to_string(),
            archetype: next_house.name.to_string(),
            sanskrit_sumerian: next_house.sanskrit_sumerian.to_string(),
            action: next_house.action.to_string(),
            phase_deg: next_house.deg,
            theta: next_house.theta,
            metrics,
            ouroboros: ouro,
            execution_us: elapsed,
        })
    }

    fn read_free_ram_ratio() -> f64 {
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            let mut total = 1.0;
            let mut available = 0.5;
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(val) = line.split_whitespace().nth(1) {
                        total = val.parse::<f64>().unwrap_or(1.0);
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(val) = line.split_whitespace().nth(1) {
                        available = val.parse::<f64>().unwrap_or(0.5);
                    }
                }
            }
            (available / total).clamp(0.0, 1.0)
        } else {
            0.5
        }
    }
}
