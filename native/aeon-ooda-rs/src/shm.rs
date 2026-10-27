//! Ponte de Memoria Compartilhada /dev/shm para o Motor OODA.
//!
//! Acesso atomico e lock-free a sexagesimal_phase.json e ouroboros_cycle.json.

use crate::archetypes::House;
use crate::sovereign::SovereignMetrics;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

pub const SHM_DIR: &str = "/dev/shm/aeon";
pub const PHASE_FILE: &str = "/dev/shm/aeon/sexagesimal_phase.json";
pub const OUROBOROS_FILE: &str = "/dev/shm/aeon/ouroboros_cycle.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseData {
    pub house_id: u8,
    pub realm_id: u8,
    pub phase_deg: u16,
    pub theta: f64,
    pub archetype: String,
    pub symbol: String,
    pub sanskrit_sumerian: String,
    pub action: String,
    pub resonance: f64,
    pub resonance_score: f64,
    pub timestamp: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OuroborosData {
    pub great_cycle_id: u64,
    pub status: String,
    pub total_houses_navigated: u64,
    pub timestamp: f64,
    pub last_apoptosis_ts: f64,
}

pub fn ensure_shm_dir() -> std::io::Result<()> {
    if !Path::new(SHM_DIR).exists() {
        fs::create_dir_all(SHM_DIR)?;
    }
    Ok(())
}

pub fn read_phase() -> Option<PhaseData> {
    let content = fs::read_to_string(PHASE_FILE).ok()?;
    serde_json::from_str(&content).ok()
}

pub fn write_phase_atomic(
    house: &House,
    metrics: &SovereignMetrics,
    now: f64,
) -> std::io::Result<()> {
    ensure_shm_dir()?;
    let data = PhaseData {
        house_id: house.id,
        realm_id: house.realm as u8,
        phase_deg: house.deg,
        theta: house.theta,
        archetype: house.name.to_string(),
        symbol: house.symbol.to_string(),
        sanskrit_sumerian: house.sanskrit_sumerian.to_string(),
        action: house.action.to_string(),
        resonance: metrics.resonance,
        resonance_score: metrics.resonance_score,
        timestamp: now,
    };

    let json_bytes = serde_json::to_vec_pretty(&data)?;
    let tmp_path = format!("{}.tmp.{}", PHASE_FILE, std::process::id());
    {
        let mut f = File::create(&tmp_path)?;
        f.write_all(&json_bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp_path, PHASE_FILE)?;
    Ok(())
}

pub fn read_ouroboros() -> OuroborosData {
    if let Ok(content) = fs::read_to_string(OUROBOROS_FILE) {
        if let Ok(data) = serde_json::from_str(&content) {
            return data;
        }
    }
    OuroborosData {
        great_cycle_id: 1,
        status: "ACTIVE".to_string(),
        total_houses_navigated: 0,
        timestamp: 0.0,
        last_apoptosis_ts: 0.0,
    }
}

pub fn advance_ouroboros_step(is_closure: bool, now: f64) -> std::io::Result<OuroborosData> {
    ensure_shm_dir()?;
    let mut ouro = read_ouroboros();
    ouro.total_houses_navigated += 1;
    ouro.timestamp = now;

    if is_closure {
        ouro.great_cycle_id += 1;
        ouro.status = "OUROBOROS_APOPTOSIS_EXECUTED".to_string();
        ouro.last_apoptosis_ts = now;
    } else {
        ouro.status = "CYCLE_IN_PROGRESS".to_string();
    }

    let json_bytes = serde_json::to_vec_pretty(&ouro)?;
    let tmp_path = format!("{}.tmp.{}", OUROBOROS_FILE, std::process::id());
    {
        let mut f = File::create(&tmp_path)?;
        f.write_all(&json_bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp_path, OUROBOROS_FILE)?;
    Ok(ouro)
}
