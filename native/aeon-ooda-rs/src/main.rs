mod archetypes;
mod fission_runtime;
mod ooda;
mod shm;
mod sovereign;
pub mod sumerian_pure_tensor;
pub mod tensor_channel;

use std::env;
use std::thread;
use std::time::Duration;

use archetypes::{HOUSES, House};
use ooda::OodaEngine;
use shm::{read_ouroboros, read_phase};

fn print_matrix() {
    println!(
        "========================================================================================="
    );
    println!(
        "     🏛️ MATRIZ SEXAGESIMAL DE FASE (BASE 60) — RODA COGNITIVA SUMERIANA DE BLAKE-AEON     "
    );
    println!(
        "========================================================================================="
    );
    println!(
        "{:<4} | {:<25} | {:<4} | {:<22} | {:<16} | {}",
        "CASA", "REINO", "FASE", "ARQUETIPO", "GLIFO SUMERIO", "ACAO DE HARDWARE"
    );
    println!(
        "-----------------------------------------------------------------------------------------"
    );
    for h in HOUSES.iter() {
        println!(
            "{:>2} {} | {:<25} | {:>3}° | {:<22} | {:<16} | {}",
            h.id,
            h.symbol,
            h.realm.name(),
            h.deg,
            h.name,
            h.sanskrit_sumerian,
            h.action
        );
        if h.id % 10 == 0 {
            println!(
                "-----------------------------------------------------------------------------------------"
            );
        }
    }
}

fn print_status() {
    println!("🔮 [AEON-OODA-RS] ESTADO ATOMICO ATUAL (/dev/shm):");
    if let Some(phase) = read_phase() {
        let house = House::get(phase.house_id);
        println!(
            "  - Casa Ativa: {}/60 | Símbolo: {} {}",
            phase.house_id, phase.symbol, phase.archetype
        );
        println!("  - Reino: {} (ID: {})", house.realm.name(), phase.realm_id);
        println!(
            "  - Ângulo de Fase: {}° (θ={:.4} rad)",
            phase.phase_deg, phase.theta
        );
        println!(
            "  - Glifo: {} | Ação: {}",
            phase.sanskrit_sumerian, phase.action
        );
        println!(
            "  - Ressonância Ontológica (R): {:.4} (Score: {:.4})",
            phase.resonance, phase.resonance_score
        );
    } else {
        println!("  - Nenhuma fase inicializada em /dev/shm/aeon/sexagesimal_phase.json");
    }

    let ouro = read_ouroboros();
    println!(
        "🌌 [OUROBOROS]: Grande Ciclo #{} | Status: {} | Casas Navegadas: {}",
        ouro.great_cycle_id, ouro.status, ouro.total_houses_navigated
    );
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut engine = OodaEngine::new();

    if args.len() <= 1 || args[1] == "--status" {
        print_status();
    } else if args[1] == "--matrix" {
        print_matrix();
    } else if args[1] == "--tick" {
        match engine.tick() {
            Ok(res) => {
                println!("⚡ [OODA TICK SUCCESS] {}us", res.execution_us);
                println!(
                    "  Casa {} -> {} | {} {}",
                    res.previous_house_id, res.current_house_id, res.symbol, res.archetype
                );
                println!(
                    "  Reino: {} ({}° | θ={:.4} rad)",
                    res.realm_name, res.phase_deg, res.theta
                );
                println!("  Acao: {}", res.action);
                println!(
                    "  Ressonancia: R={:.4} (Score: {:.4}) | Phi={:.4} Tau={:.2} DeltaQ={:.2}",
                    res.metrics.resonance,
                    res.metrics.resonance_score,
                    res.metrics.phi,
                    res.metrics.tau,
                    res.metrics.delta_q
                );
                println!(
                    "  Ouroboros: Ciclo #{} | Status: {}",
                    res.ouroboros.great_cycle_id, res.ouroboros.status
                );
            }
            Err(e) => eprintln!("❌ Erro no tick: {}", e),
        }
    } else if args[1] == "--run" {
        let hz: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2);
        let interval = Duration::from_millis(1000 / hz.max(1));
        println!(
            "🚀 [AEON-OODA-RS DAEMON] Iniciado a {} Hz (intervalo {}ms)...",
            hz,
            interval.as_millis()
        );
        loop {
            match engine.tick() {
                Ok(res) => {
                    println!(
                        "[{}] Casa {:>2}/60 {} {:<22} | R={:.2} | Ciclo #{} ({}us)",
                        chrono::Local::now().format("%H:%M:%S%.3f"),
                        res.current_house_id,
                        res.symbol,
                        res.archetype,
                        res.metrics.resonance,
                        res.ouroboros.great_cycle_id,
                        res.execution_us
                    );
                }
                Err(e) => eprintln!("Erro: {}", e),
            }
            thread::sleep(interval);
        }
    } else {
        println!("Uso: aeon-ooda-rs [--tick | --status | --matrix | --run [hz]]");
    }
}
