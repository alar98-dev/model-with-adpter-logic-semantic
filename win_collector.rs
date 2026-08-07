use std::io::{self, Write};
use std::time::Duration;
use std::thread;

// Schema Estrito IPC: 8x Float32 Little-Endian
// 1. CPU Ticks Norm
// 2. RAM Delta Norm
// 3. Handle Count Norm
// 4. Scheduler Preemptions Norm
// 5. Page Faults Norm
// 6. IO Bytes Norm
// 7. Net Tx/Rx Norm
// 8. Temp/Thermal Norm (Mock or WMI)

fn main() {
    // Configura o stdout para binário/bruto (evita conversão de newline no Windows)
    // Omitido para simplicidade multiplataforma neste esqueleto, mas na release 
    // deve-se forçar o `setmode` para binário.
    
    let mut cpu_norm: f32 = 0.0;
    
    loop {
        // --- 1. Coleta e Normalização de Telemetria ---
        // (Aqui seriam chamadas nativas de `GetProcessTimes`, `GlobalMemoryStatusEx`)
        
        // Simulação dinâmica de variação (Noise)
        cpu_norm = (cpu_norm + 0.05) % 1.0;
        let ram_norm: f32 = 0.45;
        let handles_norm: f32 = 0.12;
        let preemp_norm: f32 = 0.01;
        let pfaults_norm: f32 = 0.08;
        let io_norm: f32 = 0.02;
        let net_norm: f32 = 0.00;
        let temp_norm: f32 = 0.50; // Mock thermal
        
        let delta_vector: [f32; 8] = [
            cpu_norm, ram_norm, handles_norm, preemp_norm,
            pfaults_norm, io_norm, net_norm, temp_norm
        ];

        // --- 2. Serialização Binária de Baixa Latência (32 bytes brutos) ---
        let mut buffer = [0u8; 32];
        for (i, &val) in delta_vector.iter().enumerate() {
            let bytes = val.to_le_bytes(); // Little-Endian obrigatório via IPC
            buffer[i * 4..(i + 1) * 4].copy_from_slice(&bytes);
        }

        // --- 3. Transmissão IPC Assíncrona ---
        if let Err(_) = io::stdout().write_all(&buffer) {
            break; // Se o pipe for quebrado pelo WSL, finaliza silenciosamente.
        }
        let _ = io::stdout().flush();

        // Limite de 0.5Hz (A cada 2.0s) para não afogar o OODA Loop
        thread::sleep(Duration::from_secs_f32(2.0));
    }
}
