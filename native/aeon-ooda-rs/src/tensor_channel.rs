use std::fs::OpenOptions;
use std::io::{Result, Write};
use std::path::Path;

pub const TENSOR_MAGIC: u32 = 0x53574154; // 'SWAT'
pub const TENSOR_DIM: usize = 640;
// #[repr(C)] layout: 68-byte header, 640 float32 values, 60-byte tail
// padding introduced by align(64). Readers must use 68 as vector offset.
pub const TENSOR_HEADER_SIZE: usize = 68;
pub const TENSOR_PACKET_SIZE: usize = 2688;

#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TensorPacket {
    pub magic: u32,
    pub version: u32,
    pub sequence: u64,
    pub timestamp_sec: u64,
    pub timestamp_nsec: u32,
    pub house_id: u32,
    pub realm_id: u32,
    pub theta_rad: f32,
    pub delta_q: f32,
    pub coherence: f32,
    pub resonance: f32,
    pub vector_dim: u32,
    pub _reserved: [u32; 3],
    pub latent_vector: [f32; TENSOR_DIM],
}

impl TensorPacket {
    pub fn with_vector(
        sequence: u64,
        house_id: u32,
        realm_id: u32,
        theta_rad: f32,
        delta_q: f32,
        coherence: f32,
        resonance: f32,
        latent_vector: [f32; TENSOR_DIM],
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();

        Self {
            magic: TENSOR_MAGIC,
            version: 1,
            sequence,
            timestamp_sec: now.as_secs(),
            timestamp_nsec: now.subsec_nanos(),
            house_id,
            realm_id,
            theta_rad,
            delta_q,
            coherence,
            resonance,
            vector_dim: TENSOR_DIM as u32,
            _reserved: [0; 3],
            latent_vector,
        }
    }

    pub fn new(
        sequence: u64,
        house_id: u32,
        realm_id: u32,
        theta_rad: f32,
        delta_q: f32,
        coherence: f32,
        resonance: f32,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();

        let mut pkt = Self {
            magic: TENSOR_MAGIC,
            version: 1,
            sequence,
            timestamp_sec: now.as_secs(),
            timestamp_nsec: now.subsec_nanos(),
            house_id,
            realm_id,
            theta_rad,
            delta_q,
            coherence,
            resonance,
            vector_dim: TENSOR_DIM as u32,
            _reserved: [0; 3],
            latent_vector: [0.0; TENSOR_DIM],
        };

        // Sintetizar vetor latente 640D sob a Roda Sexagesimal e Blake Gravity
        let base_freq = theta_rad;
        for i in 0..TENSOR_DIM {
            let k = i as f32;
            let harmonic =
                (base_freq + (2.0 * std::f32::consts::PI * k / (TENSOR_DIM as f32))).sin();
            let damping = (-delta_q * (k / 100.0)).exp();
            pkt.latent_vector[i] = harmonic * resonance * damping;
        }

        pkt
    }

    pub fn write_to_shm(&self) -> Result<()> {
        let dir = Path::new("/dev/shm/aeon");
        if !dir.exists() {
            let _ = std::fs::create_dir_all(dir);
        }

        let tmp_path = "/dev/shm/aeon/tensor_bus.bin.tmp";
        let final_path = "/dev/shm/aeon/tensor_bus.bin";

        let bytes = unsafe {
            std::slice::from_raw_parts(
                (self as *const Self) as *const u8,
                std::mem::size_of::<Self>(),
            )
        };

        {
            let mut file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(tmp_path)?;
            file.write_all(bytes)?;
            file.flush()?;
        }

        std::fs::rename(tmp_path, final_path)?;
        Ok(())
    }
}
