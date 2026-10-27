#![allow(dead_code)]

use crate::tensor_channel::TENSOR_DIM;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::Path;
use std::sync::Mutex;

const SUMERIAN_HOUSES: usize = 60;
const LAIR_SHM_PATH: &str = "/dev/shm/aeon/vector_lair.bin";
const LAIR_PERSIST_PATH: &str = "/services/agent-dev/data/vector_lair.bin";

unsafe extern "C" {
    fn sumerian_init_tables();
    fn sumerian_avx2_dot_product(a: *const f32, b: *const f32) -> f32;
    fn sumerian_avx2_rmsnorm(vec: *mut f32, eps: f32);
    fn sumerian_avx2_rotate_phase(vec: *mut f32, house_id: u32, delta_q: f32);
    fn sumerian_tensor_forward_step(
        state: *mut f32,
        house_id: u32,
        realm_id: u32,
        tau: f32,
        delta_q: f32,
    ) -> f32;
    fn sumerian_lair_hebbian_update(centroid: *mut f32, current: *const f32, lr: f32);
    fn sumerian_lair_affinity(current: *const f32, centroid: *const f32) -> f32;
    fn sumerian_avx2_dot_product_packed_i8(
        u8_act: *const u8,
        i8_weight: *const i8,
        length: i32,
        scale_act: f32,
        scale_weight: f32,
    ) -> f32;
    fn sumerian_avx2_quantize_f32_to_i8(
        src: *const f32,
        dst: *mut i8,
        length: i32,
        out_scale: *mut f32,
    );
    fn sumerian_avx2_quantize_f32_to_u8(
        src: *const f32,
        dst: *mut u8,
        length: i32,
        out_scale: *mut f32,
        out_min: *mut f32,
    );
    fn export_prefix_fp16(src: *const f32, dst: *mut u16);
    fn sumerian_cortex_projection_step(
        input_640: *const f32,
        w_a_matrix: *const f32,
        w_a_bias: *const f32,
        prefix_proj_matrix: *const f32,
        prefix_proj_bias: *const f32,
        v_instinct: *const f32,
        v_teo: *const f32,
        beta: f32,
        gamma: f32,
        output_1152: *mut f32,
    );
}

#[repr(C, align(64))]
pub struct SumerianTensorState {
    pub vector: [f32; TENSOR_DIM],
}

#[repr(C, align(64))]
pub struct VectorLair {
    pub magic: u32, // 0x4C414952 'LAIR'
    pub version: u32,
    pub great_cycles: u64,
    pub total_assimilations: u64,
    pub centroids: Box<[[f32; TENSOR_DIM]; SUMERIAN_HOUSES]>,
}

impl VectorLair {
    pub fn new() -> Self {
        // Alocacao no heap com seguranca de memoria
        let mut raw_centroids = Box::new([[0.0f32; TENSOR_DIM]; SUMERIAN_HOUSES]);

        // Tentar restaurar do disco persistente se existir
        if let Ok(mut f) = File::open(LAIR_PERSIST_PATH) {
            let mut header = [0u8; 24];
            if f.read_exact(&mut header).is_ok() {
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(
                        raw_centroids.as_mut_ptr() as *mut u8,
                        SUMERIAN_HOUSES * TENSOR_DIM * 4,
                    )
                };
                let _ = f.read_exact(bytes);
                return Self {
                    magic: 0x4C414952,
                    version: 1,
                    great_cycles: 0,
                    total_assimilations: 0,
                    centroids: raw_centroids,
                };
            }
        }

        // Sintetizar centróides harmônicos primordiais
        for h in 0..SUMERIAN_HOUSES {
            let theta = (2.0 * std::f32::consts::PI * h as f32) / (SUMERIAN_HOUSES as f32);
            for i in 0..TENSOR_DIM {
                let k = i as f32;
                raw_centroids[h][i] = (theta + (2.0 * std::f32::consts::PI * k / 64.0)).sin();
            }
            unsafe {
                sumerian_avx2_rmsnorm(raw_centroids[h].as_mut_ptr(), 1e-6);
            }
        }

        Self {
            magic: 0x4C414952,
            version: 1,
            great_cycles: 0,
            total_assimilations: 0,
            centroids: raw_centroids,
        }
    }

    pub fn assimilate(
        &mut self,
        house_id: u32,
        current_vector: &[f32; TENSOR_DIM],
        delta_q: f32,
    ) -> f32 {
        let h_idx = ((house_id.saturating_sub(1)) as usize) % SUMERIAN_HOUSES;
        let lr = (0.04 / (1.0 + delta_q)).clamp(0.005, 0.08);

        unsafe {
            sumerian_lair_hebbian_update(
                self.centroids[h_idx].as_mut_ptr(),
                current_vector.as_ptr(),
                lr,
            );
            self.total_assimilations += 1;
            sumerian_lair_affinity(current_vector.as_ptr(), self.centroids[h_idx].as_ptr())
        }
    }

    pub fn persist_to_disk_and_shm(&self) {
        let header = [self.magic.to_le_bytes(), self.version.to_le_bytes()].concat();

        let centroids_bytes = unsafe {
            std::slice::from_raw_parts(
                self.centroids.as_ptr() as *const u8,
                SUMERIAN_HOUSES * TENSOR_DIM * 4,
            )
        };

        let write_all_data = |path: &str| {
            if let Some(parent) = Path::new(path).parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let Ok(mut f) = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(path)
            {
                let _ = f.write_all(&header);
                let _ = f.write_all(&self.great_cycles.to_le_bytes());
                let _ = f.write_all(&self.total_assimilations.to_le_bytes());
                let _ = f.write_all(centroids_bytes);
            }
        };

        write_all_data(LAIR_SHM_PATH);
        write_all_data(LAIR_PERSIST_PATH);
    }
}

fn load_floats_from_file(path: &str, target: &mut [f32]) -> bool {
    if let Ok(mut f) = File::open(path) {
        let byte_len = target.len() * 4;
        let byte_slice =
            unsafe { std::slice::from_raw_parts_mut(target.as_mut_ptr() as *mut u8, byte_len) };
        f.read_exact(byte_slice).is_ok()
    } else {
        false
    }
}

pub const CORTEX_DIM: usize = 1152;
const CORTEX_SHM_PATH: &str = "/dev/shm/aeon/cortex_prefix_1152.bin";
const WEIGHTS_DIR: &str = "/home/alifer/swa-aeon/native/weights";

fn write_atomic(path: &str, bytes: &[u8]) -> std::io::Result<()> {
    let temporary = format!("{path}.tmp");
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_data()?;
    std::fs::rename(temporary, path)
}

pub struct CortexBridge {
    pub w_a: Box<[f32; 128 * 640]>,
    pub w_a_bias: Box<[f32; 128]>,
    pub prefix_proj: Box<[f32; 1152 * 128]>,
    pub prefix_proj_bias: Box<[f32; 1152]>,
    pub v_instinct: Box<[f32; 1152]>,
    pub v_teo: Box<[f32; 1152]>,
    pub loaded: bool,
}

impl CortexBridge {
    pub fn new() -> Self {
        let mut w_a = Box::new([0.0f32; 128 * 640]);
        let mut w_a_bias = Box::new([0.0f32; 128]);
        let mut prefix_proj = Box::new([0.0f32; 1152 * 128]);
        let mut prefix_proj_bias = Box::new([0.0f32; 1152]);
        let mut v_instinct = Box::new([0.0f32; 1152]);
        let mut v_teo = Box::new([0.0f32; 1152]);

        let ok1 = load_floats_from_file(&format!("{}/w_a_128x640.bin", WEIGHTS_DIR), w_a.as_mut());
        let ok2 = load_floats_from_file(
            &format!("{}/w_a_bias_128.bin", WEIGHTS_DIR),
            w_a_bias.as_mut(),
        );
        let ok3 = load_floats_from_file(
            &format!("{}/prefix_proj_1152x128.bin", WEIGHTS_DIR),
            prefix_proj.as_mut(),
        );
        let ok4 = load_floats_from_file(
            &format!("{}/prefix_proj_bias_1152.bin", WEIGHTS_DIR),
            prefix_proj_bias.as_mut(),
        );
        let _ = load_floats_from_file(
            &format!("{}/v_instinct_1152.bin", WEIGHTS_DIR),
            v_instinct.as_mut(),
        );
        let _ = load_floats_from_file(&format!("{}/v_teo_1152.bin", WEIGHTS_DIR), v_teo.as_mut());

        let loaded = ok1 && ok2 && ok3 && ok4;
        Self {
            w_a,
            w_a_bias,
            prefix_proj,
            prefix_proj_bias,
            v_instinct,
            v_teo,
            loaded,
        }
    }

    pub fn project_and_publish(
        &self,
        input_640: &[f32; TENSOR_DIM],
        beta: f32,
        gamma: f32,
    ) -> [f32; CORTEX_DIM] {
        let mut out = [0.0f32; CORTEX_DIM];
        if self.loaded {
            unsafe {
                sumerian_cortex_projection_step(
                    input_640.as_ptr(),
                    self.w_a.as_ptr(),
                    self.w_a_bias.as_ptr(),
                    self.prefix_proj.as_ptr(),
                    self.prefix_proj_bias.as_ptr(),
                    self.v_instinct.as_ptr(),
                    self.v_teo.as_ptr(),
                    beta,
                    gamma,
                    out.as_mut_ptr(),
                );
            }

            // Publicar de forma atômica em /dev/shm/aeon/cortex_prefix_1152.bin
            let byte_slice =
                unsafe { std::slice::from_raw_parts(out.as_ptr() as *const u8, CORTEX_DIM * 4) };
            let _ = write_atomic(CORTEX_SHM_PATH, byte_slice);

            // Publicar também em FP16 Nativo (2304 bytes via F16C nativo da CPU)
            let mut f16_buf = [0u16; CORTEX_DIM];
            unsafe {
                export_prefix_fp16(out.as_ptr(), f16_buf.as_mut_ptr());
            }
            let byte_slice_f16 = unsafe {
                std::slice::from_raw_parts(f16_buf.as_ptr() as *const u8, CORTEX_DIM * 2)
            };
            let _ = write_atomic("/dev/shm/aeon/cortex_prefix_1152_f16.bin", byte_slice_f16);
        }
        out
    }
}

pub struct SumerianTensorEngine {
    state: SumerianTensorState,
    pub lair: VectorLair,
    pub cortex: CortexBridge,
}

impl SumerianTensorEngine {
    pub fn new() -> Self {
        unsafe {
            sumerian_init_tables();
        }

        let mut state = SumerianTensorState {
            vector: [0.0; TENSOR_DIM],
        };

        for i in 0..TENSOR_DIM {
            let k = i as f32;
            state.vector[i] = (2.0 * std::f32::consts::PI * k / 64.0).sin();
        }

        unsafe {
            sumerian_avx2_rmsnorm(state.vector.as_mut_ptr(), 1e-6);
        }

        let lair = VectorLair::new();
        lair.persist_to_disk_and_shm();
        let cortex = CortexBridge::new();

        Self {
            state,
            lair,
            cortex,
        }
    }

    pub fn forward_step(
        &mut self,
        house_id: u32,
        realm_id: u32,
        tau: f32,
        delta_q: f32,
    ) -> (f32, f32, [f32; TENSOR_DIM]) {
        let realm_affinity = unsafe {
            sumerian_tensor_forward_step(
                self.state.vector.as_mut_ptr(),
                house_id,
                realm_id,
                tau,
                delta_q,
            )
        };

        let lair_affinity = self.lair.assimilate(house_id, &self.state.vector, delta_q);

        if self.lair.total_assimilations % 10 == 0 {
            self.lair.persist_to_disk_and_shm();
        }

        // Projeção Subcórtex -> Córtex Gemma (1152D) via AVX2 em tempo de execução
        let _ = self
            .cortex
            .project_and_publish(&self.state.vector, 0.05, 0.05);

        (realm_affinity, lair_affinity, self.state.vector)
    }
}

static GLOBAL_ENGINE: Mutex<Option<SumerianTensorEngine>> = Mutex::new(None);

pub fn step_global_tensor(
    house_id: u32,
    realm_id: u32,
    tau: f32,
    delta_q: f32,
) -> (f32, f32, [f32; TENSOR_DIM]) {
    let mut guard = GLOBAL_ENGINE.lock().unwrap();
    if guard.is_none() {
        *guard = Some(SumerianTensorEngine::new());
    }
    guard
        .as_mut()
        .unwrap()
        .forward_step(house_id, realm_id, tau, delta_q)
}
