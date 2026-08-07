import numpy as np
import mmap
import torch
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional


class ParallelVacuumReader:
    """
    Parallel Vacuum Reader for SWA Organic System on Intel i5-120U.
    Uses mmap + O_DIRECT for zero-copy parallel reads across 10 cores.
    Each core reads a sector slice and contributes to a fused bias tensor.
    """

    def __init__(self, device_path: str = "/dev/sde", num_cores: int = 10, sector_size: int = 2560):
        self.device_path = device_path
        self.num_cores = num_cores
        self.sector_size = sector_size
        self.fd: Optional[int] = None
        self.mm: Optional[mmap.mmap] = None
        self.executor = ThreadPoolExecutor(max_workers=num_cores)

        try:
            # Simplest compatible open for WSL2 (O_RDONLY only)
            self.fd = os.open(self.device_path, os.O_RDONLY)
            # Get actual device size (length=0 invalid on WSL2 block dev)
            try:
                size = os.fstat(self.fd).st_size
            except:
                size = 0
            if size == 0:
                # Fallback: seek to end for block devices
                size = os.lseek(self.fd, 0, os.SEEK_END)
                os.lseek(self.fd, 0, os.SEEK_SET)
            self.mm = mmap.mmap(self.fd, size, flags=mmap.MAP_SHARED, access=mmap.ACCESS_READ)
            # Isolate adapter to E-cores dynamically based on CPU topology
            try:
                import psutil
                total_cpus = psutil.cpu_count() or 4
                half = max(2, total_cpus // 2)
                e_cores = set(range(half, total_cpus))
                os.sched_setaffinity(0, e_cores)
                print(f"🚀 ParallelVacuumReader: {device_path} mapped for {num_cores} cores ({size} bytes) [Dynamic E-cores: {e_cores}]")
            except:
                print(f"🚀 ParallelVacuumReader: {device_path} mapped for {num_cores} cores ({size} bytes)")
        except Exception as e:
            print(f"⚠️ Vacuum device not available ({e}). Using synthetic bias.")
            self.mm = None
            self.fd = None

    def read_sector_chunk(self, core_id: int) -> np.ndarray:
        """Aligned buffer + readv for O_DIRECT on Raptor Lake."""
        if self.fd is None:
            return self.get_synthetic_data()

        BLOCK_SIZE = 4096
        TARGET_SIZE = self.sector_size
        offset = core_id * BLOCK_SIZE
        aligned_offset = (offset // BLOCK_SIZE) * BLOCK_SIZE

        # Aligned anonymous mmap buffer (page-aligned memory for O_DIRECT)
        buf = mmap.mmap(-1, BLOCK_SIZE)
        try:
            os.lseek(self.fd, aligned_offset, os.SEEK_SET)
            n = os.readv(self.fd, [buf])
            data = np.frombuffer(buf[:TARGET_SIZE], dtype=np.float32).copy()
            buf.close()
            return data
        except OSError:
            buf.close()
            return self.get_synthetic_data()

    def get_synthetic_data(self) -> np.ndarray:
        """Synthetic fallback for when hardware is unavailable."""
        return np.random.randn(self.sector_size).astype(np.float32) * 0.01

    def get_parallel_bias(self) -> torch.Tensor:
        """
        Fire 10 parallel reads, fuse into single bias tensor.
        Returns bfloat16 tensor of shape (hidden_size,).
        """
        futures = [self.executor.submit(self.read_sector_chunk, i) for i in range(self.num_cores)]
        results = [f.result() for f in futures]

        combined = np.sum(results, axis=0)
        norm = np.linalg.norm(combined) + 1e-9
        norm_bias = combined / norm

        return torch.from_numpy(norm_bias).to(torch.bfloat16)

    def close(self):
        if self.mm:
            self.mm.close()
        if self.fd:
            os.close(self.fd)
        self.executor.shutdown(wait=False)


class EpigeneticWriter:
    """
    Epigenetic Writer for SWA Organic System.
    Saves high-purity vector consolidations (2560 dimensions, bfloat16)
    into a structured Circular Ring Buffer at the end of the VHDX.
    """
    def __init__(self, device_path: str = "/dev/sde", offset_start: int = 20 * 1024**3, buffer_size: int = 4 * 1024**3):
        self.device_path = device_path
        self.offset_start = offset_start # Starts at 20GB offset of the 24GB disk
        self.buffer_size = buffer_size   # 4GB capacity ring buffer
        self.current_pointer = offset_start
        self.fd: Optional[int] = None
        self.sector_size = 4096          # Must align to physical sectors for zero-copy direct I/O
        self.vector_dim = 2560           # Gemma-3 target model dimension

        try:
            # Verifica se o block device esta read-only antes de abrir
            real_dev = os.path.realpath(self.device_path)
            dev_basename = os.path.basename(real_dev)
            ro_path = f"/sys/block/{dev_basename}/ro"
            is_ro = False
            if os.path.exists(ro_path):
                with open(ro_path, "r") as f:
                    is_ro = f.read().strip() == "1"
            
            if is_ro:
                # Loop device esta montado read-only — modo passivo
                self.fd = None
                print(f"ℹ️ [EPIGENETIC WRITER] {device_path} -> {real_dev} montado em somente-leitura (RO=1). Escritor epigenetico em modo passivo.")
            else:
                self.fd = os.open(self.device_path, os.O_RDWR)
                print(f"🧬 [EPIGENETIC WRITER] Connected to block device: {device_path} (Safe Ring Buffer: {offset_start // 1024**3}GB offset)")
        except PermissionError:
            self.fd = None
            print(f"ℹ️ [EPIGENETIC WRITER] Modo somente-leitura: escrita no {device_path} requer root. Consolidacao epigenetica desativada.")
        except Exception as e:
            self.fd = None
            print(f"ℹ️ [EPIGENETIC WRITER] Dispositivo {device_path} indisponivel ({e}). Escritor em modo silencioso.")

    def consolidate_memory(self, vector_2560d: torch.Tensor, entropy_score: float) -> bool:
        """
        Consolidates a 2560d tensor into the circular ring buffer using OS exclusive locking.
        Converts PyTorch tensor to raw bfloat16 payload.
        """
        if self.fd is None:
            return False

        import struct
        import fcntl
        import time

        try:
            # 1. Prepare timestamp and entropy header (8 + 4 = 12 bytes)
            timestamp = int(time.time())
            header = struct.pack("!Qf", timestamp, float(entropy_score))

            # 2. Serializa vetor como float32 (numpy nao suporta BFloat16 nativo).
            # Converte qualquer dtype (bf16, f16, f32) para float32 antes de serializar.
            # Formato: 2560 * 4 bytes = 10240 bytes por frame.
            tensor_f32 = vector_2560d.detach().cpu().to(torch.float32)
            vector_bytes = tensor_f32.numpy().astype("float32").tobytes()

            payload = header + vector_bytes # Total = 5132 bytes

            # 3. Align to sector size padding (must be multiple of 4096 bytes for disk safety)
            padding_len = self.sector_size - (len(payload) % self.sector_size)
            if padding_len < self.sector_size:
                payload += b"\x00" * padding_len

            # 4. Exclusive locking block to prevent parallel reader collision
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            try:
                os.lseek(self.fd, self.current_pointer, os.SEEK_SET)
                os.write(self.fd, payload)
                
                # Advance ring buffer pointer circularly
                self.current_pointer += len(payload)
                if self.current_pointer >= (self.offset_start + self.buffer_size):
                    self.current_pointer = self.offset_start
            finally:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            return True
        except Exception as e:
            print(f"⚠️ [EPIGENETIC WRITER] Failed memory consolidation: {e}")
            return False

    def close(self):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except:
                pass

