import psutil
import json
import time
import os

class DebianPhysicsQuantizer:
    def __init__(self, output_file="/tmp/debian_physics.json"):
        self.output_file = output_file
        # Base readings to calculate deltas
        self.last_net_io = psutil.net_io_counters()
        self.last_disk_io = psutil.disk_io_counters()
        self.last_time = time.time()

    def quantize_physics(self):
        """Scans the Linux kernel and translates metrics into topological raw data."""
        current_time = time.time()
        dt = current_time - self.last_time if current_time - self.last_time > 0 else 1.0
        
        # 1. Thermal/Kinetic Energy (CPU)
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_cores = psutil.cpu_percent(interval=None, percpu=True)
        
        # 2. Gravitational Density (RAM)
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # 3. String Resonance (Network I/O)
        current_net = psutil.net_io_counters()
        bytes_sent_per_sec = (current_net.bytes_sent - self.last_net_io.bytes_sent) / dt
        bytes_recv_per_sec = (current_net.bytes_recv - self.last_net_io.bytes_recv) / dt
        self.last_net_io = current_net
        
        # 4. Matter Crystallization (Disk I/O)
        current_disk = psutil.disk_io_counters()
        if current_disk and self.last_disk_io:
            read_bytes_per_sec = (current_disk.read_bytes - self.last_disk_io.read_bytes) / dt
            write_bytes_per_sec = (current_disk.write_bytes - self.last_disk_io.write_bytes) / dt
        else:
            read_bytes_per_sec = 0
            write_bytes_per_sec = 0
        self.last_disk_io = current_disk
        self.last_time = current_time

        # Topological Compilation
        physics_state = {
            "kinetic_energy": {
                "global_heat": round(cpu_percent, 2),
                "core_variance": round(max(cpu_cores) - min(cpu_cores), 2)
            },
            "gravitational_density": {
                "mass_pressure": round(mem.percent, 2),
                "black_hole_event": swap.percent > 10.0 # Swap is the event horizon
            },
            "string_resonance": {
                "outward_pull": round(bytes_sent_per_sec / 1024, 2), # KB/s
                "inward_pull": round(bytes_recv_per_sec / 1024, 2)
            },
            "crystallization": {
                "formation": round(write_bytes_per_sec / 1024, 2), # KB/s
                "dissolution": round(read_bytes_per_sec / 1024, 2)
            }
        }

        # Atomically write to shared space
        tmp_file = f"{self.output_file}.tmp"
        with open(tmp_file, "w") as f:
            json.dump(physics_state, f)
        os.rename(tmp_file, self.output_file)
        
        return physics_state

    def start_loop(self, frequency=2.0):
        print(f"🌀 [DEBIAN PHYSICS] Quantizer started. Scanning kernel every {frequency}s.")
        # Prime the delta counters
        psutil.cpu_percent(interval=1.0)
        while True:
            try:
                self.quantize_physics()
                time.sleep(frequency)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️ [DEBIAN PHYSICS] Fissura na quantização: {e}")
                time.sleep(frequency)

if __name__ == "__main__":
    quantizer = DebianPhysicsQuantizer()
    quantizer.start_loop()
