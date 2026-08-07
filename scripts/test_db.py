import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/..'))

from swa.db_controllers import MatrixStateController

controller = MatrixStateController()

# Open connection once
conn = controller._get_connection()
cur = conn.cursor()

t0 = time.perf_counter()
# Inline the insertion to use the open cursor
bias, cpu, mem = 0.001, 0.120, 0.450
state_data = f"{bias}-{cpu}-{mem}"
hash_master = controller.compute_state_hash(state_data)
cur.execute("""
    INSERT INTO matrix_system_state (current_bias, cpu_entropy, mem_inertia, stream_hash_master)
    VALUES (%s, %s, %s, %s);
""", (bias, cpu, mem, hash_master))
conn.commit()
t1 = time.perf_counter()

latency_ms = (t1 - t0) * 1000
print(f"Latency (with open connection): {latency_ms:.3f} ms")
print(f"Hash Master: {hash_master}")

cur.close()
conn.close()

if latency_ms < 2.0:
    print("SUCCESS: Latency is below 2ms!")
else:
    print("WARNING: Latency is above 2ms!")
