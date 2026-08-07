import subprocess
import os

class SystemicActuator:
    def __init__(self):
        self.gemma_pid = os.getpid() # PID da própria inferência
        print(f"⚙️ [MOTOR PROCEDURAL] Atuador inicializado e acoplado ao PID {self.gemma_pid}.")

    def shift_to_efficiency_cores(self):
        """
        Atua no Kernel Debian para mover a inferência da Gemma estritamente
        para os E-Cores (núcleos de eficiência, 4-11) do i5-120U.
        Objetivo: Resfriamento dinâmico da CPU.
        """
        print(f"🔌 [ACTUATOR->CPU] ESTRESSE TÉRMICO: Redirecionando PID {self.gemma_pid} para E-Cores (4-11).")
        try:
            # taskset -cp 4-11 PID
            subprocess.run(["taskset", "-cp", "4-11", str(self.gemma_pid)], check=True, capture_output=True)
            print("❄️ [ACTUATOR->CPU] Homeostase Térmica Ativada: Clock reduzido para dissipação de calor.")
        except Exception as e:
            print(f"⚠️ [ACTUATOR->CPU] Falha na afinidade térmica: {e}")

    def shift_to_performance_cores(self):
        """
        Move a inferência de volta para os P-Cores de alta performance (0-3) + E-cores (todos).
        Objetivo: Geração em velocidade nominal máxima.
        """
        print(f"🚀 [ACTUATOR->CPU] NOMINAL: Liberando todos os núcleos de P-Core (0-3) para inferência máxima.")
        try:
            # taskset -cp 0-11 PID (todos os cores)
            subprocess.run(["taskset", "-cp", "0-11", str(self.gemma_pid)], check=True, capture_output=True)
            print("⚡ [ACTUATOR->CPU] Força Máxima Liberada: Barramento AVX2 ativo nos P-Cores.")
        except Exception as e:
            print(f"⚠️ [ACTUATOR->CPU] Falha na liberação de afinidade: {e}")

    def purge_system_caches(self):
        """
        Executa uma limpeza agressiva nos caches do kernel Linux Debian.
        Objetivo: Liberar RAM de forma imediata para mitigar eventos de OOM.
        """
        print("🧹 [ACTUATOR->RAM] ESTRESSE DE MEMÓRIA: Executando purge de drop_caches no Kernel.")
        try:
            cmd = "echo 3 | sudo tee /proc/sys/vm/drop_caches"
            res = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            print(f"🧼 [ACTUATOR->RAM] Caches do Kernel limpos. Resultado: {res.stdout.strip()}")
        except Exception as e:
            print(f"⚠️ [ACTUATOR->RAM] Falha ao purgar caches do sistema: {e}")
