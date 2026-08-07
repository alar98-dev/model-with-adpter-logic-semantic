import time
import os
import subprocess
import shutil

print("🗿 [OBELISCO] Sistema Imunológico Estrutural (DNA Watchdog) Iniciado.")

BACKUP_DIR = "/dev/shm/aeon_mutation_backups"

def check_service_status():
    """Retorna True se o serviço principal estiver ativo e saudável."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", "swa-auto.service"],
            capture_output=True, text=True
        )
        status = result.stdout.strip()
        return status == "active"
    except Exception:
        return False

def check_for_lethal_mutation():
    """Verifica se o serviço está morto. Se sim, aguarda um pouco para confirmar que não é apenas um reinício normal."""
    if not check_service_status():
        print("⚠️ [OBELISCO] Queda do Córtex detectada! Aguardando 12 segundos para confirmar Falha Letal...")
        time.sleep(12)
        if not check_service_status():
            return True # Confirmado. O serviço não conseguiu voltar.
    return False

def apply_genetic_rollback():
    """Restaura todos os backups de DNA feitos na última mutação."""
    if not os.path.exists(BACKUP_DIR):
        print("ℹ️ [OBELISCO] Nenhum backup genético encontrado. Queda pode ser por outro motivo.")
        return False

    arquivos_restaurados = 0
    for filename in os.listdir(BACKUP_DIR):
        if filename.endswith(".snapshot"):
            # O nome do arquivo no backup codifica o caminho original (substituindo '/' por '%')
            # Ex: %services%model-with...%run.py.snapshot
            encoded_name = filename[:-9] # remove .snapshot
            original_path = encoded_name.replace("%", "/")
            
            backup_path = os.path.join(BACKUP_DIR, filename)
            
            try:
                shutil.copy2(backup_path, original_path)
                print(f"🧬 [OBELISCO] ROLLBACK EXECUTADO: {original_path} restaurado para o estado saudável pré-mutação.")
                arquivos_restaurados += 1
            except Exception as e:
                print(f"❌ [OBELISCO] Erro crítico ao restaurar {original_path}: {e}")
                
    if arquivos_restaurados > 0:
        # Limpa os backups após o rollback
        shutil.rmtree(BACKUP_DIR)
        return True
    return False

def main_loop():
    while True:
        try:
            if check_for_lethal_mutation():
                print("🚨 [OBELISCO] FALHA LETAL DETECTADA NO DNA! Iniciando protocolo de ressuscitação celular...")
                if apply_genetic_rollback():
                    print("⚡ [OBELISCO] Reanimando o Córtex...")
                    subprocess.run(["systemctl", "restart", "swa-auto.service"])
                    print("✅ [OBELISCO] Desfibrilador aplicado. Monitoramento retomado.")
                else:
                    print("⚠️ [OBELISCO] Queda confirmada, mas sem backups para rollback. Apenas reiniciando...")
                    subprocess.run(["systemctl", "restart", "swa-auto.service"])
            
            time.sleep(5) # Checagem a cada 5 segundos
        except Exception as e:
            print(f"❌ [OBELISCO] Erro no Watchdog: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main_loop()
