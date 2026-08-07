import os
import sys

DEVICE_PATH = "/dev/sde"


def main() -> int:
    print(f"📊 Validando {DEVICE_PATH}...")

    if not os.path.exists(DEVICE_PATH):
        print(f"❌ Falha: {DEVICE_PATH} não existe.")
        return 1

    try:
        # Try read-only first
        with open(DEVICE_PATH, "rb") as f:
            f.seek(0)
            head = f.read(16)
            print(f"✅ OK: {DEVICE_PATH} acessível. Header(hex)={head.hex()}")

        # Try seek to the first expected offset area
        with open(DEVICE_PATH, "rb") as f:
            f.seek(0x0000)
            head2 = f.read(16)
            print(f"🧪 OK: offset 0x0000 lido. Header(hex)={head2.hex()}")

    except PermissionError:
        print("🔒 Permissão negada. Rode com sudo ou ajuste permissions/udev.")
        return 2
    except OSError as e:
        print(f"❌ Falha ao abrir/ler {DEVICE_PATH}: {e}")
        return 3

    print("🚀 Validação concluída.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

