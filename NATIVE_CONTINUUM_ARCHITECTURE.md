# SWA-Aeon: Arquitetura de Inferência Nativa & Canal Tensorial

Esta documentação descreve a transição do pipeline legado em CPython/Transformers para o pipeline nativo compilado de alta performance.

---

## 1. Desativação do Pipeline Legado
* O serviço `swa-aeonc.service` (Python + PyTorch em CPU consumindo ~4 GB RAM com risco de SIGBUS) foi desativado.
* A inferência foi desacoplada em duas frentes complementares:
  1. **Álgebra Tensorial Pura (C + Rust)**: Execução local em registradores SIMD via `aeon-ooda-rs`.
  2. **Geração de Texto Simbólica (C++ nativo)**: `llama-server` rodando na porta 20002 consumindo ~110 MB RAM.

---

## 2. Canal Tensorial (`/dev/shm/aeon/tensor_bus.bin`)
* Formato: Estrutura contígua Rust/C (`struct TensorPacket`) de 2688 bytes alinhada a 64 bytes.
* Cabeçalho: 68 bytes (`magic`, versão, sequência, fase, métricas e dimensão); os 60 bytes restantes são padding no final.
* Vetor Latente: 640 floats de 32 bits (2560 bytes) gerados via instruções AVX2/FMA com RMSNorm (RMS≈1; norma Euclidiana≈√640).
* Leitura em Python:
  ```python
  import numpy as np
  with open("/dev/shm/aeon/tensor_bus.bin", "rb") as f:
      vector = np.frombuffer(f.read(), dtype=np.float32, count=640, offset=68)
  ```

---

## 3. Lar Vetorial (`/dev/shm/aeon/vector_lair.bin`)
* 60 nichos correspondentes aos 60 estados cíclicos da Roda.
* 153.624 bytes alocados em memória compartilhada e espelhados no disco ext4 em `/services/agent-dev/data/vector_lair.bin`.
* Sincronizados com o MongoDB (`vector_sanctum`) e PostgreSQL (`vector_sanctum` com `pgvector` 640D).

---

## 4. Gerenciamento de Serviços
```bash
# Iniciar / Parar serviços do ecossistema
sudo systemctl restart swa-native-llm.service
sudo systemctl restart swa-continuum.service
sudo systemctl restart agent-dev.service
sudo systemctl restart mongod.service
```
