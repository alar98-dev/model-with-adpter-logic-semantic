# Status Infraestrutura & Troubleshooting — Gemma3-4B + SWA

## Validação Drivers NVIDIA (GPU Opc.)

```bash
nvidia-smi  # Deve mostrar GPU, VRAM 6GB+
# Ex: Driver 535+, CUDA 12.x
```

**Erro comum:** `no CUDA-capable device`. Sol: Instale drivers NVIDIA.

## Docker NVIDIA Toolkit

```bash
# Instalar runtime NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install nvidia-docker2
sudo systemctl restart docker

# Teste
docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi
```

**docker-compose.yml:** `--gpus all` auto se toolkit instalado.

## Fallback VRAM 6GB Excedida

1. **Single Gemma3-4B (RAM):** `GEMMA_DEVICE=cpu`, float16 (~8GB).
2. **Desabilite dual:** `USE_DUAL_MODEL=false`.
3. **Low-mem:** `low_cpu_mem_usage=True` (padrão).
4. **Monitor:** `docker stats` ou `htop`.

**Tune Gemma:**
```python
torch_dtype=torch.float16  # ou bfloat16
torch.backends.cuda.enable_mem_efficient_sdp(False)
```

## Comandos Diagnóstico
```bash
# Logs Docker
docker-compose logs -f swa-server

# Self-check API
curl http://localhost:20000/api/v1/utils/self_check

# Memória container
docker stats swa_logic_engine
```

## Procedimento Fallback
1. GPU fail → `GEMMA_DEVICE=cpu` + float16.
2. RAM tight → `USE_FAKE=true` dev.
3. OOM → Aumente `mem_limit: 24g`.

**Ambiente:** Linux 6.12, Docker Compose v3.8.

---
*Gemma3-4B Infra Guide (2024).*

