# Docker & Model Setup for Gemma 1B Test
Status: Completed (Migrated to Systemd Native Production)

## Steps:

### 1. [✅ DONE] Replace Dockerfile with optimized version (python:3.10-slim, git, port 20000)
### 2. [✅ DONE] Replace docker-compose.yml with user's config (swa-server, volumes .:/app + models, MODEL_DIR env)
### 3. [✅ DONE] Create .env.example for HF_TOKEN
### 4. [✅ DONE] Make scripts/clone_gemma.sh idempotent and executable
### 5. [✅ DONE] Migrado para systemd nativo (swa.target com 8 unidades)
### 6. [✅ DONE] Testado e validado (aeonc.py na 20000 e run.py na 20001)
### 7. [✅ DONE] Update original TODO.md tests & attempt_completion

> [!NOTE]
> O ecossistema SWA Aeon foi migrado do Docker-Compose para execução nativa via systemd no WSL2 para otimizar o uso de RAM/CPU e evitar colisões de portas e VRAM. O guia de execução via systemd está documentado em `deployments/README_BOOT.md`.
