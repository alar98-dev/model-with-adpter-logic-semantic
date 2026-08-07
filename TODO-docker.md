# Docker & Model Setup for Gemma 1B Test
Status: In Progress [BLACKBOXAI]

## Steps:

### 1. [✅ DONE] Replace Dockerfile with optimized version (python:3.10-slim, git, port 20000)
### 2. [✅ DONE] Replace docker-compose.yml with user's config (swa-server, volumes .:/app + models, MODEL_DIR env)
### 3. [✅ DONE] Create .env.example for HF_TOKEN
### 4. [✅ DONE] Make scripts/clone_gemma.sh idempotent and executable
### 5. [PENDING] execute docker-compose up --build -d
### 6. [PENDING] Test curl http://localhost:20000/api/v1/chat with Gemma 1B
### 7. [DONE] Update original TODO.md tests ✅ & attempt_completion

Next step? Will update after each.
