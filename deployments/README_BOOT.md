# Boot systemd — SWA Aeon (arquitetura v2)

## Ordem de inicialização (8 Unidades do Sistema)

```text
postgresql
    → swa-aeonc.service (Core Gemma-3-1B-it na porta 20000)
    → swa-mcp.service (Starlette MCP Gateway na porta 8000)
    → swa-subcortex.service (Sensorial Gemma-3-270m)
    → swa-manifold.service (Oneshot: rebuild SHM + pgvector)
    → swa-auto.service (OODA Loop na porta 20001, consome tensor do core)
    → swa-auditory.service (Sensor auditivo)
    → swa-obelisk.service (Watchdog cognitivo Gemma-3-270m+LoRA)
        → swa-obelisk-verifier.service (Watchdog bash de integridade/liveness)
```

Ou habilitar o alvo grupo:

```bash
sudo systemctl enable swa.target
```

## Instalar / atualizar units no Debian

```bash
cd /services/model-with-adpter-logic-semantic
sudo bash deployments/install_swa_systemd.sh
```

## Variáveis críticas (`swa-auto.service`)

| Variável | Valor | Função |
|----------|-------|--------|
| `MODEL_DIR` | `models/gemma-3-1b-it` | Linguagem |
| `VECTORIZER_MODEL_DIR` | `models/gemma-3-270m` | Embeddings reais |
| `SWA_REBUILD_MANIFOLD_ON_BOOT` | `0` | Rebuild feito por `swa-manifold.service` |
| `SWA_REAL_EMBEDDINGS` | `1` | Ativa pipeline 270M→2560D |
| `SWA_MAX_TRACKED_FILES` | `400` | Cap watcher (anti-EMFILE) |
| `LimitNOFILE` | `8192` | Limite de FD |

## Verificação pós-boot

```bash
systemctl status swa-aeonc swa-mcp swa-subcortex swa-manifold swa-auto swa-auditory swa-obelisk swa-obelisk-verifier
cat /dev/shm/swa_vector_manifold/manifest.json | grep embedding_source
journalctl -u swa-manifold -u swa-auto -b --no-pager | tail -40
```

## Rebuild manual do manifold

```bash
./venv/bin/python3 scripts/rebuild_vector_reality.py
```

## Limites de rede

- Serviços internos (OODA/REST `20001`, MCP `8000`, UDP interno `20005`, córtex auditivo `9200`) escutam apenas em `127.0.0.1`.
- Chat na LAN usa `python aeonc.py --api --host <IP_LAN_DO_HOST> --port 20000 --compile --tensors`; o padrão de `--host` é loopback. Não vincule o host em `0.0.0.0` e não encaminhe a porta para a Internet.
- O TTS REST é um receptor separado na LAN em `POST /api/v1/speak`. Configure `SWA_TTS_REST_URL=http://<IP_LAN_DO_RECEPTOR>:9300/api/v1/speak` no ambiente do processo AeonC; o padrão é `http://127.0.0.1:9300/api/v1/speak`. O legado UDP `9100` continua local e é uma interface diferente.
- `swa-tts.service` é o único proprietário da API canônica `9300`; não inicie `server.py` via `run.py` ao mesmo tempo. O servidor HTTP legado de `receiverController.py`, quando explicitamente iniciado, usa loopback `9301`.
- A API de chat não tem autenticação própria; aplique regra de firewall para aceitar somente clientes da LAN confiável. Veja `docs/api.md`.
- Receptores opcionais de telemetria/gravidade permanecem em loopback por padrão; habilite acesso em outro host apenas vinculando ao IP de interface necessário.
