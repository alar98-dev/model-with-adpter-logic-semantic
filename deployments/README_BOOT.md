# Boot systemd — SWA Aeon (arquitetura v2)

## Ordem de inicialização

```text
postgresql
    → swa-mcp.service
    → swa-subcortex.service (Gemma 270M, processo separado)
    → swa-manifold.service (oneshot: rebuild SHM + pgvector, embeddings reais)
    → swa-auto.service (Gemma 1B, loop OODA)
    → swa-auditory.service
    → swa-obelisk.service (watchdog, root)
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
systemctl status swa-manifold swa-auto swa-subcortex swa-mcp
cat /dev/shm/swa_vector_manifold/manifest.json | grep embedding_source
journalctl -u swa-manifold -u swa-auto -b --no-pager | tail -40
```

## Rebuild manual do manifold

```bash
./venv/bin/python3 scripts/rebuild_vector_reality.py
```
