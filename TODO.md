## TODO
- [x] Reforçar o remapeamento de chaves no `swa/gemma_integration.py` para remover definitivamente o nível intermediário `vision_model.` quando o caminho contém `vision_tower`.
- [ ] Manter/ajustar a materialização para evitar parâmetros em `meta` e garantir que embeddings estejam materializadas (CPU/bf16) no fallback.
- [ ] Executar `scripts/validate_generation_pipeline.py` e confirmar que `MISSING`/"Some parameters are on the meta device" não aparecem mais para a Vision Tower.


