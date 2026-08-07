# SWA Aeon — TODO Geral

## Pendências Lógicas e Multimodais
- [ ] Manter/ajustar a materialização para evitar parâmetros em `meta` e garantir que embeddings estejam materializadas (CPU/bf16) no fallback.
- [ ] Executar `scripts/validate_generation_pipeline.py` e confirmar que `MISSING`/"Some parameters are on the meta device" não aparecem mais para a Vision Tower.

## Infraestrutura e Monitoramento (Pós-Purificação)
- [x] Correção de colisão de portas: Mover orquestrador OODA (run.py) da porta 20000 para 20001 e dedicar a 20000 ao `aeonc.py`.
- [x] Auditoria de toda a documentação markdown contra o ecossistema de 8 serviços.
- [x] Sistema de backups seguro em disco com assinaturas SHA256 (`/var/lib/swa/backups/mutations` e PostgreSQL).
- [ ] Monitorar o primeiro disparo automático diário do `swa-backup.timer` (agendado para as 03:00).
- [ ] Monitorar a estabilidade da assinatura tensorial blake3 em ciclo de geração ativo real.
- [ ] Validar a concorrência e o consumo de swap/RAM do host sob execução simultânea do core + loops OODA.
