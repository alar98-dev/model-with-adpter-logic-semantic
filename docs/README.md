# Documentação do AEON

Os documentos operacionais atuais estão centralizados na origem canônica:

- `/services/aeon-gateway-rs/CANONICAL_ARCHITECTURE.md`
- `/services/aeon-gateway-rs/PREFIX_CONTRACT_V1.md`
- `../DOCUMENTATION_STATUS.md`
- `../OPENCODE_AEON_INTEGRATION.md`
- `/services/aeon-gateway-rs/ARCHITECTURE.md`
- `/services/agent-dev/README.md`

O runtime vigente usa o gateway Rust na porta `8050`, Córtex Gemma 3 1B na
porta `20003` e Subcórtex Gemma 3 270M na porta `20002`. A porta `20000` e o
servidor `aeonc.py` descritos em relatórios antigos não são o caminho canônico
atual.

Relatórios históricos permanecem neste diretório como registro de evolução,
mas não devem ser usados como procedimento de inicialização.
