SWA REST service (systemd)
===========================

This folder contains helper files to run the legacy `swa.rest_server` Uvicorn app as a service.

> [!WARNING]
> O serviço `swa-rest.service` está depreciado. A arquitetura atual SWA Aeon utiliza `swa-aeonc.service` para expor a API OpenAI na porta `20000` e `swa-auto.service` para rodar o loop OODA e o servidor de controle na porta `20001`.

Files
- `swa-rest.service`: systemd unit file (template uses absolute paths pointing to the project)
- `run_server.sh`: simple start/stop/restart helper that uses the project's `venv`

Install as systemd service (requires root)

1. Copy unit to systemd:

```bash
sudo cp deployments/swa-rest.service /etc/systemd/system/swa-rest.service
sudo systemctl daemon-reload
sudo systemctl enable --now swa-rest.service
sudo journalctl -u swa-rest.service -f
```

Note: the unit's `User=` field is set to `%i` so you can start the service as your user, e.g.:

```bash
# start as current user
sudo systemctl enable --now swa-rest.service@$(whoami)
```

Local helper

You can also run the included helper locally without systemd:

```bash
chmod +x deployments/run_server.sh
./deployments/run_server.sh start
tail -f server.log
```

Ports and binding
- Comunicação entre serviços fica no host local: API interna/OODA `127.0.0.1:20001`, MCP `127.0.0.1:8000`, eventos UDP `127.0.0.1:20005` e córtex auditivo `127.0.0.1:9200`.
- Chat é a interface para a LAN, não para a Internet. O servidor canônico é `aeonc.py`; inicie com `python aeonc.py --api --host <IP_LAN_DO_HOST> --port 20000 --compile --tensors`. O padrão de `--host` é `127.0.0.1`; no host, não use `0.0.0.0`.
- O Compose deste repositório define o serviço legado `swa-rest`, não inicia o AeonC canônico (`aeonc.py`). Sua porta publicada `20000` fica local por padrão; a publicação no host pode ser limitada a um IP LAN específico via `SWA_CHAT_BIND_IP`, mas isso não substitui configurar e iniciar o servidor correto. O listener `0.0.0.0` dentro do container é apenas para a rede privada do container.
- O chat via LAN exige permitir `20000/TCP` apenas para a sub-rede confiável no firewall. A opção `--host` é o IP da interface LAN do Aeon, não o endereço dos clientes.
- O único dono canônico de `TCP 9300` é `swa-tts.service`, que inicia `/services/voice-stream-client/server.py`. O AeonC envia fala ao serviço REST TTS via `POST /api/v1/speak`. Configure `SWA_TTS_REST_URL` no ambiente do processo AeonC para apontar ao receptor; o padrão é local:

  ```ini
  [Service]
  Environment="SWA_TTS_REST_URL=http://<IP_LAN_DO_RECEPTOR>:9300/api/v1/speak"
  ```

  Coloque essa variável na unidade que inicia `aeonc.py` (ou exporte-a antes de iniciar AeonC) e permita HTTP `9300/TCP` apenas do host Aeon ao receptor TTS.
- Em WSL `mirrored`, não mantenha um `netsh interface portproxy` na mesma porta do listener Linux: o proxy pode reservar `9300` no namespace compartilhado e causar `EADDRINUSE`. Use um único dono para a porta, confira o IP de interface atual e restrinja o acesso LAN no firewall ativo.
- `swa-orquestrador.service` consome UDP `9100` local e encaminha fala ao mesmo endpoint REST `9300`; isso é um transporte interno diferente, não outro sintetizador.
- Não execute `run.py` junto das unidades systemd. O launcher não deve mais matar processos em `9300`; por padrão ele respeita as unidades instaladas e omite subprocessos já gerenciados. `--standalone` só é permitido se nenhuma unidade equivalente estiver ativa.
- `controllers/receiverController.py` é a biblioteca de síntese Piper compartilhada pelo servidor canônico. Seu servidor HTTP standalone foi relegado à porta local `9301`; não o use como segundo dono de `9300`.
- Receptores opcionais de telemetria (`TOKEN_COLLECTOR_HOST`) e gravidade (`scripts/udp_gravity_receptor.py --host`) também usam loopback por padrão. Só configure um IP de interface específico se um emissor em outro host realmente precisar deles.

Security
- A API de chat não possui autenticação própria. Permita a porta `20000` somente a partir da LAN confiável no firewall; não encaminhe essa porta à Internet.
- Mantenha portas de controle, MCP, Redis, PostgreSQL e UDP interno sem publicação no host/rede. Para Docker, use a rede privada do Compose para comunicação entre containers e publique apenas a interface de chat.
- A rota de fala do AeonC usa HTTP `9300`; permita a saída somente para o IP LAN do receptor TTS. Não publique a porta UDP interna `9100` nem a API legada `9301`.
