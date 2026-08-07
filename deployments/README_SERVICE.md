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
- O núcleo (`swa-aeonc.service`) escuta na porta `20000` por padrão.
- O orquestrador autônomo (`swa-auto.service`) escuta na porta `20001` por padrão para evitar colisão de portas.
- O gateway MCP (`swa-mcp.service`) escuta na porta `8000` por padrão.

Security
- For production, consider running behind a reverse proxy, set up proper user privileges, and secure access to the port.
