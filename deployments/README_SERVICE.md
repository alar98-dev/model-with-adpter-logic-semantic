SWA REST service (systemd)
===========================

This folder contains helper files to run the `swa.rest_server` Uvicorn app as a service.

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
- The service binds to `0.0.0.0:20000` by default. Adjust the unit or script if you need another port.

Security
- For production, consider running behind a reverse proxy, set up proper user privileges, and secure access to the port.
