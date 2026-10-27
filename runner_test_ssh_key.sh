#!/bin/bash
echo "=== TESTANDO SSH KEY ED25519 PARA TERMUX (192.168.10.107:8022) ==="
ssh -p 8022 -i /home/alifer/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o ConnectTimeout=3 192.168.10.107 "uptime; python3 -c 'import psycopg2; conn=psycopg2.connect(\"dbname=crm_hub user=termux host=127.0.0.1\"); print(\"🟢 PSQL TERMUX CONECTADO DENTRO DO DISPOSITIVO!\")'"
