#!/bin/bash
echo "=== TESTANDO SSH E PGPASSWORD ==="
python3 -c "
import psycopg2
for passw in ['', 'termux', '123456', 'postgres', 'root']:
    try:
        conn = psycopg2.connect(dbname='crm_hub', user='termux', password=passw, host='192.168.10.107', port=5432)
        print(f'🟢 CONECTADO COM SUCESSO! Password: "{passw}"')
        conn.close()
        break
    except Exception as e:
        print(f'Pass "{passw}" falhou: {e}')
"
