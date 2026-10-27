import urllib.request, json, re, sys

# Fazer chamada HTTP POST para o endpoint do MCP SSE no Termux
def exec_termux_cmd(cmd):
    # Enviar para o servidor MCP SSE
    import urllib.parse
    url = 'http://192.168.10.107:8000/sse'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'PythonBridge'})
        with urllib.request.urlopen(req, timeout=3) as resp:
            # ler a primeira linha do SSE para pegar o endpoint de mensagens
            lines = []
            for _ in range(5):
                line = resp.readline().decode('utf-8').strip()
                if line.startswith('event: endpoint'):
                    next_line = resp.readline().decode('utf-8').strip()
                    if next_line.startswith('data: '):
                        post_endpoint = next_line[6:]
                        full_post_url = 'http://192.168.10.107:8000' + post_endpoint
                        
                        # Fazer chamada de inicializacao JSON-RPC
                        init_payload = json.dumps({
                            "jsonrpc": "2.0", "id": 1, "method": "initialize",
                            "params": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {},
                                "clientInfo": {"name": "wsl-client", "version": "1.0"}
                            }
                        }).encode('utf-8')
                        
                        req_init = urllib.request.Request(full_post_url, data=init_payload, headers={'Content-Type': 'application/json'})
                        urllib.request.urlopen(req_init, timeout=3)

                        # Fazer chamada para executar_comando_termux
                        exec_payload = json.dumps({
                            "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                            "params": {
                                "name": "executar_comando_termux",
                                "arguments": {"comando": cmd}
                            }
                        }).encode('utf-8')
                        
                        req_exec = urllib.request.Request(full_post_url, data=exec_payload, headers={'Content-Type': 'application/json'})
                        with urllib.request.urlopen(req_exec, timeout=10) as exec_resp:
                            print("=== RESPOSTA DO TERMUX ===")
                            for _ in range(10):
                                l = resp.readline().decode('utf-8').strip()
                                if 'result' in l:
                                    print(l)
                                    break
                        return
    except Exception as e:
        print(f"Erro ao conectar via SSE: {e}")

cmd = '''python3 -c "
import psycopg2

conn = psycopg2.connect(dbname='crm_hub', user='termux', host='127.0.0.1', port=5432)
cur = conn.cursor()

print('=== PRODUTOS CANAL 2 (GAMES CLUTCH) ===')
cur.execute('SELECT id, name, price, affiliate_url, active, channel_id FROM shop_products WHERE channel_id=2 ORDER BY id;')
prods = cur.fetchall()
for p in prods:
    print(f'Prod #{p[0]}: {p[1]} | R$ {p[2]} | Active: {p[4]}')
    print(f'   URL: {p[3]}')

print(f'Total produtos ativos canal 2: {len(prods)}')

cur.close()
conn.close()
"'''

exec_termux_cmd(cmd)
