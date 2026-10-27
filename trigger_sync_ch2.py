import urllib.request, json

req = urllib.request.Request('http://192.168.10.107:8800/channels/2/sync', data=b'', headers={'Content-Type': 'application/json'}, method='POST')
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        print("=== RESULTADO SYNC CANAL 2 ===")
        print(resp.read().decode('utf-8'))
except Exception as e:
    print(f"Erro no sync: {e}")
