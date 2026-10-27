import urllib.request, json

try:
    req = urllib.request.urlopen('http://192.168.10.107:8800/openapi.json', timeout=5)
    spec = json.loads(req.read().decode('utf-8'))
    print("=== ROTAS DA API DO CHANNELHUB (PORTA 8800) ===")
    paths = spec.get('paths', {})
    for path, methods in paths.items():
        for m in methods.keys():
            print(f"  • {m.upper()} {path}")
except Exception as e:
    print(f"Erro ao obter openapi.json: {e}")
