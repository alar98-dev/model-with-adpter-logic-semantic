import urllib.request, json

base = 'http://192.168.10.107:8800'
for ep in ['/channels/2', '/afiliado/data/2', '/afiliado/preview/2']:
    try:
        req = urllib.request.urlopen(base + ep, timeout=3)
        print(f"=== {ep} ===")
        print(req.read().decode('utf-8')[:500])
    except Exception as e:
        print(f"Erro em {ep}: {e}")
