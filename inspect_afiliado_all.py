import urllib.request, json

base = 'http://192.168.10.107:8800'

for ch_id in [1, 2, 3]:
    try:
        req = urllib.request.urlopen(f'{base}/afiliado/data/{ch_id}', timeout=3)
        data = json.loads(req.read().decode('utf-8'))
        print(f"=== CANAL {ch_id} ({data.get('channel',{}).get('name')}) ===")
        print("Posição Card:", data.get('position'))
        print("Produtos:", json.dumps(data.get('products'), indent=2, ensure_ascii=False))
        print("-" * 50)
    except Exception as e:
        print(f"Erro canal {ch_id}: {e}")
