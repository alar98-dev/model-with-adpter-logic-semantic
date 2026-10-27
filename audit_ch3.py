import urllib.request, json

base = 'http://192.168.10.107:8800'

print("=== 1. STATUS DA FILA E CANAL 3 (LkOut Clips & Insights) ===")
req = urllib.request.urlopen(base + '/api/posting-status', timeout=5)
data = json.loads(req.read().decode('utf-8'))
ch3_info = [c for c in data.get('channels', []) if c.get('id') == 3]
print(json.dumps(ch3_info, indent=2, ensure_ascii=False))

print("\n=== 2. DADOS DE AFILIADO E CARD OVERLAY DO CANAL 3 ===")
try:
    req_af = urllib.request.urlopen(base + '/afiliado/data/3', timeout=5)
    af_data = json.loads(req_af.read().decode('utf-8'))
    print(json.dumps(af_data, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"Erro ao obter afiliado/data/3: {e}")
