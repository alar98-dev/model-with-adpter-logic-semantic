import urllib.request, json

# 1. Consultar a API do ChannelHub para verificar Canal 2
req = urllib.request.urlopen('http://192.168.10.107:8800/api/posting-status', timeout=5)
data = json.loads(req.read().decode('utf-8'))
ch2_info = [c for c in data.get('channels', []) if c.get('id') == 2]
print("=== INFO CANAL 2 (GAMES CLUTCH) ===")
print(json.dumps(ch2_info, indent=2, ensure_ascii=False))
