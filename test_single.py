import urllib.request, json

url = 'http://192.168.10.107:8800/api/posting-status'
try:
    req = urllib.request.urlopen(url, timeout=5)
    data = json.loads(req.read().decode('utf-8'))
    print("🟢 HTTP 200 OK! Status:")
    print(json.dumps(data.get('totals'), indent=2))
except Exception as e:
    print(f"❌ Erro 192.168.10.107:8800 -> {e}")
