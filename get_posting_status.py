import urllib.request, json

req = urllib.request.urlopen('http://192.168.10.107:8800/api/posting-status', timeout=5)
data = json.loads(req.read().decode('utf-8'))
print(json.dumps(data, indent=2, ensure_ascii=False))
