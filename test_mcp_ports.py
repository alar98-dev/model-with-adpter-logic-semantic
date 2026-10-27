import urllib.request

for p in [8000, 8800]:
    url = f'http://192.168.10.107:{p}/sse' if p==8000 else f'http://192.168.10.107:{p}/api/posting-status'
    try:
        r = urllib.request.urlopen(url, timeout=2)
        print(f"Porta {p} respondeu com HTTP {r.status}")
    except Exception as e:
        print(f"Porta {p} erro: {e}")
