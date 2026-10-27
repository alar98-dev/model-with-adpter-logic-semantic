import urllib.request, json, concurrent.futures

ips = [f'10.0.0.{i}' for i in range(1, 255)] + ['192.168.10.107']

def test_8800(ip):
    url = f'http://{ip}:8800/api/posting-status'
    try:
        req = urllib.request.urlopen(url, timeout=0.5)
        if req.status == 200:
            return ip
    except:
        pass
    return None

with concurrent.futures.ThreadPoolExecutor(max_workers=50) as ex:
    results = [r for r in ex.map(test_8800, ips) if r]

print("=== IP DO TERMUX/CHANNELHUB WEB (PORTA 8800) ===")
print("IPs ativos na porta 8800:", results)
