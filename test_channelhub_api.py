import urllib.request, json

def test_api():
    base = 'http://192.168.10.107:8800'
    endpoints = [
        '/api/posting-status',
        '/api/channels',
        '/api/posts?channel_id=2',
        '/api/products?channel_id=2',
        '/api/affiliates/bindings'
    ]
    for ep in endpoints:
        try:
            req = urllib.request.urlopen(base + ep, timeout=3)
            data = json.loads(req.read().decode('utf-8'))
            print(f"🟢 GET {ep}: HTTP 200")
            print("   Data:", json.dumps(data)[:300])
        except Exception as e:
            print(f"❌ GET {ep}: {e}")

if __name__ == '__main__':
    test_api()
