import urllib.request, json

base = 'http://192.168.10.107:8800'
try:
    req = urllib.request.Request(base + '/api/blockages/auto-resolve', data=b'{}', headers={'Content-Type': 'application/json'}, method='POST')
    res = urllib.request.urlopen(req, timeout=5)
    print("Auto-resolve response:", res.read().decode('utf-8'))
except Exception as e:
    print("Auto-resolve error:", e)
