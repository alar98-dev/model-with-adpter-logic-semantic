import requests
try:
    r = requests.post('http://127.0.0.1:20000/v1/chat/completions', json={'messages': [{'role': 'user', 'content': 'Ping'}], 'max_tokens': 10}, timeout=10)
    print(r.status_code, r.text)
except Exception as e:
    print('Error:', e)
