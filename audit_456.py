import urllib.request, json

base = 'http://192.168.10.107:8800'

print("=== AUDITORIA COMPLETA DOS CANAIS RESTANTES (4, 5 e 6) ===")

# 1. Status das Filas no ChannelHub
req = urllib.request.urlopen(base + '/api/posting-status', timeout=5)
data = json.loads(req.read().decode('utf-8'))
channels = data.get('channels', [])

for ch_id in [4, 5, 6]:
    ch_info = [c for c in channels if c.get('id') == ch_id]
    print(f"\n--- CANAL {ch_id} ---")
    if ch_info:
        info = ch_info[0]
        print(f"Nome: {info.get('name')} | YT ID: {info.get('yt_channel_id')}")
        print(f"Postados: {info.get('posted')} | Fila com vídeo: {info.get('pending_with_video')} | Fila sem vídeo: {info.get('pending_no_video')}")
        print(f"Próximo agendado: {info.get('next_scheduled')} - Título: {info.get('next_title')}")
    
    # 2. Afiliados e Posição do Card
    try:
        req_af = urllib.request.urlopen(f"{base}/afiliado/data/{ch_id}", timeout=3)
        af_data = json.loads(req_af.read().decode('utf-8'))
        print(f"Posição do Card Overlay: {af_data.get('position')}")
        print(f"Produtos configurados: {len(af_data.get('products', []))}")
        for p in af_data.get('products', []):
            print(f"   - {p.get('name')}: {p.get('url')[:80]}...")
    except Exception as e:
        print(f"Erro ao obter dados de afiliado canal {ch_id}: {e}")
