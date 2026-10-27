import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

TERMUX_URL = "http://192.168.10.107:8000/sse"

async def run():
    print(f"Conectando ao Termux MCP em {TERMUX_URL} a partir do WSL...")
    async with sse_client(TERMUX_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            cmd = '''python3 -c "
import psycopg2

conn = psycopg2.connect(dbname='crm_hub', user='termux', host='127.0.0.1', port=5432)
cur = conn.cursor()

print('=== 1. CANAL 2 (GAMES CLUTCH) DETALHES ===')
cur.execute('SELECT id, name, yt_channel_id, custom_url, subscriber_count, video_count FROM channels WHERE id=2;')
for ch in cur.fetchall():
    print(f'ID {ch[0]}: {ch[1]} | YT: {ch[2]} | Handle: {ch[3]} | Subs: {ch[4]} | Videos: {ch[5]}')

print('\n=== 2. PRODUTOS VINCULADOS AO CANAL 2 ===')
cur.execute('SELECT id, name, price, affiliate_url, active, channel_id FROM shop_products WHERE channel_id=2 ORDER BY id;')
prods = cur.fetchall()
for p in prods:
    print(f'Prod #{p[0]}: {p[1]} | R$ {p[2]} | Active: {p[4]}')
    print(f'   URL: {p[3]}')

print(f'Total produtos canal 2: {len(prods)}')

print('\n=== 3. CONFIGURAÇÃO DE OVERLAY / BINDING CANAL 2 ===')
cur.execute(\"SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name='channel_affiliate_bindings';\")
if cur.fetchone():
    cur.execute('SELECT * FROM channel_affiliate_bindings WHERE channel_id=2;')
    print('Bindings Ch2:', cur.fetchall())
else:
    print('Tabela channel_affiliate_bindings nao existe ainda')

print('\n=== 4. ÚLTIMOS 5 POSTS CANAL 2 ===')
cur.execute('SELECT id, title, status, created_at FROM posts WHERE channel_id=2 ORDER BY id DESC LIMIT 5;')
for pst in cur.fetchall():
    print(f'Post #{pst[0]} | Status: {pst[2]} | Título: {pst[1]}')

cur.close()
conn.close()
"'''
            res = await session.call_tool("executar_comando_termux", {"comando": cmd})
            print(res.content[0].text)

if __name__ == "__main__":
    asyncio.run(run())
