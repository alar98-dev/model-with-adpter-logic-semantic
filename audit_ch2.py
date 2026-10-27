import psycopg2

def audit():
    conn = None
    for user, db in [('alifer', 'crm_hub'), ('postgres', 'crm_hub'), ('termux', 'crm_hub')]:
        try:
            conn = psycopg2.connect(dbname=db, user=user, host='127.0.0.1', port=5432)
            print(f"Conectado com user={user} db={db}")
            break
        except Exception as e:
            pass

    if not conn:
        print("Não foi possível conectar ao PostgreSQL")
        return

    cur = conn.cursor()
    print("=== CANAL 2: GAMES CLUTCH ===")
    cur.execute("SELECT id, name, yt_channel_id, custom_url, subscriber_count, video_count FROM channels WHERE id=2 OR name ILIKE '%games%';")
    for ch in cur.fetchall():
        print(f"Channel ID {ch[0]}: {ch[1]} | YT: {ch[2]} | Handle: {ch[3]} | Subs: {ch[4]} | Videos: {ch[5]}")

    print("\n=== PRODUTOS DO CANAL 2 OU GERAIS ===")
    cur.execute("SELECT id, name, price, affiliate_url, active, channel_id FROM shop_products WHERE channel_id=2 OR channel_id IS NULL ORDER BY id;")
    for p in cur.fetchall():
        print(f"Prod #{p[0]} [Ch {p[5]}]: {p[1]} | R$ {p[2]} | Active: {p[4]}")
        print(f"   URL: {p[3]}")

    print("\n=== BINDINGS OVERLAY ===")
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_name='channel_affiliate_bindings';")
    if cur.fetchone():
        cur.execute("SELECT * FROM channel_affiliate_bindings WHERE channel_id=2;")
        print("Bindings Ch2:", cur.fetchall())

    print("\n=== POSTS CANAL 2 ===")
    cur.execute("SELECT id, title, status, created_at FROM posts WHERE channel_id=2 ORDER BY id DESC LIMIT 10;")
    for pst in cur.fetchall():
        print(f"Post #{pst[0]} | Status: {pst[2]} | Título: {pst[1]}")

    cur.close()
    conn.close()

if __name__ == '__main__':
    audit()
