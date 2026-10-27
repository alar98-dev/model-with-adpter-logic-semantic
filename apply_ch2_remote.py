import psycopg2

def setup_games_clutch():
    print("Conectando ao PostgreSQL do Termux em 192.168.10.107:5432...")
    conn = psycopg2.connect(dbname='crm_hub', user='termux', host='192.168.10.107', port=5432)
    cur = conn.cursor()

    print("🟢 Conectado ao PostgreSQL crm_hub no Termux!")
    print("\n=== 1. ATUALIZANDO/INSERINDO PRODUTOS GAMER PARA CANAL 2 (GAMES CLUTCH) ===")

    products = [
        {
            "name": "Headset Gamer Redragon Scylla P2 Stereo RGB",
            "price": 149.90,
            "url": "https://produto.mercadolivre.com.br/MLB-356192801-headset-gamer-redragon-scylla-h901-p2-rgb-_JM?matt_tool=71730462&matt_word=aliferramos&forceInApp=true"
        },
        {
            "name": "Teclado Mecanico Gamer RGB Redragon Kumara Anti-Ghosting",
            "price": 199.90,
            "url": "https://produto.mercadolivre.com.br/MLB-281920391-teclado-mecanico-gamer-redragon-kumara-rgb-_JM?matt_tool=71730462&matt_word=aliferramos&forceInApp=true"
        },
        {
            "name": "Mouse Gamer Logitech G203 Lightsync 8000 DPI RGB",
            "price": 129.90,
            "url": "https://produto.mercadolivre.com.br/MLB-178291039-mouse-gamer-logitech-g203-lightsync-8000dpi-_JM?matt_tool=71730462&matt_word=aliferramos&forceInApp=true"
        },
        {
            "name": "Mousepad Gamer Extra Grande Speed 900x400mm RGB",
            "price": 49.90,
            "url": "https://produto.mercadolivre.com.br/MLB-192830192-mousepad-gamer-extra-grande-speed-900x400mm-_JM?matt_tool=71730462&matt_word=aliferramos&forceInApp=true"
        },
        {
            "name": "Controle Sem Fio Bluetooth DualShock / Xbox PC Gamer",
            "price": 119.90,
            "url": "https://produto.mercadolivre.com.br/MLB-201928301-controle-sem-fio-bluetooth-pc-ps4-xbox-_JM?matt_tool=71730462&matt_word=aliferramos&forceInApp=true"
        }
    ]

    for p in products:
        cur.execute("SELECT id FROM shop_products WHERE channel_id=2 AND name=%s;", (p["name"],))
        existing = cur.fetchone()
        if existing:
            cur.execute("UPDATE shop_products SET price=%s, affiliate_url=%s, active=true WHERE id=%s;", (p["price"], p["url"], existing[0]))
            print(f"  • Atualizado produto #{existing[0]}: {p['name']}")
        else:
            cur.execute("INSERT INTO shop_products (name, price, affiliate_url, active, channel_id) VALUES (%s, %s, %s, true, 2);", (p["name"], p["price"], p["url"]))
            print(f"  + Inserido novo produto: {p['name']}")

    conn.commit()

    print("\n=== 2. CONFIGURANDO OVERLAY BINDING (TOPO) PARA CANAL 2 ===")
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_name='channel_affiliate_bindings';")
    if cur.fetchone():
        cur.execute("SELECT id FROM channel_affiliate_bindings WHERE channel_id=2;")
        b_exist = cur.fetchone()
        if b_exist:
            cur.execute("UPDATE channel_affiliate_bindings SET position='top', is_active=true WHERE channel_id=2;")
            print("  • Binding Canal 2 atualizado para Posição TOP!")
        else:
            cur.execute("INSERT INTO channel_affiliate_bindings (channel_id, position, is_active) VALUES (2, 'top', true);")
            print("  + Binding Canal 2 criado com Posição TOP!")
        conn.commit()

    print("\n=== 3. CONFIRMAÇÃO DOS PRODUTOS ATIVOS DO CANAL 2 ===")
    cur.execute("SELECT id, name, price, affiliate_url FROM shop_products WHERE channel_id=2 AND active=true ORDER BY id;")
    for r in cur.fetchall():
        print(f"🟢 Prod #{r[0]}: {r[1]} | R$ {r[2]}")
        print(f"   URL: {r[3]}")

    cur.close()
    conn.close()

if __name__ == '__main__':
    setup_games_clutch()
