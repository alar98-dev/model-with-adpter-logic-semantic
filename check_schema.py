import psycopg2
conn = psycopg2.connect('dbname=swa_matrix user=alifer')
cur = conn.cursor()
cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'blake_semantic_anchors'")
for row in cur.fetchall():
    print(row)
