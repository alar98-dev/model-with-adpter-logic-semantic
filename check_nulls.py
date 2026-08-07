import psycopg2
conn = psycopg2.connect('dbname=swa_matrix user=alifer')
cur = conn.cursor()
cur.execute('SELECT COUNT(*), COUNT(embedding_2560) FROM blake_semantic_anchors;')
print(cur.fetchone())
