import psycopg2
import traceback
conn = psycopg2.connect('dbname=swa_matrix user=alifer')
cur = conn.cursor()
try:
    print('Testing DB...')
    vec = [0.1] * 2560
    query = 'SELECT embedding_2560 FROM blake_semantic_anchors LIMIT 1;'
    cur.execute(query)
    res = cur.fetchone()
    print('Result type:', type(res))
    if res:
        print('Col type:', type(res[0]))
        print('String length:', len(str(res[0])))
except Exception as e:
    print('Error:', e)
    traceback.print_exc()
