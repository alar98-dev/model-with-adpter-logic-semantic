import psycopg2
import traceback
conn = psycopg2.connect('dbname=swa_matrix user=alifer')
cur = conn.cursor()
vec = [0.1] * 2560
try:
    query = '''
        SELECT embedding_2560
        FROM blake_semantic_anchors 
        WHERE embedding_2560 IS NOT NULL
        ORDER BY 
           (signature_hash = 'test') DESC NULLS LAST, 
           (embedding_2560 <=> %s::vector) ASC
        LIMIT 1;
    '''
    cur.execute(query, (vec,))
    res = cur.fetchone()
    if res:
        print('res type:', type(res))
        print('res[0] type:', type(res[0]))
        print('res[0] truth value:', bool(res[0]))
    else:
        print('res is None!')
except Exception as e:
    print('SQL Error:', e)
    traceback.print_exc()
