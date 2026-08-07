import psycopg2
import blake3
import json

def test_blake():
    token_text = 'rooms'
    current_vector_state = [0.1] * 1152
    
    hasher = blake3.blake3()
    hasher.update(token_text.encode('utf-8'))
    b3_hash = hasher.hexdigest()
    
    conn = psycopg2.connect('dbname=swa_matrix user=alifer')
    cur = conn.cursor()
    
    original_dim = len(current_vector_state)
    query_vector = current_vector_state.copy()
    if original_dim < 2560:
        query_vector.extend([0.0] * (2560 - original_dim))
    elif original_dim > 2560:
        query_vector = query_vector[:2560]
        
    query = '''
        SELECT embedding_2560 
        FROM blake_semantic_anchors 
        ORDER BY 
           (signature_hash = %s) DESC, 
           (embedding_2560 <=> %s::vector) ASC
        LIMIT 1;
    '''
    cur.execute(query, (b3_hash, query_vector))
    result = cur.fetchone()
    
    if result and result[0]:
        memory_vector = result[0]
        if isinstance(memory_vector, str):
            memory_vector = json.loads(memory_vector)
            
        if len(memory_vector) > original_dim:
            memory_vector = memory_vector[:original_dim]
        elif len(memory_vector) < original_dim:
            memory_vector.extend([0.0] * (original_dim - len(memory_vector)))
            
        delta_sinaptico = [m - c for m, c in zip(memory_vector, current_vector_state)]
        print('Success. Delta length:', len(delta_sinaptico))
        return delta_sinaptico
    print('Result empty')
    return None

import traceback
try:
    test_blake()
except Exception as e:
    print('Exception occurred:')
    traceback.print_exc()
