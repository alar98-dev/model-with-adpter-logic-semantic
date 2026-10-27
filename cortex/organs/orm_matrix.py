import psycopg2
from datetime import datetime
import json

DB_CONFIG = {
    'dbname': 'swa_matrix',
    'host': '/var/run/postgresql',
    'user': 'alifer'
}

def init_db():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Tabela 1: Sessões de Interação (Onde tudo começa)
    cur.execute('''
        CREATE TABLE IF NOT EXISTS swa_interaction_sessions (
            session_id VARCHAR(128) PRIMARY KEY,
            client_origin VARCHAR(64),
            spatial_anchor VARCHAR(255),
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP
        )
    ''')
    
    # Tabela 2: Pulsos Topológicos (Middlewares IN/OUT)
    cur.execute('''
        CREATE TABLE IF NOT EXISTS swa_topological_pulses (
            pulse_id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
            session_id VARCHAR(128) REFERENCES swa_interaction_sessions(session_id),
            direction VARCHAR(16),
            topological_temperature FLOAT,
            is_anomaly BOOLEAN,
            raw_payload TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Tabela 3: KVs Dinâmicos (Espaço Físico)
    cur.execute('''
        CREATE TABLE IF NOT EXISTS swa_spatial_kvs (
            workspace_hash VARCHAR(128) PRIMARY KEY,
            session_id VARCHAR(128) REFERENCES swa_interaction_sessions(session_id),
            physical_pt_path VARCHAR(255),
            tensor_vector vector(2560),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    cur.close()
    conn.close()
    print('[ORM MATRIX] Tabelas dimensionais forjadas no postgres (swa_matrix).')

def register_session(session_id, client_origin, spatial_anchor):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO swa_interaction_sessions (session_id, client_origin, spatial_anchor)
        VALUES (%s, %s, %s) ON CONFLICT (session_id) DO NOTHING
    ''', (session_id, client_origin, spatial_anchor))
    conn.commit()
    cur.close()
    conn.close()

def log_pulse(session_id, direction, temperature, is_anomaly, payload):
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO swa_topological_pulses (session_id, direction, topological_temperature, is_anomaly, raw_payload)
        VALUES (%s, %s, %s, %s, %s)
    ''', (session_id, direction, temperature, is_anomaly, json.dumps(payload)))
    conn.commit()
    cur.close()
    conn.close()

if __name__ == '__main__':
    init_db()
