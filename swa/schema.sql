-- -----------------------------------------------------------------------------
-- SWA KERNEL - SEMANTIC SCHEMA BLUEPRINT
-- -----------------------------------------------------------------------------

-- Ativa a extensão de vetores se não estiver ativa
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Tabela Base de Âncoras Semânticas (Arquivos e Conceitos do Crawler)
CREATE TABLE IF NOT EXISTS blake_semantic_anchors (
    id SERIAL PRIMARY KEY,
    file_path TEXT UNIQUE,
    concept_group VARCHAR(64),          -- 'MEMORY', 'CPU', 'SECURITY', 'GOOGLE_NATIVE_MANIFOLD'
    signature_hash VARCHAR(64),        -- SWA_HASH_MASTER
    raw_content TEXT,
    embedding_384 vector(384),         -- Embedding ultra-rápido do crawler
    embedding_2560 vector(2560),       -- Projeção nativa para o Gemma-3
    weight REAL DEFAULT 1.0            -- Peso de importância semântica
);

-- 2. Tabela de Episódios Cognitivos e Memórias Episódicas
CREATE TABLE IF NOT EXISTS swa_episodic_episodes (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    concept_group VARCHAR(64) NOT NULL,
    content TEXT NOT NULL,
    embedding_2560 vector(2560) NOT NULL,
    weight REAL DEFAULT 1.0
);

-- 3. Tabela de Logs de Agendamento do Controlador P-Core (Particionada por Range de Tempo)
CREATE TABLE IF NOT EXISTS swa_cpu_scheduler_logs (
    id SERIAL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    utilization_average REAL NOT NULL,
    temperature_average REAL NOT NULL,
    active_cores INTEGER NOT NULL,
    throttled_cores INTEGER NOT NULL,
    diagnostic_summary TEXT,
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- 4. Tabela de Registros de Afinidade e Controle Nominal de Core Físico
CREATE TABLE IF NOT EXISTS swa_cpu_affinity_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    core_id INTEGER NOT NULL,
    logical_processors TEXT NOT NULL,
    semantic_trace vector(2560)         -- Assinatura de estado de afinidade projetada em 2560d
);

-- 5. Tabela de Logs de Páginas e Alocação Semântica de Memória RAM (Particionada por Range de Tempo)
CREATE TABLE IF NOT EXISTS swa_memory_page_logs (
    id SERIAL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    total_ram_gb REAL NOT NULL,
    used_ram_gb REAL NOT NULL,
    percent_used REAL NOT NULL,
    semantic_state_signature VARCHAR(64),
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (timestamp);

-- 6. Tabela de Estados Dinâmicos do Kernel SWA
CREATE TABLE IF NOT EXISTS matrix_system_state (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    current_bias REAL NOT NULL,
    cpu_entropy REAL NOT NULL,
    mem_inertia REAL NOT NULL,
    stream_hash_master VARCHAR(64) NOT NULL
);

-- 7. Tabela de Relacionamentos de Embeddings Eventuais e Métricas Físicas
CREATE TABLE IF NOT EXISTS swa_eventual_embeddings (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    state_id INTEGER REFERENCES matrix_system_state(id) ON DELETE CASCADE,
    cpu_log_id INTEGER,
    mem_log_id INTEGER,
    eventual_bias_2560 vector(2560) NOT NULL,
    phase_shift_angle REAL NOT NULL,
    signature_hash VARCHAR(64) NOT NULL
);

-- 8. Tabela de Referência de Erros Normalizados (Consulta Humana)
CREATE TABLE IF NOT EXISTS swa_error_reference (
    error_code VARCHAR(16) PRIMARY KEY,
    error_class VARCHAR(64) NOT NULL,
    log_pattern TEXT NOT NULL,
    physical_explanation TEXT NOT NULL,
    auto_mitigation TEXT NOT NULL,
    action_needed TEXT NOT NULL
);
