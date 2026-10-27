-- -----------------------------------------------------------------------------
-- SWA KERNEL - SEMANTIC SCHEMA BLUEPRINT
-- -----------------------------------------------------------------------------

-- Ativa a extensão de vetores se não estiver ativa
CREATE EXTENSION IF NOT EXISTS vector;

-- Versioned semantic projection registry (640D -> 1152D -> 2560D).
CREATE TABLE IF NOT EXISTS swa_projection_registry (
    name VARCHAR(96) PRIMARY KEY,
    version INTEGER NOT NULL DEFAULT 1,
    source_dim INTEGER NOT NULL,
    target_dim INTEGER NOT NULL,
    artifact_path TEXT,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

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

CREATE TABLE IF NOT EXISTS swa_narrative_events (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    wave INTEGER NOT NULL DEFAULT 0,
    event_type VARCHAR(32) NOT NULL,
    summary TEXT NOT NULL,
    entities JSONB NOT NULL DEFAULT '[]'::jsonb,
    objectives JSONB NOT NULL DEFAULT '[]'::jsonb,
    time_context TEXT NOT NULL DEFAULT '',
    location TEXT NOT NULL DEFAULT ''
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

-- 9. Realidade persistente do Aeon
-- Modelo relacional estável e diretamente mapeável para um ORM.
CREATE TABLE IF NOT EXISTS aeon_reality_worlds (
    id BIGSERIAL PRIMARY KEY,
    world_key VARCHAR(128) NOT NULL UNIQUE,
    schema_version INTEGER NOT NULL DEFAULT 1 CHECK (schema_version > 0),
    storage_uri TEXT NOT NULL,
    root_hash VARCHAR(128) NOT NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS aeon_reality_entities (
    id BIGSERIAL PRIMARY KEY,
    world_id BIGINT NOT NULL REFERENCES aeon_reality_worlds(id) ON DELETE CASCADE,
    entity_key VARCHAR(160) NOT NULL,
    entity_type VARCHAR(64) NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (world_id, entity_key)
);

CREATE TABLE IF NOT EXISTS aeon_reality_tensor_states (
    id BIGSERIAL PRIMARY KEY,
    world_id BIGINT NOT NULL REFERENCES aeon_reality_worlds(id) ON DELETE CASCADE,
    entity_id BIGINT REFERENCES aeon_reality_entities(id) ON DELETE SET NULL,
    wave INTEGER NOT NULL DEFAULT 0 CHECK (wave >= 0),
    dimension INTEGER NOT NULL CHECK (dimension > 0),
    dtype VARCHAR(32) NOT NULL,
    storage_uri TEXT NOT NULL,
    content_hash VARCHAR(128) NOT NULL,
    norm REAL NOT NULL DEFAULT 0 CHECK (norm >= 0),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (world_id, content_hash)
);

CREATE TABLE IF NOT EXISTS aeon_reality_events (
    id BIGSERIAL PRIMARY KEY,
    world_id BIGINT NOT NULL REFERENCES aeon_reality_worlds(id) ON DELETE CASCADE,
    wave INTEGER NOT NULL DEFAULT 0 CHECK (wave >= 0),
    event_type VARCHAR(64) NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    previous_hash VARCHAR(128),
    event_hash VARCHAR(128) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (world_id, event_hash)
);

CREATE TABLE IF NOT EXISTS aeon_reality_waves (
    id BIGSERIAL PRIMARY KEY,
    world_id BIGINT NOT NULL REFERENCES aeon_reality_worlds(id) ON DELETE CASCADE,
    conversation_key VARCHAR(160) NOT NULL,
    wave_index INTEGER NOT NULL CHECK (wave_index >= 0),
    parent_wave_id BIGINT REFERENCES aeon_reality_waves(id) ON DELETE SET NULL,
    phase VARCHAR(16) NOT NULL DEFAULT 'unknown'
        CHECK (phase IN ('inicio', 'meio', 'fim', 'deriva', 'unknown')),
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    state_tensor_id BIGINT REFERENCES aeon_reality_tensor_states(id) ON DELETE SET NULL,
    event_id BIGINT REFERENCES aeon_reality_events(id) ON DELETE SET NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (world_id, conversation_key, wave_index)
);

CREATE TABLE IF NOT EXISTS aeon_reality_experiences (
    id BIGSERIAL PRIMARY KEY,
    world_id BIGINT NOT NULL REFERENCES aeon_reality_worlds(id) ON DELETE CASCADE,
    conversation_key VARCHAR(160) NOT NULL,
    wave_index INTEGER NOT NULL CHECK (wave_index >= 0),
    wave_id BIGINT REFERENCES aeon_reality_waves(id) ON DELETE SET NULL,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    outcome VARCHAR(64) NOT NULL DEFAULT 'unknown',
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (world_id, conversation_key, wave_index)
);

CREATE TABLE IF NOT EXISTS aeon_reality_relations (
    id BIGSERIAL PRIMARY KEY,
    world_id BIGINT NOT NULL REFERENCES aeon_reality_worlds(id) ON DELETE CASCADE,
    source_entity_id BIGINT NOT NULL REFERENCES aeon_reality_entities(id) ON DELETE CASCADE,
    target_entity_id BIGINT NOT NULL REFERENCES aeon_reality_entities(id) ON DELETE CASCADE,
    relation_type VARCHAR(64) NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (source_entity_id <> target_entity_id),
    UNIQUE (world_id, source_entity_id, target_entity_id, relation_type)
);

CREATE TABLE IF NOT EXISTS aeon_reality_blobs (
    id BIGSERIAL PRIMARY KEY,
    world_id BIGINT NOT NULL REFERENCES aeon_reality_worlds(id) ON DELETE CASCADE,
    blob_key VARCHAR(160) NOT NULL,
    media_type VARCHAR(128) NOT NULL,
    storage_uri TEXT NOT NULL,
    content_hash VARCHAR(128) NOT NULL,
    byte_size BIGINT NOT NULL CHECK (byte_size >= 0),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (world_id, blob_key),
    UNIQUE (world_id, content_hash)
);

CREATE TABLE IF NOT EXISTS aeon_reality_tasks (
    id BIGSERIAL PRIMARY KEY,
    world_id BIGINT NOT NULL REFERENCES aeon_reality_worlds(id) ON DELETE CASCADE,
    parent_task_id BIGINT REFERENCES aeon_reality_tasks(id) ON DELETE SET NULL,
    task_key VARCHAR(160) NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    status VARCHAR(16) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'in_progress', 'blocked', 'done', 'cancelled')),
    priority INTEGER NOT NULL DEFAULT 0,
    state_tensor_id BIGINT REFERENCES aeon_reality_tensor_states(id) ON DELETE SET NULL,
    completion_event_id BIGINT REFERENCES aeon_reality_events(id) ON DELETE SET NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    UNIQUE (world_id, task_key)
);

CREATE TABLE IF NOT EXISTS aeon_contexts (
    id BIGSERIAL PRIMARY KEY,
    world_id BIGINT NOT NULL REFERENCES aeon_reality_worlds(id) ON DELETE CASCADE,
    path TEXT NOT NULL,
    content_hash VARCHAR(128) NOT NULL,
    content_type VARCHAR(128) NOT NULL DEFAULT 'text/plain',
    version INTEGER NOT NULL DEFAULT 1 CHECK (version > 0),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (world_id, content_hash)
);

CREATE TABLE IF NOT EXISTS aeon_context_embeddings (
    id BIGSERIAL PRIMARY KEY,
    world_id BIGINT NOT NULL REFERENCES aeon_reality_worlds(id) ON DELETE CASCADE,
    context_id BIGINT NOT NULL REFERENCES aeon_contexts(id) ON DELETE CASCADE,
    model_name VARCHAR(160) NOT NULL,
    dimensions INTEGER NOT NULL CHECK (dimensions > 0),
    embedding vector,
    norm REAL NOT NULL CHECK (norm > 0),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (context_id, model_name)
);

CREATE TABLE IF NOT EXISTS aeon_gravity_fields (
    id BIGSERIAL PRIMARY KEY,
    world_id BIGINT NOT NULL REFERENCES aeon_reality_worlds(id) ON DELETE CASCADE,
    conversation_key VARCHAR(160) NOT NULL,
    objective_vector vector NOT NULL,
    strength REAL NOT NULL DEFAULT 1.0 CHECK (strength >= 0 AND strength <= 2),
    context_hashes JSONB NOT NULL DEFAULT '[]'::jsonb,
    aligned_tokens JSONB NOT NULL DEFAULT '[]'::jsonb,
    opposed_tokens JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS aeon_context_observations (
    id BIGSERIAL PRIMARY KEY,
    world_id BIGINT NOT NULL REFERENCES aeon_reality_worlds(id) ON DELETE CASCADE,
    context_hash VARCHAR(128) NOT NULL,
    conversation_key VARCHAR(160) NOT NULL,
    wave_index INTEGER NOT NULL CHECK (wave_index >= 0),
    phase VARCHAR(16) NOT NULL,
    semantic_distance REAL NOT NULL CHECK (semantic_distance >= 0 AND semantic_distance <= 1),
    convergence REAL NOT NULL CHECK (convergence >= 0 AND convergence <= 1),
    decision VARCHAR(64) NOT NULL,
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS aeon_context_relations (
    id BIGSERIAL PRIMARY KEY,
    world_id BIGINT NOT NULL REFERENCES aeon_reality_worlds(id) ON DELETE CASCADE,
    source_hash VARCHAR(128) NOT NULL,
    target_hash VARCHAR(128) NOT NULL,
    relation_type VARCHAR(64) NOT NULL,
    similarity REAL NOT NULL CHECK (similarity >= 0 AND similarity <= 1),
    weight REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (source_hash <> target_hash),
    UNIQUE (world_id, source_hash, target_hash, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_reality_entities_world
    ON aeon_reality_entities(world_id);
CREATE INDEX IF NOT EXISTS idx_reality_tensor_states_world_wave
    ON aeon_reality_tensor_states(world_id, wave DESC);
CREATE INDEX IF NOT EXISTS idx_reality_events_world_created
    ON aeon_reality_events(world_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reality_events_meta_window
    ON aeon_reality_events(world_id, event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_reality_waves_conversation
    ON aeon_reality_waves(world_id, conversation_key, wave_index);
CREATE INDEX IF NOT EXISTS idx_reality_waves_parent
    ON aeon_reality_waves(parent_wave_id);
CREATE INDEX IF NOT EXISTS idx_reality_experiences_conversation
    ON aeon_reality_experiences(world_id, conversation_key, wave_index);
CREATE INDEX IF NOT EXISTS idx_reality_relations_source
    ON aeon_reality_relations(source_entity_id);
CREATE INDEX IF NOT EXISTS idx_reality_relations_target
    ON aeon_reality_relations(target_entity_id);
CREATE INDEX IF NOT EXISTS idx_reality_tasks_ready
    ON aeon_reality_tasks(world_id, status, priority DESC);
CREATE INDEX IF NOT EXISTS idx_aeon_contexts_hash
    ON aeon_contexts(world_id, content_hash);
CREATE INDEX IF NOT EXISTS idx_aeon_context_embeddings_context
    ON aeon_context_embeddings(context_id);
CREATE INDEX IF NOT EXISTS idx_aeon_gravity_fields_conversation
    ON aeon_gravity_fields(world_id, conversation_key, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_aeon_context_observations_wave
    ON aeon_context_observations(world_id, conversation_key, wave_index);
CREATE INDEX IF NOT EXISTS idx_aeon_context_relations_source
    ON aeon_context_relations(world_id, source_hash);
