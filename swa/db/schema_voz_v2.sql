-- ==========================================================================
-- schema_voz_v2.sql — PADRÃO DE VOZ: IDENTIDADE ÚNICA + HIERARQUIA DE TENSORES
-- ==========================================================================
-- Substitui o modelo plano (acoustic_chunk_snapshots solto + vocal_utterances
-- sem grupo) por três níveis encadeados, cada um com vetorização acústica,
-- representação de energia e tensor binário:
--
--     vocal_segments   (conjunto de frases — sessão/bloco de contexto)
--           └─ vocal_utterances   (frase)
--                  └─ vocal_words  (palavra)
--
-- Princípios:
--  1. IDENTIDADE ÚNICA — tudo referencia swa_identities por FK. Nenhum
--     `speaker_identified` solto pode decidir sozinho quem falou.
--  2. TENSOR BINÁRIO — os vetores vivem em `bytea` float32 little-endian,
--     que é o que o PostgreSQL manipula sem parse. O JSONB ao lado é apenas
--     a sombra legível, nunca a fonte.
--  3. ROLAGEM — frase herda a média das palavras, conjunto herda a média das
--     frases. O agregado é sempre recalculável a partir da base.
--  4. IDEMPOTENTE — pode rodar quantas vezes quiser.
-- ==========================================================================

BEGIN;

-- ---------------------------------------------------------------------------
-- NÍVEL 3 — conjunto de frases
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS vocal_segments (
    id                  bigserial PRIMARY KEY,
    identity_id         uuid        NOT NULL REFERENCES swa_identities(id) ON DELETE CASCADE,
    session_key         text        NOT NULL,
    started_at          timestamptz NOT NULL,
    ended_at            timestamptz,
    word_count          integer     NOT NULL DEFAULT 0,
    utterance_count     integer     NOT NULL DEFAULT 0,
    duration_ms         double precision,
    vector_acoustic     bytea,          -- float32[]  — média das frases
    vector_energia      bytea,          -- float32[16] — energia agregada
    tensor_acoustic     bytea,          -- float32[]  — tensor bruto do agregado
    shadow              jsonb,          -- sombra legível (NÃO é a fonte)
    created_at          timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_vocal_segments_session
    ON vocal_segments (session_key);
CREATE INDEX IF NOT EXISTS ix_vocal_segments_identity
    ON vocal_segments (identity_id, started_at DESC);

-- ---------------------------------------------------------------------------
-- NÍVEL 2 — frase
-- ---------------------------------------------------------------------------
-- vocal_utterances já existe do trabalho anterior; aqui entra o que faltava:
-- a ligação com o conjunto, com o chunk acústico de origem, e os vetores.
ALTER TABLE vocal_utterances ADD COLUMN IF NOT EXISTS segment_id bigint
    REFERENCES vocal_segments(id) ON DELETE SET NULL;
ALTER TABLE vocal_utterances ADD COLUMN IF NOT EXISTS acoustic_chunk_id integer;
ALTER TABLE vocal_utterances ADD COLUMN IF NOT EXISTS started_at timestamptz;
ALTER TABLE vocal_utterances ADD COLUMN IF NOT EXISTS ended_at timestamptz;
ALTER TABLE vocal_utterances ADD COLUMN IF NOT EXISTS text_hash text;
ALTER TABLE vocal_utterances ADD COLUMN IF NOT EXISTS vector_acoustic bytea;
ALTER TABLE vocal_utterances ADD COLUMN IF NOT EXISTS vector_energia bytea;
ALTER TABLE vocal_utterances ADD COLUMN IF NOT EXISTS tensor_acoustic bytea;
ALTER TABLE vocal_utterances ADD COLUMN IF NOT EXISTS shadow jsonb;

CREATE INDEX IF NOT EXISTS ix_vocal_utterances_segment
    ON vocal_utterances (segment_id);
CREATE INDEX IF NOT EXISTS ix_vocal_utterances_chunk
    ON vocal_utterances (acoustic_chunk_id);
CREATE INDEX IF NOT EXISTS ix_vocal_utterances_started
    ON vocal_utterances (started_at DESC);
-- Uma frase não pode ser contada duas vezes a partir do mesmo chunk.
CREATE UNIQUE INDEX IF NOT EXISTS ux_vocal_utterances_chunk
    ON vocal_utterances (acoustic_chunk_id) WHERE acoustic_chunk_id IS NOT NULL;

-- ---------------------------------------------------------------------------
-- NÍVEL 1 — palavra
-- ---------------------------------------------------------------------------
ALTER TABLE vocal_words ADD COLUMN IF NOT EXISTS abs_timestamp double precision;
ALTER TABLE vocal_words ADD COLUMN IF NOT EXISTS t_source varchar(16);
ALTER TABLE vocal_words ADD COLUMN IF NOT EXISTS confidence double precision;
ALTER TABLE vocal_words ADD COLUMN IF NOT EXISTS energy_rms double precision;
ALTER TABLE vocal_words ADD COLUMN IF NOT EXISTS z_score double precision;
ALTER TABLE vocal_words ADD COLUMN IF NOT EXISTS vector_acoustic bytea;  -- float32[]
ALTER TABLE vocal_words ADD COLUMN IF NOT EXISTS vector_energia bytea;   -- float32[16]
ALTER TABLE vocal_words ADD COLUMN IF NOT EXISTS tensor_acoustic bytea;  -- float32[]

CREATE INDEX IF NOT EXISTS ix_vocal_words_utterance
    ON vocal_words (utterance_id, ordinal);
CREATE INDEX IF NOT EXISTS ix_vocal_words_abs_ts
    ON vocal_words (abs_timestamp) WHERE abs_timestamp IS NOT NULL;

-- ---------------------------------------------------------------------------
-- IDENTIDADE ÚNICA — o chunk acústico passa a ter FK de identidade
-- ---------------------------------------------------------------------------
-- 129 linhas hoje com `speaker_identified` em texto livre ('escritorio',
-- 'Desconhecido', 'Alifer Ramos'). Isso permits que a mesma pessoa seja
-- gravada sob dois nomes e que nenhum FK reclame. A coluna vira o espelho
-- textual do FK: continua legível, mas não decide sozinha.
ALTER TABLE acoustic_chunk_snapshots
    ADD COLUMN IF NOT EXISTS identity_id uuid;
ALTER TABLE acoustic_chunk_snapshots
    ADD COLUMN IF NOT EXISTS word_count integer NOT NULL DEFAULT 0;
ALTER TABLE acoustic_chunk_snapshots
    ADD COLUMN IF NOT EXISTS segment_id bigint;

UPDATE acoustic_chunk_snapshots AS c
   SET word_count = COALESCE(jsonb_array_length(c.word_snapshots), 0)
 WHERE c.word_count = 0;

COMMIT;

-- ---------------------------------------------------------------------------
-- Ligar o chunk à identidade. Nomes que não existem em swa_identities ficam
-- com NULL de propósito: a FK prefere negar a atribuição a inventar um dono.
-- Rola fora de transação para não abortar tudo por uma linha órfã.
-- ---------------------------------------------------------------------------
UPDATE acoustic_chunk_snapshots c
   SET identity_id = i.id
  FROM swa_identities i
 WHERE i.nome = c.speaker_identified
   AND c.identity_id IS NULL;

BEGIN;
DO $$
DECLARE n int;
BEGIN
    SELECT count(*) INTO n FROM acoustic_chunk_snapshots WHERE identity_id IS NULL;
    IF n > 0 THEN
        RAISE NOTICE 'acoustic_chunk_snapshots: % linhas sem identidade (locutor nao cadastrado) — mantidas com NULL', n;
    END IF;
END $$;
COMMIT;

CREATE INDEX IF NOT EXISTS ix_acoustic_identity
    ON acoustic_chunk_snapshots (identity_id) WHERE identity_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS ix_acoustic_speaker_identified
    ON acoustic_chunk_snapshots (speaker_identified);

-- ---------------------------------------------------------------------------
-- Visão única: a hierarquia inteira em uma consulta.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE VIEW vocal_hierarquia AS
SELECT s.id            AS segment_id,
       s.session_key,
       i.nome          AS identidade,
       s.started_at,
       u.id            AS utterance_id,
       u.text,
       u.started_at    AS fala_inicio,
       u.ended_at      AS fala_fim,
       u.similarity,
       u.acoustic_chunk_id,
       w.id            AS word_id,
       w.ordinal,
       w.surface,
       w.abs_timestamp,
       w.t_start_ms,
       w.t_end_ms,
       w.confidence,
       w.energy_rms,
       w.z_score,
       octet_length(w.tensor_acoustic) AS tensor_bytes
  FROM vocal_segments s
  JOIN swa_identities i        ON i.id = s.identity_id
  LEFT JOIN vocal_utterances u ON u.segment_id = s.id
  LEFT JOIN vocal_words     w  ON w.utterance_id = u.id
 ORDER BY s.started_at, u.started_at, w.ordinal;
