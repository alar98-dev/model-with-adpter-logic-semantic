"""Camada ORM estruturada (SQLAlchemy 2.0) para as tabelas relacionais da Aeon.

Mapeia as entidades de identidade e o grafo relacional de forma declarativa,
com relacionamentos reais (foreign keys) expostos como atributos ORM — em vez
de SQL cru espalhado pelos controllers.

Tabelas mapeadas:
  - swa_identities            (1)
  - swa_faces                 (N por identity, ON DELETE CASCADE)
  - swa_recognition_sessions  (N por identity/face, log auditável)
  - agentes_registry          (agentes ativos do ecossistema)
  - aeon_reality_relations    (grafo de entidades do mundo)
  - aeon_reality_entities
  - vocal_utterances          (1 fala por Identity, com snapshot tensorial)
  - vocal_words               (N palavras por fala, com janela temporal)

Uso:
    from swa.db.models import session_scope, Identity
    with session_scope() as db:
        master = db.query(Identity).filter_by(papel='admin').first()
"""

from __future__ import annotations

import os
import uuid
import warnings
from datetime import datetime
from typing import Iterator, Optional
from contextlib import contextmanager

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    Table,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    scoped_session,
    sessionmaker,
)

DEFAULT_DSN = os.environ.get(
    "SWA_DB_DSN",
    "postgresql+psycopg2://alifer@/swa_matrix?host=/var/run/postgresql",
)

class _ORMBase:
    """Permite mapear com ``Column()`` + anotações ``list[...]`` (legado aceito)."""
    __allow_unmapped__ = True


Base = declarative_base(cls=_ORMBase)

engine = create_engine(
    DEFAULT_DSN,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, future=True, expire_on_commit=False)
ScopedSession = scoped_session(SessionLocal)


@contextmanager
def session_scope() -> Iterator:
    """Context manager que abre/fecha uma sessão de forma segura."""
    session = ScopedSession()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ─── Identidade e faces (reconhecimento facial Master/ALifer) ──────────────

class Identity(Base):
    __tablename__ = "swa_identities"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    nome = Column(String(128), nullable=False)
    apelido = Column(String(64))
    email = Column(String(128))
    papel = Column(String(32), default="usuario")
    ativo = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    meta = Column("metadata", JSONB, default=dict)

    faces: list["Face"] = relationship(
        "Face", back_populates="identity", cascade="all, delete-orphan"
    )
    recognition_sessions: list["RecognitionSession"] = relationship(
        "RecognitionSession", back_populates="identity"
    )
    utterances: list["VocalUtterance"] = relationship(
        "VocalUtterance", back_populates="identity", cascade="all, delete-orphan"
    )

    @property
    def is_master(self) -> bool:
        return self.papel == "admin" and self.ativo


class Face(Base):
    __tablename__ = "swa_faces"

    id = Column(Integer, primary_key=True)
    identity_id = Column(UUID(as_uuid=True), ForeignKey("swa_identities.id", ondelete="CASCADE"))
    embedding_128 = Column("embedding_128", Text, nullable=False)  # vector(128) via texto JSON
    angle = Column(String(16), default="front")
    photo_path = Column(Text)
    confidence_threshold = Column(Float, default=0.50)
    subtleties = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    identity: Identity = relationship("Identity", back_populates="faces")
    recognition_sessions: list["RecognitionSession"] = relationship(
        "RecognitionSession", back_populates="face"
    )


class RecognitionSession(Base):
    __tablename__ = "swa_recognition_sessions"

    id = Column(Integer, primary_key=True)
    identity_id = Column(UUID(as_uuid=True), ForeignKey("swa_identities.id"))
    face_id = Column(Integer, ForeignKey("swa_faces.id"))
    confidence = Column(Float, nullable=False)
    angle = Column(String(16))
    detected_at = Column(DateTime, default=datetime.utcnow)
    photo_snapshot_path = Column(Text)

    identity: Identity = relationship("Identity", back_populates="recognition_sessions")
    face: Face = relationship("Face", back_populates="recognition_sessions")


# ─── Agentes do ecossistema ────────────────────────────────────────────────

class AgentRegistry(Base):
    __tablename__ = "agentes_registry"

    name = Column(String(255), primary_key=True)
    agent_type = Column(String(50), nullable=False)
    pid = Column(Integer)
    endpoint = Column(Text)
    config = Column(JSONB, default=dict)
    session_id = Column(String(64))
    active = Column(Boolean, default=True)
    connected_at = Column(DateTime, default=datetime.utcnow)
    last_heartbeat = Column(DateTime, default=datetime.utcnow)


# ─── Grafo relacional de realidade (entidades + relações) ──────────────────

class RealityEntity(Base):
    __tablename__ = "aeon_reality_entities"

    id = Column(Integer, primary_key=True)
    world_id = Column(Integer, ForeignKey("aeon_reality_worlds.id"))
    entity_type = Column(String(64))
    name = Column(String(255))
    properties = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    outgoing: list["RealityRelation"] = relationship(
        "RealityRelation", foreign_keys="RealityRelation.source_entity_id",
        back_populates="source",
    )
    incoming: list["RealityRelation"] = relationship(
        "RealityRelation", foreign_keys="RealityRelation.target_entity_id",
        back_populates="target",
    )


class RealityWorld(Base):
    __tablename__ = "aeon_reality_worlds"

    id = Column(Integer, primary_key=True)
    name = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

    entities: list[RealityEntity] = relationship("RealityEntity", backref="world")


class RealityRelation(Base):
    __tablename__ = "aeon_reality_relations"

    id = Column(Integer, primary_key=True)
    world_id = Column(Integer, ForeignKey("aeon_reality_worlds.id"))
    source_entity_id = Column(Integer, ForeignKey("aeon_reality_entities.id"))
    target_entity_id = Column(Integer, ForeignKey("aeon_reality_entities.id"))
    relation_type = Column(String(64), nullable=False)
    weight = Column(Float, default=1.0)
    properties = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    source: RealityEntity = relationship(
        "RealityEntity", foreign_keys=[source_entity_id], back_populates="outgoing"
    )
    target: RealityEntity = relationship(
        "RealityEntity", foreign_keys=[target_entity_id], back_populates="incoming"
    )


# ─── Âncoras semânticas (gravadas por SQL cru no pipeline sensorial) ─────────
# Reflexão só para o FK de vocal_utterances.anchor_id resolver — a tabela tem
# embedding pgvector que o ORM não gerencia. A falha é engolida para que o
# import de models.py nunca dependa do banco estar no ar.

try:
    with warnings.catch_warnings():
        # A reflexão encontra colunas pgvector, que o SQLAlchemy 2.0 ainda não
        # tipa. Não usamos esses tipos — só precisamos do id para o FK — então o
        # aviso é ruído que poluiria o journal a cada fala registrada.
        warnings.filterwarnings("ignore", message="Did not recognize type 'vector'")
        Table(
            "blake_semantic_anchors",
            Base.metadata,
            autoload_with=engine,
            extend_existing=True,
        )
except Exception:  # pragma: no cover — banco indisponível
    pass


# ─── Ledger vocal: cada fala de Alifer, com snapshot tensorial e tempo por palavra ──

class VocalUtterance(Base):
    """Uma fala identificada pela assinatura vocal, persistida de forma rastreável.

    `anchor_id` liga a fala à âncora semântica em `blake_semantic_anchors` (que
    carrega o embedding 2560D indexável). `tensor_snapshot` guarda o retrato
    compacto do vetor — o perfil de energia por banda — para consulta JSONB sem
    duplicar 2560 floats por linha.
    """

    __tablename__ = "vocal_utterances"

    id = Column(Integer, primary_key=True)
    identity_id = Column(UUID(as_uuid=True), ForeignKey("swa_identities.id", ondelete="CASCADE"))
    anchor_id = Column(Integer, ForeignKey("blake_semantic_anchors.id", ondelete="SET NULL"))

    text = Column(Text, nullable=False)
    source = Column(String(32), default="MICROPHONE")
    captured_at = Column(DateTime, default=datetime.utcnow)

    # Identidade vocal (assinatura MFCC 128D avaliada na origem)
    similarity = Column(Float, default=0.0)
    is_owner = Column(Boolean, default=False)
    energy = Column(Float)

    # Proveniência temporal — nunca mistura fala real com fala reconstruída
    duration_ms = Column(Float)
    timing_source = Column(String(16), default="interpolated")

    tensor_snapshot = Column(JSONB, default=dict)
    meta = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

    identity: Identity = relationship("Identity", back_populates="utterances")
    words: list["VocalWord"] = relationship(
        "VocalWord", back_populates="utterance", cascade="all, delete-orphan",
        order_by="VocalWord.ordinal",
    )

    @property
    def is_confirmed(self) -> bool:
        """Fala do dono com similaridade acima do limiar de confiança."""
        return bool(self.is_owner) and float(self.similarity or 0.0) >= 0.85


class VocalWord(Base):
    """Uma palavra da fala, comjanela temporal e posição no chunk original."""

    __tablename__ = "vocal_words"

    id = Column(Integer, primary_key=True)
    utterance_id = Column(
        Integer, ForeignKey("vocal_utterances.id", ondelete="CASCADE"), nullable=False
    )
    ordinal = Column(Integer, nullable=False)
    surface = Column(Text, nullable=False)
    char_start = Column(Integer)
    char_end = Column(Integer)
    chunk_index = Column(Integer, default=0)
    t_start_ms = Column(Float)
    t_end_ms = Column(Float)
    confidence = Column(Float)

    utterance: VocalUtterance = relationship("VocalUtterance", back_populates="words")

    __table_args__ = (
        UniqueConstraint("utterance_id", "ordinal", name="uq_vocal_word_ordinal"),
    )


# ─── Roda Sexagesimal (Base 60) de Blake-Aeon ─────────────────────────────
from swa.db.sexagesimal_models import (
    SexagesimalRealm,
    SexagesimalHouse,
    RealmRelation,
    HistorianChronicle,
)


def init_models() -> None:
    """Cria as tabelas caso ainda não existam (idempotente)."""
    Base.metadata.create_all(engine)


if __name__ == "__main__":
    init_models()
    with session_scope() as db:
        for ident in db.query(Identity).all():
            role = "MASTER" if ident.is_master else ident.papel
            print(f"• {ident.nome} ({ident.apelido or '-'}) -> {role} | "
                  f"{len(ident.faces)} faces | {len(ident.recognition_sessions)} sessões")
