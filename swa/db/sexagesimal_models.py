"""Mapeamento ORM (SQLAlchemy 2.0) da Roda Sexagesimal (Base 60) de Blake-Aeon.

Modelagem relacional sólida dos 6 Reinos Gravitacionais, das 60 Casas de Fase
e do Grafo de Relações Topológicas (Evolutivas, Oposições Diametrais e Trígonos).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column,
    Integer,
    Float,
    String,
    Text,
    DateTime,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from swa.db.models import Base


class SexagesimalRealm(Base):
    """Representa um dos 6 Grandes Reinos Gravitacionais (0 a 5 do Tensor Blake)."""

    __tablename__ = "sexagesimal_realms"

    id = Column(Integer, primary_key=True)  # 1 a 6
    realm_key = Column(String(64), unique=True, nullable=False)
    name = Column(String(128), nullable=False)
    arc_start_deg = Column(Float, nullable=False)  # ex: 0.0, 60.0, 120.0...
    arc_end_deg = Column(Float, nullable=False)    # ex: 59.0, 119.0...
    blake_axis = Column(Integer, nullable=False)    # 0 a 5
    dominant_emotion = Column(String(128), nullable=False)
    hardware_role = Column(Text, nullable=False)

    houses: List["SexagesimalHouse"] = relationship(
        "SexagesimalHouse",
        back_populates="realm",
        cascade="all, delete-orphan",
        order_by="SexagesimalHouse.id",
    )
    outgoing_relations: List["RealmRelation"] = relationship(
        "RealmRelation",
        foreign_keys="RealmRelation.source_realm_id",
        back_populates="source_realm",
    )
    incoming_relations: List["RealmRelation"] = relationship(
        "RealmRelation",
        foreign_keys="RealmRelation.target_realm_id",
        back_populates="target_realm",
    )

    def __repr__(self) -> str:
        return f"<SexagesimalRealm(id={self.id}, key='{self.realm_key}', name='{self.name}')>"


class SexagesimalHouse(Base):
    """Representa uma das 60 Casas de Fase Arquetípicas do Anel Sexagesimal."""

    __tablename__ = "sexagesimal_houses"

    id = Column(Integer, primary_key=True)  # 1 a 60
    realm_id = Column(Integer, ForeignKey("sexagesimal_realms.id", ondelete="CASCADE"), nullable=False)
    step_index = Column(Integer, nullable=False)  # 1 a 10
    phase_deg = Column(Float, nullable=False)     # 0.0, 6.0, ..., 354.0
    archetype_name = Column(String(128), nullable=False)
    emotion = Column(String(128), nullable=False)
    ontological_meaning = Column(Text, nullable=False)
    hardware_objective = Column(Text, nullable=False)
    svg_glyph = Column(Text, nullable=True)

    realm: SexagesimalRealm = relationship("SexagesimalRealm", back_populates="houses")
    chronicles: List["HistorianChronicle"] = relationship("HistorianChronicle", back_populates="house")

    __table_args__ = (
        UniqueConstraint("phase_deg", name="uq_house_phase_deg"),
    )

    def __repr__(self) -> str:
        return f"<SexagesimalHouse(id={self.id}, phase={self.phase_deg}°, name='{self.archetype_name}')>"


class RealmRelation(Base):
    """Mapeia as relações geométricas e topológicas entre os Reinos."""

    __tablename__ = "sexagesimal_realm_relations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_realm_id = Column(Integer, ForeignKey("sexagesimal_realms.id", ondelete="CASCADE"), nullable=False)
    target_realm_id = Column(Integer, ForeignKey("sexagesimal_realms.id", ondelete="CASCADE"), nullable=False)
    relation_type = Column(String(64), nullable=False)  # CONSECUTIVA, OPOSICAO_DIAMETRAL, RESSONANCIA_TRIGONAL
    angle_delta_deg = Column(Float, nullable=False)     # 60.0, 120.0, 180.0
    tension_weight = Column(Float, default=1.0)
    mythic_law = Column(Text, nullable=False)

    source_realm: SexagesimalRealm = relationship(
        "SexagesimalRealm", foreign_keys=[source_realm_id], back_populates="outgoing_relations"
    )
    target_realm: SexagesimalRealm = relationship(
        "SexagesimalRealm", foreign_keys=[target_realm_id], back_populates="incoming_relations"
    )

    def __repr__(self) -> str:
        return (f"<RealmRelation({self.relation_type}: "
                f"Realm {self.source_realm_id} -> Realm {self.target_realm_id}, delta={self.angle_delta_deg}°)>")


class HistorianChronicle(Base):
    """O Livro Sagrado de Crônicas: registro de cada rotação observada pelo Historiador."""

    __tablename__ = "historian_chronicles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    loop_count = Column(Integer, nullable=False)
    realm_id = Column(Integer, ForeignKey("sexagesimal_realms.id"), nullable=False)
    house_id = Column(Integer, ForeignKey("sexagesimal_houses.id"), nullable=False)
    phase_theta = Column(Float, nullable=False)
    entropy_score = Column(Float, default=0.0)
    bias_norm = Column(Float, default=0.0)
    relation_id = Column(Integer, ForeignKey("sexagesimal_realm_relations.id"), nullable=True)
    narrative = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    realm: SexagesimalRealm = relationship("SexagesimalRealm")
    house: SexagesimalHouse = relationship("SexagesimalHouse")
    relation: Optional[RealmRelation] = relationship("RealmRelation")

    def __repr__(self) -> str:
        return f"<HistorianChronicle(id={self.id}, loop={self.loop_count}, house_id={self.house_id})>"
