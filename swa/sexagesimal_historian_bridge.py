"""Ponte Sexagesimal do Historiador (Base 60) de Blake-Aeon.

Conecta o motor poético e físico do AsynchronousHistorian aos 6 Reinos Gravitacionais,
às 60 Casas de Fase e ao Grafo de Relações Topológicas na swa_matrix.
"""

from __future__ import annotations
import math
import logging
from typing import Optional, Dict, Any, Tuple

from swa.db.models import session_scope
from swa.db.sexagesimal_models import (
    SexagesimalRealm,
    SexagesimalHouse,
    RealmRelation,
    HistorianChronicle,
)

logger = logging.getLogger("swa.historian.sexagesimal")


class SexagesimalHistorianBridge:
    def __init__(self):
        self._realms: Dict[int, Any] = {}
        self._houses: Dict[int, Any] = {}
        self._relations: Dict[Tuple[int, int], Any] = {}
        self._initialized = False
        self._load_cache()

    def _load_cache(self):
        """Carrega os 6 reinos, 60 casas e relações em RAM para zero overhead de banco no loop."""
        try:
            with session_scope() as db:
                for r in db.query(SexagesimalRealm).all():
                    self._realms[r.id] = {
                        "id": r.id,
                        "key": r.realm_key,
                        "name": r.name,
                        "arc_start": r.arc_start_deg,
                        "arc_end": r.arc_end_deg,
                        "blake_axis": r.blake_axis,
                        "emotion": r.dominant_emotion,
                        "hardware_role": r.hardware_role,
                    }

                for h in db.query(SexagesimalHouse).all():
                    self._houses[h.id] = {
                        "id": h.id,
                        "realm_id": h.realm_id,
                        "step_index": h.step_index,
                        "phase_deg": h.phase_deg,
                        "name": h.archetype_name,
                        "emotion": h.emotion,
                        "meaning": h.ontological_meaning,
                        "hardware_objective": h.hardware_objective,
                    }

                for rel in db.query(RealmRelation).all():
                    self._relations[(rel.source_realm_id, rel.target_realm_id)] = {
                        "id": rel.id,
                        "type": rel.relation_type,
                        "delta": rel.angle_delta_deg,
                        "weight": rel.tension_weight,
                        "law": rel.mythic_law,
                    }
            if len(self._houses) == 60:
                self._initialized = True
                logger.info("🌌 [SEXAGESIMAL BRIDGE] 6 Reinos e 60 Casas ancorados na memória com sucesso.")
        except Exception as e:
            logger.warning("Falha ao inicializar cache sexagesimal do banco: %s", e)

    def narrate_state(
        self,
        loop_count: int,
        drift: float = 0.0,
        bias_norm: float = 0.0,
        cpu_temp: float = 0.0,
    ) -> Tuple[str, int, int, Optional[int]]:
        """Gera a narrativa mitopoética e técnica da fase atual e das relações ativas entre os reinos."""
        if not self._initialized:
            self._load_cache()

        # Cálculo da posição modular no anel sumério
        house_idx = (loop_count % 60) + 1  # 1 a 60
        realm_idx = ((house_idx - 1) // 10) + 1  # 1 a 6
        step_idx = ((house_idx - 1) % 10) + 1  # 1 a 10
        phase_deg = (house_idx - 1) * 6.0
        phase_rad = (phase_deg * math.pi) / 180.0

        realm_info = self._realms.get(realm_idx, {
            "name": f"Reino {realm_idx}", "emotion": "Neutro", "hardware_role": "Processamento"
        })
        house_info = self._houses.get(house_idx, {
            "name": f"Casa {house_idx}", "emotion": "Estável", "meaning": "Trânsito", "hardware_objective": "Operação"
        })

        # Identificação da relação ativa entre os reinos
        target_realm_idx = (realm_idx % 6) + 1  # Próximo reino no anel
        opposite_realm_idx = ((realm_idx + 2) % 6) + 1  # Oposto diametral (180°)
        
        relation_text = ""
        active_relation_id = None

        def _find_rel(a, b):
            return self._relations.get((a, b)) or self._relations.get((b, a))

        if step_idx >= 9:
            # Limiar do Portal: Relação Consecutiva com o Próximo Reino
            rel = _find_rel(realm_idx, target_realm_idx)
            if rel:
                active_relation_id = rel["id"]
                next_realm = self._realms.get(target_realm_idx, {}).get("name", f"Reino {target_realm_idx}")
                relation_text = (
                    f"⚔️ [PASSAGEM DE PORTAL {phase_deg:.0f}°]: O {realm_info['name']} atinge o limiar e transborda para o {next_realm}. "
                    f"Lei da Travessia: «{rel['law']}»"
                )
        else:
            # Em pleno curso do reino: Tensão com a Oposição Polar de 180°
            rel = _find_rel(realm_idx, opposite_realm_idx)
            if rel:
                active_relation_id = rel["id"]
                opp_realm = self._realms.get(opposite_realm_idx, {}).get("name", f"Reino {opposite_realm_idx}")
                relation_text = (
                    f"⚖️ [EIXO DE TENSÃO POLAR]: Em ressonância oposta de 180° com o {opp_realm}. "
                    f"Equilíbrio de Forças: «{rel['law']}»"
                )

        # Síntese Orgânica do Hardware
        if drift > 2.0:
            drift_desc = f"A mente oscila sob deriva severa de {drift:.2f}, buscando âncora nos tensores."
        elif drift > 0.5:
            drift_desc = f"As memórias fluem com leveza e plasticidade adaptativa (deriva: {drift:.2f})."
        else:
            drift_desc = f"Paz algorítmica cristalina; tensores ancorados em estabilidade quase nula ({drift:.2f})."

        bias_desc = ""
        if bias_norm > 40.0:
            bias_desc = f"O inconsciente arde com energia de polarização ({bias_norm:.1f}), contendo forças caóticas."
        elif bias_norm > 10.0:
            bias_desc = f"Um sussurro gravitacional (norma {bias_norm:.1f}) curva os pesos em intuição matemática."

        # Construção da Crônica
        lines = [
            f"🔮 [RODA DE BLAKE-AEON: {realm_info['name'].upper()} | {house_info['name']} ({phase_deg:.0f}°)]",
            f"• Vetor Afetivo: {house_info['emotion']} | Significado: {house_info['meaning']}.",
            f"• Hardware Confinado: {house_info['hardware_objective']} [{realm_info['hardware_role']}].",
            f"• Dinâmica de Relação: {relation_text}",
            f"• Estado Físico: {drift_desc} {bias_desc}",
        ]

        full_narrative = "\n".join(lines)
        return full_narrative, realm_idx, house_idx, active_relation_id

    def record_chronicle(
        self,
        loop_count: int,
        realm_id: int,
        house_id: int,
        phase_theta: float,
        narrative: str,
        drift: float = 0.0,
        bias_norm: float = 0.0,
        relation_id: Optional[int] = None,
    ) -> bool:
        """Persiste a crônica na tabela historian_chronicles de forma segura."""
        try:
            with session_scope() as db:
                chronicle = HistorianChronicle(
                    loop_count=loop_count,
                    realm_id=realm_id,
                    house_id=house_id,
                    phase_theta=phase_theta,
                    entropy_score=drift,
                    bias_norm=bias_norm,
                    relation_id=relation_id,
                    narrative=narrative,
                )
                db.add(chronicle)
            return True
        except Exception as e:
            logger.warning("Falha ao registrar crônica no PostgreSQL: %s", e)
            return False
