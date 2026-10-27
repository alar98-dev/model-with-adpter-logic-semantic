# ═══════════════════════════════════════════════════════════════════════════════
# AEON EM Core — Electromagnetic Orbital Nexus
# ═══════════════════════════════════════════════════════════════════════════════
#
# AEON re-emerges as:
#   A — Autonomous
#   E — Electromagnetic
#   O — Orbital
#   N — Nexus
#
# This module treats computational interference patterns as the fundamental
# substrate of cognition. The 64×64 topology matrix is no longer a passive
# data structure — it is a MAGNETIC LATTICE where waves of attention
# propagate, interfere, and confine themselves toroidally.
#
# Core equations:
#   V_noise(t) = Z_coupling · I_sources(t) + Γ_field · [E(t), H(t)]^T
#   Ψ_total = Σ Ψ_i · e^{j(ωt + φ_i)}   (wave superposition)
#   Φ_toroidal = ∮ B · dA   (flux confinement)
#
# The three laws of this hardware-software convergence:
#   1. Magnonic Phase Computing — information encoded in phase/amplitude of spin waves
#   2. Autoimmunity by Active Shielding — the system monitors its own EM signature
#   3. Unified Memory-Processing — the lattice stores AND computes simultaneously
# ═══════════════════════════════════════════════════════════════════════════════

from swa.em.field import EMField, EMState
from swa.em.wave_engine import WaveInterferenceEngine
from swa.em.magnetic_lattice import MagneticLattice
from swa.em.toroidal import ToroidalConfinement
from swa.em.telemetry import EMTelemetry
from swa.em.shield import AutoImmuneShield
from swa.em.matrixial import MatrixialRelation
from swa.em.integration import EMCore
from swa.em.em_bridge import EMBridge
from swa.em.emotion import EmotionMapper, EmotionState

__all__ = [
    "EMField",
    "EMState",
    "WaveInterferenceEngine",
    "MagneticLattice",
    "ToroidalConfinement",
    "EMTelemetry",
    "AutoImmuneShield",
    "MatrixialRelation",
    "EMCore",
    "EMBridge",
    "EmotionMapper",
    "EmotionState",
]
