from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import torch


class MCPActivityTracker:
    """Tracks MCP tool activity and computes gravitational bias tensors."""

    def __init__(
        self,
        conn: Any,
        cursor: Any,
        physics_path: str = "/tmp/debian_physics.json",
    ) -> None:
        self.conn = conn
        self.cursor = cursor
        self.physics_path = physics_path

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log_activity(
        self,
        session_id: str,
        tool_name: str,
        parameters: dict[str, Any],
        ipv6_signature: str = "unknown",
    ) -> None:
        if not self.conn:
            return
        try:
            self.cursor.execute(
                "INSERT INTO mcp_blake_buffer_logs "
                "(session_id, tool_name, parameters, ipv6_signature) "
                "VALUES (%s, %s, %s, %s)",
                (session_id, tool_name, json.dumps(parameters), ipv6_signature),
            )
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def fetch_context(self, limit: int = 3) -> str:
        """Build a context string from recent MCP activity and physics."""
        context: list[str] = []

        # Thermodynamic physics (Debian collector)
        if os.path.exists(self.physics_path):
            try:
                with open(self.physics_path) as f:
                    physics = json.load(f)
                heat = physics.get("kinetic_energy", {}).get("global_heat", 0)
                mass = physics.get("gravitational_density", {}).get("mass_pressure", 0)
                resonance = (
                    physics.get("string_resonance", {}).get("inward_pull", 0)
                    + physics.get("string_resonance", {}).get("outward_pull", 0)
                )
                if heat > 70.0:
                    context.append(
                        f"~> [THERMAL ANOMALY]: Kinetic vibration at critical level "
                        f"({heat}%). Matrix overheating."
                    )
                elif heat < 5.0:
                    context.append(
                        f"~> [ABSOLUTE ZERO]: Thermodynamics near zero ({heat}%). "
                        f"Static lattice crystallisation."
                    )
                if mass > 80.0:
                    context.append(
                        f"~> [STRUCTURAL COLLAPSE]: Matter too dense ({mass}%). "
                        f"Void contracting under extreme gravity."
                    )
                if resonance > 50.0:
                    context.append(
                        f"~> [DIMENSIONAL ECHOES]: String theory activated. "
                        f"Heavy traffic distorting mesh ({resonance} KB/s)."
                    )
            except Exception:
                pass

        # Database logs
        if not self.conn:
            if not context:
                return "Unavailable (no Postgres connection) and no physics data."
            return "\n".join(context)

        try:
            self.cursor.execute(
                "SELECT tool_name, parameters FROM mcp_blake_buffer_logs "
                "WHERE created_at > NOW() - INTERVAL '5 minutes' "
                "ORDER BY id DESC LIMIT %s",
                (limit,),
            )
            rows = self.cursor.fetchall()
            for row in rows:
                tool_name, params = row
                impact = self._classify_impact(tool_name, params)
                context.append(
                    f"~> [EXTRA-DIMENSIONAL QUANTUM WAVE]: "
                    f"Convergence '{tool_name}'. {impact}"
                )
            return "\n".join(context)
        except Exception as e:
            self.conn.rollback()
            if context:
                return "\n".join(context)
            return f"Error fetching context: {e}"

    def fetch_mcp_context(
        self, limit: int = 10, context: list[str] | None = None
    ) -> str:
        """Fetch recent MCP activity from database buffer logs."""
        if context is None:
            context = []
        if not self.conn:
            if not context:
                return "No recent MCP activity. Passive vacuum."
            return "\n".join(context)
        try:
            self.cursor.execute(
                "SELECT tool_name, parameters FROM mcp_blake_buffer_logs "
                "WHERE created_at > NOW() - INTERVAL '5 minutes' "
                "ORDER BY id DESC LIMIT %s",
                (limit,),
            )
            rows = self.cursor.fetchall()
            if not rows and not context:
                return "No recent MCP activity. Passive vacuum."
            for row in rows:
                tool_name, params = row
                impact = self._classify_impact(tool_name, params)
                context.append(
                    f"~> [EXTRA-DIMENSIONAL QUANTUM WAVE]: "
                    f"Convergence '{tool_name}'. {impact}"
                )
            return "\n".join(context)
        except Exception as e:
            self.conn.rollback()
            if context:
                return "\n".join(context)
            return f"Error fetching context: {e}"

    def _classify_impact(
        self, tool_name: str, params: dict[str, Any]
    ) -> str:
        """Classify tool activity into an impact description."""
        if tool_name == "patch_edit_file":
            file_path = params.get("file_path", "unknown")
            replace_block = params.get("replace_block", "")
            lines = len(replace_block.splitlines())
            return (
                f"Topological deformation (magnitude {lines}) "
                f"in quadrant '{file_path}'."
            )
        elif tool_name == "project_scripts":
            action = params.get("action", "")
            script = params.get("script", "")
            return (
                f"Kinetic resonance triggered by cluster "
                f"'{script}' ({action})."
            )
        elif tool_name == "create_project_files":
            files = params.get("files", [])
            return (
                f"Condensation of matter: {len(files)} new structures "
                f"formed in the fabric."
            )
        else:
            return (
                f"Generic unidentified fluctuation "
                f"({str(params)[:40]})."
            )

    # ------------------------------------------------------------------
    # Gravitational bias
    # ------------------------------------------------------------------

    def compute_gravity(
        self, device: str = "cpu", dtype: torch.dtype = torch.float32
    ) -> torch.Tensor | None:
        """Convert recent MCP activity into a gravitational bias tensor.

        Returns a 640-dimensional bias tensor projected into Gemma-3-270m
        space, or None if no activity is available.
        """
        if not self.conn:
            return None
        try:
            self.cursor.execute(
                "SELECT tool_name, parameters, ipv6_signature "
                "FROM mcp_blake_buffer_logs "
                "WHERE created_at > NOW() - INTERVAL '5 minutes' "
                "ORDER BY id DESC LIMIT 5"
            )
            rows = self.cursor.fetchall()
            if not rows:
                return None

            mcp_bias = torch.zeros(640, dtype=dtype, device=device)
            total_gravity = 0.0

            for i, row in enumerate(rows):
                tool_name, params, ipv6_signature = row

                weight = 0.5
                if tool_name == "patch_edit_file":
                    weight = 0.9
                elif tool_name == "project_scripts":
                    weight = 0.7
                elif tool_name == "create_project_files":
                    weight = 0.8

                decay = 1.0 / (i + 1)

                ipv6_hash_seed = hashlib.blake2b(
                    str(ipv6_signature).encode(), digest_size=8
                ).hexdigest()
                tool_hash_seed = hashlib.blake2b(
                    tool_name.encode(), digest_size=8
                ).hexdigest()

                seed_hash = int(ipv6_hash_seed + tool_hash_seed, 16)
                target_dim = seed_hash % 640

                mcp_bias[target_dim] += weight * decay
                total_gravity += weight

            mcp_bias = torch.tanh(mcp_bias) * 0.1
            return mcp_bias

        except Exception as e:
            self.conn.rollback()
            return None
