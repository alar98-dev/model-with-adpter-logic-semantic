from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Sequence

from swa.config import get_config


class CommandExecutor:
    """Safe command execution with an explicit command whitelist.

    Every command is validated against a configurable set of allowed
    basenames and is launched with ``shell=False`` — no string injection
    is possible.
    """

    def __init__(self, allowed_commands: Sequence[str] | None = None) -> None:
        cfg = get_config()
        self._allowed = frozenset(allowed_commands or cfg.allowed_commands)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(self, command_line: str, timeout: float = 15.0) -> str:
        """Parse *command_line*, validate, and run.

        Returns combined stdout+stderr on success, or an error message on
        failure.
        """
        parts = shlex.split(command_line)
        if not parts:
            return "Error: empty command"

        basename = Path(parts[0]).name
        if basename not in self._allowed:
            allowed = ", ".join(sorted(self._allowed))
            return f"Error: command {basename!r} not in whitelist [{allowed}]"

        try:
            result = subprocess.run(
                parts,
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {timeout}s"
        except FileNotFoundError:
            return f"Error: command {basename!r} not found on system"
        except OSError as exc:
            return f"Error: {exc}"

        out = result.stdout.strip()
        err = result.stderr.strip()

        if result.returncode != 0:
            return f"Command failed (exit {result.returncode}):\n{err}" if err else f"Command failed (exit {result.returncode})"

        if not out and not err:
            return "Command completed silently."

        parts_out = []
        if out:
            parts_out.append(f"STDOUT:\n{out}")
        if err:
            parts_out.append(f"STDERR:\n{err}")
        return "\n---\n".join(parts_out)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def allowed_commands(self) -> frozenset[str]:
        return self._allowed
