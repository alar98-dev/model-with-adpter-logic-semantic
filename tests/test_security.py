"""Tests for swa.security — CommandExecutor and PathValidator."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from swa.security.command_executor import CommandExecutor
from swa.security.path_validator import PathValidator


# ── CommandExecutor ───────────────────────────────────────────────────────

class TestCommandExecutor:
    def test_allowed_command_succeeds(self) -> None:
        executor = CommandExecutor(allowed_commands=["echo", "pwd"])
        result = executor.execute("echo hello")
        assert "hello" in result

    def test_disallowed_command_blocked(self) -> None:
        executor = CommandExecutor(allowed_commands=["echo"])
        result = executor.execute("rm -rf /")
        assert "not in whitelist" in result

    def test_timeout_returns_error(self) -> None:
        executor = CommandExecutor(allowed_commands=["sleep"])
        result = executor.execute("sleep 100", timeout=0.1)
        assert "timed out" in result

    def test_empty_command(self) -> None:
        executor = CommandExecutor()
        result = executor.execute("")
        assert "empty" in result.lower()

    def test_nonexistent_command(self) -> None:
        executor = CommandExecutor(allowed_commands=["thiscmdshouldnotexist"])
        result = executor.execute("thiscmdshouldnotexist")
        assert "not found" in result.lower()

    def test_allowed_commands_property(self) -> None:
        executor = CommandExecutor(allowed_commands=["ls", "pwd"])
        assert executor.allowed_commands == frozenset({"ls", "pwd"})

    def test_complex_args_preserved(self) -> None:
        executor = CommandExecutor(allowed_commands=["echo"])
        result = executor.execute("echo -n 'complex  args'")
        assert "complex  args" in result  # whitespace preserved


# ── PathValidator ─────────────────────────────────────────────────────────

class TestPathValidator:
    def test_write_inside_allowed(self, tmp_path: Path) -> None:
        validator = PathValidator(write_prefixes=[tmp_path])
        test_file = tmp_path / "test.txt"
        # Should not raise
        validator.assert_can_write(str(test_file))

    def test_write_outside_allowed_blocked(self, tmp_path: Path) -> None:
        validator = PathValidator(write_prefixes=[tmp_path])
        outside = Path("/etc/passwd")
        assert not validator.can_write(str(outside))

    def test_write_blocked_kernel_path(self) -> None:
        validator = PathValidator()
        assert not validator.can_write("/dev/sda")
        assert not validator.can_write("/proc/cpuinfo")

    def test_read_allowed(self, tmp_path: Path) -> None:
        validator = PathValidator(read_prefixes=[tmp_path])
        test_file = tmp_path / "readable.txt"
        test_file.write_text("data")
        assert validator.can_read(str(test_file))

    def test_read_blocked_kernel(self) -> None:
        validator = PathValidator()
        assert not validator.can_read("/sys/class/power_supply")

    def test_assert_can_write_raises(self) -> None:
        validator = PathValidator(write_prefixes=[Path("/tmp")])
        with pytest.raises(PermissionError, match="Write not allowed"):
            validator.assert_can_write("/root/secret.txt")

    def test_assert_can_read_raises(self) -> None:
        validator = PathValidator(read_prefixes=[Path("/tmp")])
        with pytest.raises(PermissionError, match="Read not allowed"):
            validator.assert_can_read("/etc/shadow")
