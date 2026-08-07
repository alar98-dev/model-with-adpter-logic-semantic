# Authorship: ALifer Ramos & Antigravity (AI Co-Architect)
"""
conftest.py - Pytest session fixtures for SWA Vision Loop tests.

Handles hardware device attachment (loop0 / /dev/sde) at session start,
with graceful skip markers for tests that require hardware access.
"""
import os
import subprocess
import pytest

VHDX_PATH    = "/mnt/c/SWA_ROOT.vhdx"
LOOP_DEV     = "/dev/loop0"
SDE_SYMLINK  = "/dev/sde"


def _is_loop_attached() -> bool:
    try:
        result = subprocess.run(
            ["losetup", LOOP_DEV],
            capture_output=True, text=True
        )
        return result.returncode == 0 and VHDX_PATH in result.stdout
    except Exception:
        return False


def _attach_loop_device() -> bool:
    """Attach VHDX to loop0 and create /dev/sde symlink. Returns True on success."""
    try:
        # Try attaching read-only
        subprocess.run(
            ["sudo", "losetup", "-r", LOOP_DEV, VHDX_PATH],
            capture_output=True, check=False
        )
        subprocess.run(
            ["sudo", "ln", "-sf", LOOP_DEV, SDE_SYMLINK],
            capture_output=True, check=True
        )
        subprocess.run(
            ["sudo", "chmod", "666", LOOP_DEV],
            capture_output=True, check=False
        )
        return _is_loop_attached()
    except Exception as e:
        print(f"\n⚠️  Could not attach loop device: {e}")
        return False


@pytest.fixture(scope="session", autouse=True)
def ensure_hardware_device():
    """Session-scoped fixture: attach /dev/loop0 → /dev/sde before any test."""
    attached = _is_loop_attached()
    if not attached:
        print(f"\n🔧 [conftest] /dev/loop0 not attached — attempting to attach {VHDX_PATH}...")
        attached = _attach_loop_device()

    if attached:
        print(f"✅ [conftest] Hardware device ready: {SDE_SYMLINK} → {LOOP_DEV}")
    else:
        print(f"⚠️  [conftest] Hardware device unavailable — hardware tests will be skipped.")

    # Store availability as a session attribute for tests to check
    pytest.hardware_available = os.path.exists(SDE_SYMLINK) and os.path.lexists(SDE_SYMLINK)
    yield
    # No teardown — keep device mounted for the session


def pytest_collection_modifyitems(items):
    """Skip hardware tests when /dev/sde is not available."""
    skip_hw = pytest.mark.skip(reason="/dev/sde not available — run: sudo losetup -r /dev/loop0 /mnt/c/SWA_ROOT.vhdx")
    for item in items:
        if "hardware" in item.name.lower() and not os.path.exists(SDE_SYMLINK):
            item.add_marker(skip_hw)
