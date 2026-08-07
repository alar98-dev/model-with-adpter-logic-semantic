"""Tests for swa.config."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from swa.config import SWAConfig, get_config, set_config


class TestSWAConfig:
    def test_defaults_are_sane(self) -> None:
        cfg = SWAConfig()
        assert cfg.gemma_device in ("cpu", "cuda", "xpu")
        assert cfg.hardware_state_size_gb == 24
        assert cfg.rest_port == 20000

    def test_hardware_fallback_file_auto_path(self) -> None:
        cfg = SWAConfig(hardware_state_fallback="file")
        assert cfg.hardware_state_fallback_path is not None
        assert "swa_hardware_fallback" in str(cfg.hardware_state_fallback_path)

    def test_model_dir_resolution(self) -> None:
        cfg = SWAConfig(model_dir=Path("models/gemma-3-4-b"))
        resolved = cfg.resolved_model_dir
        assert resolved.is_absolute()
        assert "models/gemma-3-4-b" in str(resolved)

    def test_absolute_model_dir_stays_absolute(self) -> None:
        cfg = SWAConfig(model_dir=Path("/custom/models/my-model"))
        assert str(cfg.resolved_model_dir) == "/custom/models/my-model"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SWA_CPU_TOPOLOGY", "dual_xeon")
        monkeypatch.setenv("REST_PORT", "30000")
        cfg = SWAConfig()
        assert cfg.swa_cpu_topology == "dual_xeon"
        assert cfg.rest_port == 30000

    def test_token_warning_on_placeholder(self) -> None:
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SWAConfig(hf_token="hf_your_token_here")
            token_warnings = [x for x in w if "HF_TOKEN" in str(x.message)]
            assert len(token_warnings) == 1

    def test_allowed_write_prefixes_default_type(self) -> None:
        cfg = SWAConfig()
        assert len(cfg.allowed_write_prefixes) >= 1
        assert all(isinstance(p, Path) for p in cfg.allowed_write_prefixes)


class TestGetSetConfig:
    def test_global_get_config(self) -> None:
        cfg = get_config()
        assert isinstance(cfg, SWAConfig)

    def test_set_config_overrides_global(self) -> None:
        custom = SWAConfig(rest_port=9999)
        set_config(custom)
        assert get_config().rest_port == 9999
        # Reset for other tests
        set_config(SWAConfig())
