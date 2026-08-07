from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional
from swa.cpu import CPUTopology



def _safe_bool(x: Any) -> bool:
    return bool(x) if x is not None else False


@dataclass(frozen=True)
class BackendResolution:
    requested_device: str
    validated_device: str
    fallback_active: bool
    fallback_reason: str
    backend_state: str  # "ready" | "degraded"
    details: Dict[str, Any]


class HardwareBackend:
    name: str

    def detect(self) -> Dict[str, Any]:
        raise NotImplementedError

    def health(self) -> Dict[str, Any]:
        raise NotImplementedError

    def capabilities(self) -> Dict[str, Any]:
        return {}


class CPUBackend(HardwareBackend):
    name = "cpu"

    def __init__(self) -> None:
        self.topology = CPUTopology()

    def detect(self) -> Dict[str, Any]:
        cores_info = []
        for core in self.topology.cores:
            cores_info.append({
                "core_id": core.core_id,
                "physical_id": core.physical_id,
                "logical_processors": core.logical_processors,
                "cache": {
                    "l1": core.l1_cache,
                    "l2": core.l2_cache,
                    "l3": core.l3_cache
                },
                "frequency": core.frequency,
                "state": core.eventual_cycle_state
            })
        return {
            "available": True,
            "physical_core_count": len(self.topology.cores),
            "logical_core_count": sum(len(c.logical_processors) for c in self.topology.cores),
            "cores": cores_info,
            "average_utilization": self.topology.utilization_average,
            "average_temperature": self.topology.temperature_average
        }

    def health(self) -> Dict[str, Any]:
        diags = self.topology.run_diagnostics()
        healthy_cores = sum(1 for d in diags if d["status"] == "healthy")
        all_ok = healthy_cores == len(self.topology.cores)
        return {
            "available": True,
            "state": "healthy" if all_ok else "degraded",
            "healthy_cores": healthy_cores,
            "total_cores": len(self.topology.cores),
            "diagnostics": diags
        }



class IntelXPUBackend(HardwareBackend):
    name = "xpu"

    def detect(self) -> Dict[str, Any]:
        import torch

        has_xpu_mod = hasattr(torch, "xpu")
        is_available = bool(has_xpu_mod and torch.xpu.is_available())

        name = None
        eu_count = None
        pci_vendor_id = None
        pci_device_id = None
        xpu_props_error = None

        if is_available:
            try:
                props = torch.xpu.get_device_properties(0)
                eu_count = getattr(props, "gpu_eu_count", None)
                name = getattr(props, "name", None)
                pci_vendor_id = getattr(
                    props,
                    "pci_vendor_id",
                    getattr(props, "vendor_id", None),
                )
                pci_device_id = getattr(
                    props,
                    "pci_device_id",
                    getattr(props, "device_id", None),
                )
            except Exception as e:
                xpu_props_error = str(e)

        return {
            "has_torch_xpu_module": has_xpu_mod,
            "xpu_available": is_available,
            "device_name": name,
            "eu_count": eu_count,
            "pci_vendor_id": pci_vendor_id,
            "pci_device_id": pci_device_id,
            "xpu_props_error": xpu_props_error,
        }

    def health(self) -> Dict[str, Any]:
        det = self.detect()
        # Minimal health = detect says available. (Other checks live in details.)
        return {"available": det.get("xpu_available", False)}

    def capabilities(self) -> Dict[str, Any]:
        # Keep lightweight and defensive: only infer bf16 if possible.
        import torch

        bf16 = False
        try:
            if hasattr(torch, "bfloat16"):
                bf16 = True
        except Exception:
            bf16 = False
        return {"bf16": bf16}


class FakeBackend(HardwareBackend):
    name = "fake"

    def detect(self) -> Dict[str, Any]:
        return {"available": True}

    def health(self) -> Dict[str, Any]:
        return {"available": True}


class HardwareManager:
    """Hardware validation + resolution.

    Design goals:
    - never raise during detect/health
    - always return a resolution object with fallback reasoning
    - requested_device vs validated_device separated
    """

    def __init__(self) -> None:
        self._last_detection: Dict[str, Any] = {}

    def detect(self) -> Dict[str, Any]:
        det: Dict[str, Any] = {}
        try:
            det["cpu"] = CPUBackend().detect()
        except Exception as e:
            det["cpu_error"] = str(e)

        try:
            det["xpu"] = IntelXPUBackend().detect()
        except Exception as e:
            det["xpu_error"] = str(e)

        # IPEX availability (best-effort)
        try:
            try:
                import intel_extension_for_pytorch  # noqa: F401

                det["ipex_available"] = True
            except Exception:
                det["ipex_available"] = False
        except Exception as e:
            det["ipex_error"] = str(e)

        # oneAPI/Level Zero evidence (best-effort via env and /dev/dri)
        det["dev_dri_present"] = os.path.exists("/dev/dri")
        det["dev_dri_ls"] = None
        try:
            if det["dev_dri_present"]:
                det["dev_dri_ls"] = os.listdir("/dev/dri")
        except Exception:
            det["dev_dri_ls"] = "permission_or_io_error"

        det["env"] = {
            "GEMMA_DEVICE": os.environ.get("GEMMA_DEVICE"),
            "HARDWARE_POLICY": os.environ.get("HARDWARE_POLICY"),
        }

        self._last_detection = det
        return det

    def health(self) -> Dict[str, Any]:
        det = self.detect()
        # Normalize known flags
        xpu_available = bool(det.get("xpu", {}).get("xpu_available", False))
        ipex_available = bool(det.get("ipex_available", False))

        # If xpu module is absent or not available, health is degraded.
        return {
            "xpu_available": xpu_available,
            "ipex_available": ipex_available,
            "torch_xpu_available": xpu_available,
            "backend_state": "ready" if xpu_available else "degraded",
            "details": det,
        }

    def resolve(self, requested_device: str) -> BackendResolution:
        req = (requested_device or "cpu").strip().lower()
        policy = (os.environ.get("HARDWARE_POLICY", "fallback") or "fallback").strip().lower()

        det = self.detect()
        xpu_ok = bool(det.get("xpu", {}).get("xpu_available", False))
        ipex_ok = bool(det.get("ipex_available", False))

        # For now: we validate XPU only by torch.xpu.is_available().
        # ipex_ok is still surfaced for observability.
        validated = req
        fallback_active = False
        fallback_reason = ""
        backend_state = "ready"

        if req in ("xpu", "intel", "igpu"):
            if xpu_ok:
                validated = "xpu"
                fallback_active = False
                backend_state = "ready"
            else:
                validated = "cpu"
                fallback_active = True
                backend_state = "degraded"
                fallback_reason = "XPU runtime unavailable"

                if policy == "strict":
                    # In strict mode, we still return a degraded resolution but let caller decide.
                    fallback_reason = "XPU runtime unavailable (strict policy)"

        elif req in ("cpu",):
            validated = "cpu"
            fallback_active = False
            backend_state = "ready"
        else:
            validated = "cpu"
            fallback_active = True
            backend_state = "degraded"
            fallback_reason = f"Unknown requested device: {req}"

        resolution_details = {
            "requested_device": req,
            "xpu_available": xpu_ok,
            "ipex_available": ipex_ok,
            "torch_xpu_available": xpu_ok,
            "diagnostic": det,
            "policy": policy,
        }

        return BackendResolution(
            requested_device=req,
            validated_device=validated,
            fallback_active=fallback_active,
            fallback_reason=fallback_reason,
            backend_state=backend_state,
            details=resolution_details,
        )


# Singleton accessor (cheap; safe to call multiple times)
_hardware_manager: Optional[HardwareManager] = None


def get_hardware_manager() -> HardwareManager:
    global _hardware_manager
    if _hardware_manager is None:
        _hardware_manager = HardwareManager()
    return _hardware_manager


def hardware_status_payload() -> Dict[str, Any]:
    """Return JSON-ready payload for /api/v1/internal/hardware-status."""
    req = os.environ.get("GEMMA_DEVICE", "cpu")
    res = get_hardware_manager().resolve(req)

    # Ensure required fields from user request
    xpu_av = bool(res.details.get("xpu_available", False))
    ipex_av = bool(res.details.get("ipex_available", False))

    return {
        "requested_device": res.requested_device,
        "validated_device": res.validated_device,
        "fallback_active": res.fallback_active,
        "fallback_reason": res.fallback_reason,
        "xpu_available": xpu_av,
        "ipex_available": ipex_av,
        "torch_xpu_available": bool(res.details.get("torch_xpu_available", xpu_av)),
        "backend_state": res.backend_state,
        "details": {
            # keep structured but not huge; diagnostic is included.
            "policy": res.details.get("policy"),
            "diagnostic": res.details.get("diagnostic"),
        },
    }

