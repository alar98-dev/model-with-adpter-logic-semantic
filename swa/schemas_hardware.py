from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel


class HardwareStatusResponse(BaseModel):
    requested_device: str
    validated_device: str
    fallback_active: bool
    fallback_reason: str
    xpu_available: bool
    ipex_available: bool
    torch_xpu_available: bool
    backend_state: str
    details: Dict[str, Any]

