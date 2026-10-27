import os
import uvicorn
from fastapi import FastAPI
from aeon.api.middleware import ScopeIsolationMiddleware
from aeon.api.router_native import router as native_router

app = FastAPI(title="SWA-Aeon 3.0 Core Organism", version="3.0.0")

# Registra middleware unificado de isolamento de escopo ASGI
app.add_middleware(ScopeIsolationMiddleware)

# Registra roteadores modulares da API
app.include_router(native_router)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "3.0.0",
        "architecture": "Clean Core Bicameral (270M -> 1B)",
        "isometry": "SVD Orthogonal FP32 -> bfloat16",
        "entropy_regulator": "Adaptive Z-Score (N=50, sigma_min=0.05)"
    }

if __name__ == "__main__":
    host = os.environ.get("SWA_CHAT_HOST", "127.0.0.1")
    port = int(os.environ.get("SWA_CHAT_PORT", "20000"))
    uvicorn.run("aeon.main:app", host=host, port=port, reload=False)
