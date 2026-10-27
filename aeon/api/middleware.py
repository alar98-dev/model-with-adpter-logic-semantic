import contextvars
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Variáveis de contexto assíncronas para isolamento rígido por requisição
request_context_var = contextvars.ContextVar("request_context", default={})

class ScopeIsolationMiddleware(BaseHTTPMiddleware):
    """
    Middleware que assegura o isolamento rigoroso do contexto de cada requisição HTTP/ASGI,
    prevenindo vazamento de variáveis de escopo ou UnboundLocalErrors em rotas concorrentes.
    """
    async def dispatch(self, request: Request, call_next):
        token = request_context_var.set({
            "client_host": request.client.host if request.client else "127.0.0.1",
            "path": request.url.path
        })
        try:
            response = await call_next(request)
            return response
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content={"error": "Internal SWA Core Error", "detail": str(exc)}
            )
        finally:
            request_context_var.reset(token)
