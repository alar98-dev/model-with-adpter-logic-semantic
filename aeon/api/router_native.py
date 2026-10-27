from typing import Optional
from fastapi import APIRouter, Request, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from aeon.core.hybrid_stream import BicameralSSEEngine

router = APIRouter(prefix="/api/v1", tags=["Bicameral Stream"])

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 256

def get_engine(request: Request) -> BicameralSSEEngine:
    return request.app.state.bicameral_engine

@router.post("/chat")
async def chat_streaming_endpoint(
    body: ChatRequest,
    engine: BicameralSSEEngine = Depends(get_engine),
):
    """
    Endpoint SSE Bicameral nativo: streaming tensorial contínuo
    com comutação dinâmica de câmaras neurais em tempo real.
    """
    stream_generator = engine.stream_bicameral(
        prompt=body.prompt,
        max_new_tokens=body.max_tokens,
    )

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",  # Impede buffering em proxies reversos NGINX
    }

    return StreamingResponse(
        stream_generator,
        media_type="text/event-stream",
        headers=headers,
    )
