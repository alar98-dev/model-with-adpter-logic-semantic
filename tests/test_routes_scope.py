import asyncio
import pytest
from aeon.api.middleware import request_context_var

def test_concurrent_context_isolation():
    """
    Valida que requisições assíncronas concorrentes mantém variáveis de contexto
    isoladas via contextvars sem interpenetração de escopo.
    """
    async def worker(task_id: int):
        token = request_context_var.set({"id": task_id})
        await asyncio.sleep(0.01)
        val = request_context_var.get()
        assert val["id"] == task_id, f"Vazamento de escopo detectado! Esperado {task_id}, obtido {val['id']}"
        request_context_var.reset(token)

    async def main():
        tasks = [worker(i) for i in range(20)]
        await asyncio.gather(*tasks)

    asyncio.run(main())
