import functools
import logging
from swa.semantic_protocol import SWAEvent

def verify_integrity(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        result = await func(*args, **kwargs)
        if result is not None and not isinstance(result, SWAEvent):
            error_msg = f"INTEGRITY_VIOLATION: Organ {func.__qualname__} returned non-SWAEvent data: {type(result)}"
            logging.error(error_msg)
            raise TypeError(error_msg)
        return result
    return wrapper
