from swa.aimd_controller import controller
from swa.broadcast_controller import broadcaster
import asyncio

class aimd_property(object):
    def __init__(self, fget=None, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, objtype=None):
        if obj is None: return self
        window = controller.on_success()
        if self.fget: return self.fget(obj)
        return None

    def __set__(self, obj, value):
        if self.fset: self.fset(obj, value)
        else: raise AttributeError("can not set attribute")
        
    def setter(self, fset):
        return aimd_property(self.fget, fset)
        
    def trigger_error(self):
        window = controller.on_error()
        try:
            loop = asyncio.get_event_loop()
            loop.create_task(broadcaster.broadcast("systemicEvent", {"type": "AIMD_THROTTLE", "window": window}))
        except:
            pass
