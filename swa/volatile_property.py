import builtins
import logging
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SWA-VolatileProperty")

class VolatileProperty(object):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None: return self
        if self.fget is None: raise AttributeError("unreadable attribute")
        return self.fget(obj)

    def __set__(self, obj, value):
        if self.fset is None: raise AttributeError("can not set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        if self.fdel is None: raise AttributeError("can not delete attribute")
        self.fdel(obj)

    def getter(self, fget):
        self.fget = fget
        return self

    def setter(self, fset):
        self.fset = fset
        return self

    def deleter(self, fdel):
        self.fdel = fdel
        return self

def replace_builtin_property_volatile():
    # Apenas substitui se não for uma propriedade do multiprocessing (que usa o builtin original)
    # Como o multiprocessing usa decoradores, substituir globalmente quebra o sistema.
    # A recomendação é usar @aimd_property explicitamente nos órgãos e NÃO substituir globalmente.
    pass

# Mantemos a classe mas removemos a substituição global agressiva.
