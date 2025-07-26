import abc
import common.base.defs as defs


class CompileException(Exception):
    pass


class BaseModelCompile(metaclass=abc.ABCMeta):
    step = defs.NotebookStep.MODEL_COMPILE
    """Base class that enforces implementation of expected notebook workflows"""
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate_context_binary(self):
        """Enforces implementation and use of the context binary generator tool
          for a particular model"""
