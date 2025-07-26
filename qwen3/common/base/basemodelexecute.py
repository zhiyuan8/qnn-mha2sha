import abc
import common.base.defs as defs


class ExecutionException(Exception):
    pass


class BaseModelExecute(metaclass=abc.ABCMeta):
    step = defs.NotebookStep.MODEL_EXECUTE
    """Base class that enforces implementation of expected notebook workflows"""
    @abc.abstractmethod
    def __init__(self):
        pass

    @abc.abstractmethod
    def setup_target(self):
        """Sets up the runtime target, organizing a runtime directory. """

    @abc.abstractmethod
    def run(self):
        """Execution method on target. """

    @abc.abstractmethod
    def print_configuration(self):
        """Prints command line configuration. """

    @abc.abstractmethod
    def print_assets(self):
        """Prints a list of libraries that are necessary for execution on target. """

