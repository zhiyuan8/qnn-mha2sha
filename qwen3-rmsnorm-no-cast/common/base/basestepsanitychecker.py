import abc
from common.utilities.logging_util import logger
import common.base.defs as defs


class SanityException(Exception):
    pass

## NEXA: This script is used to check if current step can work with previous step done.
class BaseStepSanityChecker(metaclass=abc.ABCMeta):
    """Base class for concrete model implementations to extend with particulars around
        validation between notebook execution steps."""
    @abc.abstractmethod
    def __init__(self, model):
        """Load a sanity json report"""

    @abc.abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Save a sanity json report to disk"""

    @abc.abstractmethod
    def print_required_exports(self, step: defs.NotebookStep):
        """Print model assets of a particular step"""

    def verify(self, current_step: defs.NotebookStep):
        if current_step == defs.NotebookStep.MODEL_COMPILE:
            self._check_step_model_compile()
        elif current_step == defs.NotebookStep.MODEL_EXECUTE:
            self._check_step_model_execute()
        else:
            logger.warning("No previous step to check")

    @abc.abstractmethod
    def _check_step_model_compile(self):
        """Concrete class to overload and check for previous step continuity"""

    @abc.abstractmethod
    def _check_step_model_execute(self):
        """Concrete class to overload and check for previous step continuity"""
