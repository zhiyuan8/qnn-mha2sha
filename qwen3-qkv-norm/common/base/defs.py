from enum import Enum


## NEXA: The overall steps or enums of the NotebookSteps.
class NotebookStep(Enum):
    MODEL_PREPARE = 1
    MODEL_COMPILE = 2
    MODEL_EXECUTE = 3
