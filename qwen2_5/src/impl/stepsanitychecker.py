from pathlib import Path
from common.utilities.logging_util import logger
from common.base.basestepsanitychecker import BaseStepSanityChecker, SanityException
from common.base.defs import NotebookStep
from typing import List
from config.modelconfig import ModelConfig
import os
import common.base.defs as defs


class StepSanityChecker(BaseStepSanityChecker):
    """Implementation of StepSanityChecker with implementation specifics to this model"""

    step1_sanity_json_path = "Cache/SanityStep1.json"
    step2_sanity_json_path = "Cache/SanityStep2.json"

    check_sanity = True
    expectedArtifacts: List[str]

    def __init__(self, step: defs.NotebookStep, cfg: ModelConfig):
        if step == defs.NotebookStep.MODEL_COMPILE:
            # self.expected_artifacts = [cfg.onnx_filename, cfg.quantization_encodings, cfg.pickle_filedir]
            pass
        elif step == defs.NotebookStep.MODEL_EXECUTE:
            self.expected_artifacts = ["Model.so", "ContextBinary"]

    def get_required_exports_model_compile(self):
        """Concrete class required to implement step1 getter for required exports"""

    def get_required_exports_model_execute(self):
        """Concrete class required to implement step2 getter for required exports"""

    def verify(self, current_step: defs.NotebookStep):
        try:
            if current_step == defs.NotebookStep.MODEL_COMPILE:
                self._check_step_model_compile()
            elif current_step == defs.NotebookStep.MODEL_EXECUTE:
                self._check_step_model_execute()
            else:
                logger.info("no previous step to check")
        except (OSError, IOError) as e:
            logger.exception(e)
            raise SanityException()

    def _check_step_model_compile(self):
        f = Path(self.step1_sanity_json_path)
        if not f.exists:
            logger.warning("Could not determine if previous step was run")
            if self.check_sanity is True:
                logger.error(
                    "check_sanity is set to true, "
                    "if you wish to override validation of previous "
                    "notebook steps, set check_sanity to false in the config."
                )
                raise FileNotFoundError(
                    "Could not find %s, are you sure you ran the previous step",
                    self.step1_sanity_json_path,
                )

        for art in self.expected_artifacts:
            # if not Path(art).is_file() and not os.path.isdir(art):
            if not os.path.exists(art):
                # raise FileNotFoundError("Did not find required artifact %s from step 1", art)
                logger.error(
                    "<SanityChecker>Did not find required artifact %s from step 1", art
                )
                raise SanityException()
            else:
                logger.info("<SanityChecker>Found required step 1 artifact %s", art)

    def _check_step_model_execute(self):
        # QNN Context binaries and model preparation libs
        pass

    def print_required_exports(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass  # TODO impl saving to JSON
