import os

from pydantic import BaseModel, field_validator, ValidationError
from pydantic_core.core_schema import FieldValidationInfo
from typing import Optional

from common.utilities.logging_util import logger


class QairtSdkVersion(BaseModel):
    version: str


## NEXA: This is OS requirements, OS Reqs
class OSReqs(BaseModel):
    """Dataclass capturing versioning information related to the operating system."""
    platform: str
    os: str
    version: str


class PrepareDeps(BaseModel):
    """Requirements for the model prepare notebook"""
    qairt_sdk_version: QairtSdkVersion
    os_version: OSReqs
    aimet_version: Optional[str] = None
    weights_version: Optional[str] = None
    gpu_driver_version: Optional[str] = None

    @field_validator('aimet_version')
    def aimet_required(cls, v: str, info: FieldValidationInfo):
        return v

    @field_validator('weights_version')
    def weights_required(cls, v: str, info: FieldValidationInfo):
        return v

    @field_validator('gpu_driver_version')
    def gpu_driver_required(cls, v: str, info: FieldValidationInfo):
        return v


class CompileDeps(BaseModel):
    """Requirements for the model compile notebook"""
    qairt_sdk_version: QairtSdkVersion
    os_version: OSReqs


class ExecuteDeps(BaseModel):
    """Requirements for the model execute notebook"""
    qairt_sdk_version: QairtSdkVersion
    os_version: OSReqs
    android_platform_tools_version: Optional[str] = None

    @field_validator('android_platform_tools_version')
    def android_pt_required(cls, v: str, info: FieldValidationInfo):
        return v


class ModelDeps(BaseModel):
    """Dataclass representing external dependencies required for this model to run."""
    prepare_deps: Optional[PrepareDeps] = None
    compile_deps: CompileDeps
    execute_deps: ExecuteDeps


def get_model_deps(config_json_filename) -> ModelDeps:
    """Deserializes a JSON into the model dependencies"""

    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))
    with open(os.path.join(__location__, config_json_filename), encoding="utf-8") as f:
        json_data = f.read()
        try:
            model_deps = ModelDeps.model_validate_json(json_data)
            logger.info("Read dependencies: %s", model_deps)
            return model_deps
        except ValidationError as e:
            logger.exception(e)
            return None
