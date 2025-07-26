from pydantic import BaseModel, DirectoryPath, ValidationError, field_validator, Field
from pydantic_core.core_schema import FieldValidationInfo
import json
from common.utilities.logging_util import logger
from typing import List
from pathlib import Path
import os
from typing import Optional


## NEXA: If we just use the field_validator, it is likely doing nothing.
class BackendExtensions(BaseModel):
    """Model for the backend extensions json"""
    shared_library_path: Path = Field(default=Path("libQnnHtpNetRunExtensions.so"))
    config_file_path: Path = Field(default=Path("htp_config.json"))

    @field_validator('shared_library_path')
    def shared_library_path_valid(cls, v: str, info: FieldValidationInfo):
        return v

    @field_validator('config_file_path')
    def config_file_path_valid(cls, v: str, info: FieldValidationInfo):
        return v


class BackendExtensionsConfig(BaseModel):
    backend_extensions: BackendExtensions = Field(default=BackendExtensions())


def get_config(config_json_filename) -> BackendExtensionsConfig:
    """Deserializes the HTP backend extensions JSON"""
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    with open(os.path.join(__location__, config_json_filename), encoding="utf-8") as f:
        json_data = f.read()
        try:
            cfg = BackendExtensionsConfig.model_validate_json(json_data)
            logger.info("Read json config: %s", cfg)
            return cfg
        except ValidationError as e:
            logger.exception(e)


def write_config_to_json(dirname, cfg: BackendExtensionsConfig) -> Path:
    os.makedirs(dirname, exist_ok=True)
    config_location = os.path.join(dirname, "htp_backend_extensions.json")
    with open(config_location, "w", encoding="utf-8") as f:
        f.write(cfg.model_dump_json())
    return config_location
