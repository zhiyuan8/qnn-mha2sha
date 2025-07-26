from pydantic import BaseModel, ValidationError, field_validator, DirectoryPath, Field, FilePath
from pydantic_core.core_schema import FieldValidationInfo
from common.utilities.logging_util import logger
from pathlib import Path
import os


class NotebookConfigException(Exception):
    pass


class NotebookConfig(BaseModel):
    """Dataclass representing the model specific json config"""
    qnn_sdk_path: DirectoryPath
    export_dir: Path = Field(default=Path(os.getcwd()) / "export")

    @field_validator('qnn_sdk_path')
    def qnn_sdk_path_valid(cls, v: str, info: FieldValidationInfo):
        return v


def get_config(config_json_filename) -> NotebookConfig:
    """Deserializes a JSON into a model config object"""
    config_dir = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    with open(os.path.join(config_dir, config_json_filename), encoding="utf-8") as f:
        json_data = f.read()
        try:
            model_config = NotebookConfig.model_validate_json(json_data)
            logger.info("Read llama config %s", model_config)
            return model_config
        except ValidationError as e:
            logger.exception(e)
            raise e
