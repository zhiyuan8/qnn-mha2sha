from pydantic import BaseModel, ValidationError, field_validator
import json
from common.utilities.logging_util import logger
from typing import List
from pathlib import Path
import os
from typing import Optional


class Core(BaseModel):
    """Dataclass representing core configuration for the HTP backend"""
    core_id: int
    perf_profile: str
    rpc_control_latency: int


class Graph(BaseModel):
    vtcm_mb: Optional[int]
    graph_names: List[str]
    # fp16_relaxed_precision: Optional[int]
    # hvx_threads: Optional[int]
    O: Optional[int]                           # Graph optimization value, 1-3
    # dlbc: Optional[int]                        # deep learning bw compression value


class Device(BaseModel):
    soc_id: str
    dsp_arch: str
    cores: List[Core]

    @field_validator("soc_id", mode='before')
    def coerce_to_str(cls, value) -> str:
        return str(value)


class Context(BaseModel):
    weight_sharing_enabled: bool


class HtpBackendConfig(BaseModel): 
    """Dataclass representing an HTP config.
      Please refer to QNN SDK documentation for more detail on the HTP backend config."""
    graphs: List[Graph]
    devices: List[Device]
    context: Context


def get_config(config_json_filename) -> HtpBackendConfig:
    """Deserializes the HTP backend config JSON"""
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

    with open(os.path.join(__location__, config_json_filename), encoding="utf-8") as f:
        json_data = f.read()
        try:
            cfg = HtpBackendConfig.model_validate_json(json_data)
            logger.info("Read json config: %s", cfg)
            return cfg
        except ValidationError as e:
            logger.exception(e)


def write_config_to_json(dirname, cfg: HtpBackendConfig) -> Path:
    os.makedirs(dirname, exist_ok=True)
    config_location = os.path.join(dirname, "htp_config.json")
    with open(config_location, "w", encoding="utf-8") as f:
        f.write(cfg.model_dump_json())
    return config_location
