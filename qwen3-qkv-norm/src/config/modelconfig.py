from pydantic import BaseModel, ValidationError, field_validator, Field, FilePath, ConfigDict
from pathlib import Path
from pydantic_core.core_schema import FieldValidationInfo
from common.utilities.logging_util import logger
from typing import Optional, Union, List
import os
from enum import Enum


class Mode(str, Enum):
    weightsharing = 'weightsharing'


class SplitterConfig(BaseModel):
    num_splits: int = Field(default=1)
    split_embedding: bool = Field(default=False)


class ConverterOptions(BaseModel):
    input_network: Optional[FilePath] = ''
    output_path: Optional[FilePath] = ''
    input_list: Optional[FilePath] = ''
    act_bw: int = Field(default=16)
    bias_bw: int = Field(default=32)
    quantization_overrides: Optional[FilePath] = ''


class ModelLibGeneratorOptions(BaseModel):
    c: Optional[FilePath] = ''                           # Filepath to qnn model .cpp file
    b: Optional[FilePath] = ''                           # Filepath to qnn model .bin file
    t: str = Field(default="x86_64-linux-clang")         # Specifies target to build models for
    o: Optional[FilePath] = ''                           # Output location


class ContextBinaryGeneratorOptions(BaseModel):
    model: FilePath = ''
    backend: FilePath = Field(default="libQnnHtp.so")
    output_dir: FilePath = ''
    binary_file: FilePath = ''
    config_file: FilePath = ''
    log_level: str = Field(default="verbose")


class HyperparameterConfig(BaseModel):
    context_length: int = 4096
    eos_id: int = 128001
    bos_id: int = 128000
    tokenid_ph: int = 2
    logits_dim: int = 32000
    layers: int = 28


class ModelConfig(BaseModel):
    """Dataclass representing the model specific json config"""
    model_config = ConfigDict(protected_namespaces=())
    splitter_config: SplitterConfig = Field(default=SplitterConfig())
    converter_options: ConverterOptions = Field(default=ConverterOptions())
    model_lib_gen_options: ModelLibGeneratorOptions = Field(default=ModelLibGeneratorOptions())
    context_bin_gen_options: ContextBinaryGeneratorOptions = Field(default=ContextBinaryGeneratorOptions())
    hyperparameter_config: HyperparameterConfig = Field(default=HyperparameterConfig())
    mode: Mode = Field(default=Mode.weightsharing)
    model_name: str = Field(default="llama32")


def build_mha2sha_cli(onnxfile, encoding_file, model_name, output_dir, mha2sha_tools_dir) -> str:
    cli_str = [mha2sha_tools_dir + "/mha2sha-onnx-converter",
                            "--sha-export-path", output_dir,
                            "--model-name", model_name,
                            "--exported-model-encoding-path", encoding_file,
                            "--exported-model-path", onnxfile,
                            "--base-llm", "llama2",
                            "--mha-conv",
                            "--nchw-aligned"]
    return cli_str

def build_converter_cli(cfg: ModelConfig, onnx_file, cpp_file, input_layout,
                        artifacts, hex_tools_dir, quant_encoding_path) -> str:
    conv = cfg.converter_options
    cli_str = [hex_tools_dir + "/qnn-onnx-converter",
               "--input_network", onnx_file,
               "--output_path", cpp_file,
               "--input_list", f"{artifacts['input_list']}",
               "--act_bw", str(conv.act_bw),
               "--bias_bw", str(conv.bias_bw),
               "--quantization_overrides", str(quant_encoding_path)]
    for entry in input_layout:
        cli_str += entry
    return cli_str

def build_qairt_converter_cli(onnx_file, output_dlc_file, input_layout, hex_tools_dir, quant_encoding_path) -> str:
    cli_str = [hex_tools_dir + "/qairt-converter",
                        "--input_network", onnx_file,
                        "--quantization_overrides", quant_encoding_path,
                        "-o", output_dlc_file
                        ]        
    for entry in input_layout:
        cli_str += entry
    return cli_str

def build_model_lib_gen_cli(name, export_dir, hex_tools_dir) -> str:
    cli_str = [hex_tools_dir + "/qnn-model-lib-generator",
               "-c", os.path.join(export_dir, "converted", f"{name}.cpp"),
               "-b", os.path.join(export_dir, "converted", f"{name}.bin"),
               "-t", "x86_64-linux-clang",
               "-o", os.path.join(export_dir, "compiled")
               ]
    return cli_str

def build_qairt_quant_cli(cfg: ModelConfig, input_dlc, input_list_file, output_dlc_file, hex_tools_dir) -> str:
    options = cfg.converter_options
    cli_str = [hex_tools_dir + "/qairt-quantizer",
               "--input_dlc", input_dlc,
                "--input_list", input_list_file,
                "--output_dlc", output_dlc_file,
                "--act_bitwidth", str(options.act_bw),
                "--bias_bitwidth", str(options.bias_bw)
               ]
    return cli_str


def build_weightsharing_context_bin_gen_cli(prefix_model_name, kv_model_name, export_dir, hex_tools_dir,
                                            config_file: Path, split_num, num_splits) -> str:
    model_1_path = os.path.join(export_dir, "compiled", "x86_64-linux-clang", f'lib{prefix_model_name}.so')
    model_2_path = os.path.join(export_dir, "compiled", "x86_64-linux-clang", f'lib{kv_model_name}.so')
    output_binary_name = f'weightshare_{split_num}_of_{num_splits}.serialized'

    cli_str = [hex_tools_dir + "/qnn-context-binary-generator",
               "--model", f'{model_1_path},{model_2_path}',
               "--backend", "libQnnHtp.so",
               "--input_output_tensor_mem_type", "memhandle",
               "--output_dir",  os.path.join(export_dir, "serialized_binaries"),
               "--binary_file", f"{output_binary_name}",
               "--config_file", str(config_file)
               ]
    return cli_str


def build_context_bin_gen_cli(name, export_dir, hex_tools_dir,
                                          config_file: Path) -> str:

    cli_str = [hex_tools_dir + "/qnn-context-binary-generator",
               "--model", os.path.join(export_dir, "compiled", "x86_64-linux-clang", f"lib{name}.so"),
               "--backend", "libQnnHtp.so",
               "--output_dir",  os.path.join(export_dir, "serialized_binaries"),
               "--binary_file", f"{name}.serialized",
               "--config_file", str(config_file)
               ]
    return cli_str


def build_qairt_cb_gen_cli(dlc_path_list: List[str], config_file, binary_name, output_dir, hex_tools_dir):
    dlc_path = ""
    for entry in dlc_path_list:
        dlc_path += entry + ","
    cli_str = [hex_tools_dir + "/qnn-context-binary-generator",
               "--log_level=error",
                "--backend","libQnnHtp.so",
                "--model", "libQnnModelDlc.so",
                "--input_output_tensor_mem_type", "memhandle",
                "--output_dir", output_dir,
                "--config_file",config_file,
                "--binary_file", binary_name,
                "--dlc_path", dlc_path[:-1]]
    return cli_str
