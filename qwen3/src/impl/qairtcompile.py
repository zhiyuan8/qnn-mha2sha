import concurrent.futures
import glob
import json
import os
import pickle
import subprocess
import sys
import time

from typing import List, Optional, Union
from pathlib import Path

from common.base.basemodelcompile import BaseModelCompile
from common.utilities import logging_util
from common.utilities.logging_util import logger
from common.utilities import model_utils
from common.G2G import change_hardcoding
from common.utilities.nsptargets import NspTargets
from common.utilities.profiler import event_marker
from common.utilities.profiler import EventProfiler
from config import (
    htp_backend_config,
    htp_backend_extensions,
    modelconfig,
    notebookconfig,
)
from impl.dependencymanager import DependencyManager
from impl.stepsanitychecker import StepSanityChecker


class QairtCompile(BaseModelCompile):
    """
    QairtCompile implements the Step 2 notebook pipeline for compiling and preparing LLM models
    (e.g., Llama) for QNN/HTP deployment. It automates the process of model export, splitting,
    conversion, quantization, and context binary generation for multiple Attention Ratios (ARs)
    and model splits.

    Main workflow:
        1. Export models for different ARs (prepare_exports)
        2. Split exported ONNX models into subgraphs (split_model)
        3. Convert MHA to SHA format for each split (mha_to_sha_conversion)
        4. Convert ONNX to QNN DLC format (convert)
        5. Quantize DLCs (quantize)
        6. Generate context binaries for weight sharing (generate_context_binary)

    The class manages environment setup, parallel execution, and file organization for each step.
    """

    # Hardware Configs
    nsp_target = NspTargets.Windows.GEN2

    # Model Configs

    proc_env = os.environ.copy()

    def __init__(self, model_config=None):
        """
        Initialize QairtCompile with model configuration and environment setup.

        Args:
            model_config: Optional custom model configuration. If not provided, uses default.
        """
        logging_util.setup_logging()
        self.soc_id = self.nsp_target.soc_id
        self.dsp_arch = self.nsp_target.dsp_arch
        self.go_parallel = True
        self._notebook_config = notebookconfig.get_config("notebookconfig.json")
        self.QNN_SDK_ROOT = str(self._notebook_config.qnn_sdk_path)
        self.workfolder = str(self._notebook_config.export_dir)
        self.LLAMA_MODELS = self.workfolder + "/models"
        assert (
            os.path.exists(self.QNN_SDK_ROOT) == True
        ), "QNN_SDK_ROOT path does not exist"
        assert (
            os.path.exists(self.LLAMA_MODELS) == True
        ), "LLAMA_MODELS path does not exist"
        self.mha2sha_root = os.getcwd() + "/../../../common/G2G/MHA2SHA"
        self.qnn_env = os.environ.copy()
        self.g2g_env = os.environ.copy()
        self._model_config = model_config if model_config else modelconfig.ModelConfig()
        self.CL = self._model_config.hyperparameter_config.context_length
        self.ARNs = [1, 128]
        self.EXPORT_AR = 2073
        self.EXPORT_CONTEXT_LENGTH = 4096
        self.onnx_name = "llama32"
        self.num_splits = self._model_config.splitter_config.num_splits
        self.splits = range(1, self.num_splits + 1)
        self.arn_list = [arn for arn in self.ARNs for i in self.splits]
        self.split_idxs = [i for arn in self.ARNs for i in self.splits]
        self.setup_env()

    def check_sanity(self):
        """
        Run step sanity checks using StepSanityChecker.
        """
        self._sanity_checker.verify(current_step=self.step)

    def check_dependencies(self):
        """
        Verify all required dependencies using DependencyManager.
        """
        self._dep_manager.verify()

    def _set_env_path_variable(self, var_name: str, path: str):
        """
        Prepend a path to an environment variable in the process environment.

        Args:
            var_name: Name of the environment variable.
            path: Path to prepend.
        """
        if var_name in self.proc_env:
            path = path + os.pathsep + self.proc_env[var_name]
        self.proc_env[var_name] = path

    def setup_env(self):
        os.environ["QNN_SDK_ROOT"] = self.QNN_SDK_ROOT
        print(
            "All task list:",
            [f"ar{arn}-{n}" for arn, n in zip(self.arn_list, self.split_idxs)],
        )

        self.qnn_env["QNN_SDK_ROOT"] = self.QNN_SDK_ROOT
        self.qnn_env["PYTHONPATH"] = (
            self.QNN_SDK_ROOT + "/benchmarks/QNN/:" + self.QNN_SDK_ROOT + "/lib/python"
        )
        self.qnn_env["PATH"] = (
            self.QNN_SDK_ROOT + "/bin/x86_64-linux-clang:" + self.qnn_env["PATH"]
        )
        self.qnn_env["LD_LIBRARY_PATH"] = self.QNN_SDK_ROOT + "/lib/x86_64-linux-clang"
        self.qnn_env["HEXAGON_TOOLS_DIR"] = (
            self.QNN_SDK_ROOT + "/bin/x86_64-linux-clang"
        )
        self.qnn_env["NUM_LAYERS_PER_SPLIT"] = "9"
        self.qnn_env["LLM"] = "1"
        self.qnn_env["split_embedding"] = "1"
        self.qnn_env["split_lmhead"] = "0"
        os.environ = self.qnn_env

        os.makedirs(f"{self.workfolder}/models_ar_n", exist_ok=True)
        self.g2g_env["PYTHONPATH"] = os.pathsep.join(
            [
                self.g2g_env.get("PYTHONPATH", ""),
                os.path.join(self.mha2sha_root, "src/python"),
            ]
        )
        self.g2g_env["PATH"] = os.pathsep.join(
            [self.g2g_env.get("PATH", ""), os.path.join(self.mha2sha_root, "bin")]
        )
        print(f"MHA2SHA tool root set to: {self.mha2sha_root}")

    ## NEXA: gen_ar and prepare_exports are used to cut the model into different ARs.
    def gen_ar(self, arn):
        """
        Export model for a given Attention Ratio (AR) using change_hardcoding.

        Args:
            arn: Attention Ratio value (e.g., 1 or 128).
        """
        change_hardcoding.execute(
            f"{self.LLAMA_MODELS}",
            f"{self.workfolder}/models_ar_n/ar{arn}-cl{self.CL}",
            [
                f" {self.EXPORT_AR},{arn}",
                f" -{self.EXPORT_AR},-1",
                f" {self.EXPORT_CONTEXT_LENGTH},{self.CL}",
                f" {self.EXPORT_CONTEXT_LENGTH-self.EXPORT_AR},{self.CL-arn}",
            ],
        )

    def prepare_exports(self):
        with event_marker(f"prepare-export"):
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(self.ARNs) if self.go_parallel else 1
            ) as executor:
                results = executor.map(self.gen_ar, self.ARNs)
        print(f"Prepare AR128 AR1 export done.")

    ## NEXA: thread_split and split_model are used to split the model into different splits.
    def thread_split(self, arn):
        """
        Split the exported ONNX model for a given AR into subgraphs.

        Args:
            arn: Attention Ratio value.
        """
        name = f"ar{arn}-cl{self.CL}"
        model_export = f"{self.workfolder}/models_ar_n"
        model_artifact = f"{self.workfolder}/artifacts/ar{arn}-cl{self.CL}/"
        os.makedirs(model_artifact, exist_ok=True)

        # create symlink to export
        symlink_src = os.path.join(model_artifact, "src")
        symlink_path = Path(symlink_src)
        if symlink_path.is_symlink():
            os.unlink(symlink_src)
        os.symlink(src=os.path.join(model_export, name), dst=symlink_src)

        os.makedirs(f"{model_artifact}/split_onnx", exist_ok=True)
        test_vector_pickle_type = "pkl"
        print(f"Starting {self.onnx_name}.onnx")
        model_utils.split_onnx(
            onnxfile=f"{model_artifact}/src/onnx/{self.onnx_name}.onnx",
            modelname=name,
            pickle_filedir=os.path.join(
                model_export, f"ar{arn}-cl{self.CL}/test_vectors"
            ),
            num_splits=self.num_splits,
            output_dir=model_artifact,
            split_embedding=True,
            encoding_file=f"{model_artifact}/src/onnx/{self.onnx_name}.encodings",
            using_qairt_workflow=True,
        )
        print(f"Ending {self.onnx_name}.onnx")

    def split_model(self):
        with event_marker(f"split-onnx"):
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(self.ARNs) if self.go_parallel else 1
            ) as executor:
                results = executor.map(self.thread_split, self.ARNs)
        print(f"All onnx model splitted.")

    ## NEXA: thread_g2g and mha_to_sha_conversion are used to convert the model to SHA.
    def thread_g2g(self, arn, split):
        os.chmod(
            os.path.join(self.mha2sha_root, "bin", "mha2sha-onnx-converter"), 0o777
        )
        os.chmod(os.path.join(self.mha2sha_root, "bin", "env_setup.sh"), 0o777)
        if split == 1:
            print(
                "As first split only include embedding layer, so let's skip first split"
            )
            return
        print("arn, split:", arn, split)
        model_artifact = f"{self.workfolder}/artifacts/ar{arn}-cl{self.CL}/"
        split_work_dir = os.path.join(model_artifact, f"{split}_of_{self.num_splits}")
        name = f"ar{arn}-cl{self.CL}_{split}_of_{self.num_splits}"
        os.makedirs(split_work_dir, exist_ok=True)
        sha_folder = f"{split_work_dir}/sha_output/"
        os.makedirs(sha_folder, exist_ok=True)
        name = f"ar{arn}-cl{self.CL}_{split}_of_{self.num_splits}"
        print(f"mha2sha-onnx-converter {name} running...")
        args = [
            "mha2sha-onnx-converter",
            "--sha-export-path",
            sha_folder,
            "--model-name",
            name,
            "--exported-model-encoding-path",
            f"{model_artifact}/src/onnx/{self.onnx_name}.encodings",
            "--exported-model-path",
            f"{model_artifact}/split_onnx/{name}.onnx",
            "--base-llm",
            "llama3",
            "--mha-conv",
            "--nchw-aligned",
        ]
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.g2g_env
        )
        output, error = proc.communicate()
        print(output.decode(), error.decode())
        print(f"mha2sha-onnx-converter {name} done.")

    def mha_to_sha_conversion(self):
        with event_marker(f"mha2sha"):
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(self.arn_list) if self.go_parallel else 1
            ) as executor:
                results = executor.map(self.thread_g2g, self.arn_list, self.split_idxs)
        print(f"All mha2sha convert done.")

    ## NEXA: thread_convert and convert are used to convert the model to DLC.
    def thread_convert(self, arn, split):
        """
        Convert ONNX (or SHA ONNX) to QNN DLC format for a given AR and split.

        Args:
            arn: Attention Ratio value.
            split: Split index.
        """
        model_artifact = f"{self.workfolder}/artifacts/ar{arn}-cl{self.CL}/"
        split_work_dir = os.path.join(model_artifact, f"{split}_of_{self.num_splits}")
        name = f"ar{arn}-cl{self.CL}_{split}_of_{self.num_splits}"
        os.makedirs(split_work_dir, exist_ok=True)
        out_dir = os.path.join(split_work_dir, "converted_model")
        os.makedirs(out_dir, exist_ok=True)

        # create symlink to export
        for src in [f"input_list_{name}.txt", f"test_inputs_{name}"]:
            symlink_input = os.path.join(split_work_dir, src)
            symlink_path = Path(symlink_input)
            if symlink_path.is_symlink():
                os.unlink(symlink_input)
            os.symlink(src=os.path.join(model_artifact, src), dst=symlink_input)

        if split != 1:
            input_onnx = f"{split_work_dir}/sha_output/{name}.onnx"
            quantization_overrides = f"{split_work_dir}/sha_output/{name}.encodings"
        else:
            # mha2sha not applied to fisrt split
            input_onnx = f"{model_artifact}/split_onnx/{name}.onnx"
            quantization_overrides = (
                f"{model_artifact}/src/onnx/{self.onnx_name}.encodings"
            )

        args = [
            self.QNN_SDK_ROOT + "/bin/x86_64-linux-clang/qairt-converter",
            "--input_network",
            input_onnx,
            "--quantization_overrides",
            quantization_overrides,
            "-o",
            f"{out_dir}/{name}.dlc",
        ]
        options = model_utils.get_input_layout(input_onnx, using_qairt_workflow=True)
        for entry in options:
            args += entry

        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.qnn_env
        )
        output, error = proc.communicate()
        print(output.decode(), error.decode())
        print(f"qairt-converter {name} done!")

    def convert(self):
        with event_marker(f"convert-onnx"):
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(self.split_idxs) if self.go_parallel else 1
            ) as executor:
                results = executor.map(
                    self.thread_convert, self.arn_list, self.split_idxs
                )
        print(f"All qairt-converter done.")

    ## NEXA: thread_genlib and quantize are used to quantize the model.
    def thread_genlib(self, arn, split):
        model_artifact = f"{self.workfolder}/artifacts/ar{arn}-cl{self.CL}/"
        split_work_dir = os.path.join(model_artifact, f"{split}_of_{self.num_splits}")
        name = f"ar{arn}-cl{self.CL}_{split}_of_{self.num_splits}"
        os.chdir(split_work_dir)
        out_dir = os.path.join(split_work_dir, "compiled_model")
        os.makedirs(os.path.join(split_work_dir, "compiled_model"), exist_ok=True)

        float_dlc_file = os.path.join(split_work_dir, "converted_model", f"{name}.dlc")
        quantized_dlc_file = os.path.join(out_dir, f"{name}_quantized.dlc")
        ip_list_file = os.path.join(model_artifact, f"input_list_{name}.txt")

        proc = subprocess.Popen(
            [
                self.QNN_SDK_ROOT + "/bin/x86_64-linux-clang/qairt-quantizer",
                "--input_dlc",
                float_dlc_file,
                "--input_list",
                ip_list_file,
                "--output_dlc",
                quantized_dlc_file,
                "--act_bitwidth",
                "16",
                "--bias_bitwidth",
                "32",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.qnn_env,
        )
        output, error = proc.communicate()
        print(output.decode(), error.decode())
        print(f"qairt-quantizer {name} done!")
        os.chdir(self.workfolder)

    def quantize(self):
        with event_marker(f"qairt-quantizer"):
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(self.split_idxs) if self.go_parallel else 1
            ) as executor:
                results = executor.map(
                    self.thread_genlib, self.arn_list, self.split_idxs
                )
        print(f"All qairt-quantizer done.")

    def make_config_file(self, index, folder, src_graphs, soc_id=69, dsp_arch="v79"):
        """
        Generate HTP backend and performance config files for a given split.

        Args:
            index: Split index.
            folder: Output folder for config files.
            src_graphs: List of graph names for this split.
            soc_id: SoC ID for the target device.
            dsp_arch: DSP architecture string.
        """
        htp_config_json = os.path.join(folder, f"HtpConfigFile_API_{index}.json")
        perf_config_json = os.path.join(folder, f"PerfSetting_API_{index}.conf")

        self.soc_id = int(self.soc_id)
        with open(htp_config_json, "w") as f:
            config = {
                "backend_extensions": {
                    "shared_library_path": "libQnnHtpNetRunExtensions.so",
                    "config_file_path": f"{perf_config_json}",
                }
            }

            json.dump(config, f, indent=4)

        with open(perf_config_json, "w") as f:
            config = {
                "graphs": [
                    {
                        "O": 3.0,
                        "vtcm_mb": 8,
                        "graph_names": src_graphs,
                        "fp16_relaxed_precision": 0,
                    }
                ],
                "devices": [
                    {
                        "soc_id": self.soc_id,
                        "dsp_arch": self.dsp_arch,
                        "cores": [
                            {"perf_profile": "burst", "rpc_control_latency": 100}
                        ],
                        "pd_session": "unsigned",
                    }
                ],
                "context": {"weight_sharing_enabled": len(src_graphs) > 1},
                "memory": {"mem_type": "shared_buffer"},
            }
            json.dump(config, f, indent=4)

    ## NEXA: thread_gen_ws_cb and generate_context_binary are used to generate the context binary.
    def thread_gen_ws_cb(self, i):
        """
        Generate context binary for weight sharing for a given split index.

        Args:
            i: Split index.
        """
        ar128_src = f"{self.workfolder}/artifacts/ar128-cl{self.CL}/"
        ar1_src = f"{self.workfolder}/artifacts/ar1-cl{self.CL}/"
        output_dir = f"{self.workfolder}/artifacts/ar128-ar1-cl{self.CL}_conf_files/"
        ctx_output_dir = f"{self.workfolder}/artifacts/serialized_binaries/"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(ctx_output_dir, exist_ok=True)

        src1_split_folder = os.path.join(
            ar128_src, f"{i}_of_{self.num_splits}", "compiled_model"
        )
        src2_split_folder = os.path.join(
            ar1_src, f"{i}_of_{self.num_splits}", "compiled_model"
        )

        src1_graph_name = f"ar128-cl{self.CL}_{i}_of_{self.num_splits}"
        src1_q_dlc = os.path.join(src1_split_folder, f"{src1_graph_name}_quantized.dlc")
        src2_graph_name = f"ar1-cl{self.CL}_{i}_of_{self.num_splits}"
        src2_q_dlc = os.path.join(src2_split_folder, f"{src2_graph_name}_quantized.dlc")

        graph_list = [src1_graph_name, src2_graph_name]
        self.make_config_file(i, output_dir, graph_list, self.soc_id, self.dsp_arch)

        cmd = [
            "qnn-context-binary-generator",
            "--log_level=verbose",
            "--backend",
            "libQnnHtp.so",
            "--model",
            "libQnnModelDlc.so",
            "--input_output_tensor_mem_type",
            "memhandle",
            "--output_dir",
            ctx_output_dir,
            "--config_file",
            f"{output_dir}/HtpConfigFile_API_{i}.json",
            "--binary_file",
            f"weight_sharing_model_{i}_of_{self.num_splits}.serialized",
            "--dlc_path",
            f"{src1_q_dlc},{src2_q_dlc}",
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.qnn_env
        )
        output, error = proc.communicate()
        print(output.decode(), error.decode())
        print(f"#{i} weight sharing model generated")

    def generate_context_binary(self):
        with event_marker(f"gen-binary"):
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=len(self.splits) if self.go_parallel else 1
            ) as executor:
                results = executor.map(self.thread_gen_ws_cb, self.splits)
        print(f"All weight shared qnn-context-binary generated.")

    def summarize(self):
        """
        Print and dump profiling statistics for the compilation process.
        """
        EventProfiler().report()
        EventProfiler().json_dump(os.path.join(self.workfolder, "profiling_stats.json"))
