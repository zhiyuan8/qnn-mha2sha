import os
import json

from .dependencymanager import DependencyManager
from .stepsanitychecker import StepSanityChecker
from common.base.basemodelexecute import BaseModelExecute
from common.utilities import logging_util
from common.utilities.adb_wrapper import Adb
from common.utilities.logging_util import logger
from common.utilities.nsptargets import NspTargets
from config import htp_backend_config
from config import htp_backend_extensions
from config import modelconfig
from config import notebookconfig


class QairtExecute(BaseModelExecute):
    """Class containing implementation for Llama3 execution on target."""

    _nsp_target = nsp_target = NspTargets.Android.GEN4

    def __init__(self, model_config=None):

        logging_util.setup_logging()
        self._context_binary_paths = []
        self._backend_config_dirs = []
        self._weight_sharing = True
        self._usr_prompt = ""
        self._sys_prompt = ""
        self._notebook_config = notebookconfig.get_config("notebookconfig.json")
        self._dep_manager = DependencyManager(
            step=self.step, nb_cfg=self._notebook_config
        )
        self._model_config = model_config if model_config else modelconfig.ModelConfig()
        self._sanity_checker = StepSanityChecker(self.step, self._model_config)

        _lib_pfx = "lib"
        _lib_sfx = ".so"
        libs_dir = f"{self._notebook_config.qnn_sdk_path}/lib/aarch64-android/"
        skel = (
            f"{self._notebook_config.qnn_sdk_path}/lib/hexagon-{self._nsp_target.dsp_arch}/unsigned/"
            f"{_lib_pfx}{self._nsp_target.qnn_htp_lib_name}Skel{_lib_sfx}"
        )
        self.nb_config_dir = os.path.join(
            os.getcwd(), "../../config/"
        )  # TODO remove relative path
        self.genie_config_file = f"{self.nb_config_dir}genie/llama3-htp-gqa.json"
        self._assets = [
            f"{libs_dir}{_lib_pfx}QnnGenAiTransformerModel{_lib_sfx}",
            f"{libs_dir}{_lib_pfx}QnnGenAiTransformer{_lib_sfx}",
            f"{libs_dir}{_lib_pfx}QnnHtpNetRunExtensions{_lib_sfx}",
            f"{self._notebook_config.qnn_sdk_path}/bin/aarch64-android/genie-t2t-run",
            f"{libs_dir}{_lib_pfx}Genie{_lib_sfx}",
            f"{libs_dir}{_lib_pfx}QnnSystem{_lib_sfx}",
            f"{libs_dir}{_lib_pfx}QnnHtp{_lib_sfx}",
            f"{libs_dir}{_lib_pfx}{self._nsp_target.qnn_htp_lib_name}Stub{_lib_sfx}",
            f"{skel}",
            f"{self.nb_config_dir}genie/htp_backend_ext_config.json",
            f"{self.nb_config_dir}/htp_backend_config.json",
            f"{self.nb_config_dir}genie/tokenizer.json",
        ]
        self.adb_wrapper = Adb(
            self._notebook_config.device_id, self._notebook_config.adb_executable
        )

    def init_target(self):
        logger.info(f"Detected ADB Devices: {self.adb_wrapper.get_devices()}")

    def generate_genie_config(self):
        # modify genie_config_path, include tokenizer.json path, context length size, context binary name, then save this genie_config_file
        context_binaries = []
        for i in range(self._model_config.splitter_config.num_splits):
            context_binaries.append(os.path.basename(self._context_binary_paths[i]))
        json_data = {}
        if os.path.exists(self.genie_config_file):
            with open(self.genie_config_file, encoding="utf-8") as f:
                json_data = json.load(f)
                json_data["dialog"]["context"][
                    "size"
                ] = self._model_config.hyperparameter_config.context_length
                json_data["dialog"]["engine"]["model"]["binary"][
                    "ctx-bins"
                ] = context_binaries
                json_data["dialog"]["tokenizer"][
                    "path"
                ] = f"{self._notebook_config.target_dir}{os.sep}tokenizer.json"
        else:
            logger.error(f"missing input files: {self.genie_config_file}")
        save_genie_config_dir = os.path.join(os.getcwd(), "genie_config")
        if not os.path.exists(save_genie_config_dir):
            os.makedirs(save_genie_config_dir)
        save_genie_config_file = os.path.join(
            save_genie_config_dir, os.path.basename(self.genie_config_file)
        )
        with open(save_genie_config_file, "w") as file:
            json.dump(json_data, file, indent=4)
        self._assets.append(save_genie_config_file)

    def setup_target(self):
        tar_dir = self._notebook_config.target_dir
        self.adb_wrapper.shell(f"rm -r {tar_dir}")
        self.adb_wrapper.shell(f"mkdir {tar_dir}")
        logger.info(f"Pushing assets: {self._context_binary_paths}")
        # check self._assets file must exist
        for a in self._assets:
            if not os.path.exists(a):
                logger.error(f"missing input files: {a}")
        # push input files
        self.generate_genie_config()
        for a in self._assets:
            self.adb_wrapper.push(a, tar_dir)
        for i in range(self._model_config.splitter_config.num_splits):
            self.adb_wrapper.push(self._context_binary_paths[i], tar_dir)

    def _create_cli_str(self) -> str:
        tar_dir = self._notebook_config.target_dir
        t2t_run_bin = f"{tar_dir}{os.sep}genie-t2t-run"
        model_config_path = (
            f"{tar_dir}{os.sep}{os.path.basename(self.genie_config_file)}"
        )
        context_binaries = ""
        for i in range(self._model_config.splitter_config.num_splits):
            context_binaries += (
                f"{tar_dir}{os.sep}{os.path.basename(self._context_binary_paths[i])},"
            )

        cli_str = (
            f"export ADSP_LIBRARY_PATH={tar_dir} && export LD_LIBRARY_PATH={tar_dir} && "
            f"cd {tar_dir} && "
            f'{t2t_run_bin} -p "{self._get_prompt()}" '
            f"-c {model_config_path} "
        )
        logger.info(f"CLI String:\n{cli_str}")
        return cli_str

    def run(self):
        logger.info(f"Running on device [{self._notebook_config.device_id}]\n")
        logger.info(f"With prompt {self._get_prompt()}\n")
        self.adb_wrapper.shell(self._create_cli_str())

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def print_configuration(self):
        logger.info(self._notebook_config)

    def print_assets(self):
        logger.info(self._assets)

    def check_sanity(self):
        self._sanity_checker.verify(current_step=self.step)

    def check_deps(self):
        self._dep_manager.verify()

    def set_user_prompt(self, p: str):
        self._usr_prompt = p

    def set_sys_prompt(self, p: str):
        self._sys_prompt = f"<|start_header_id|>system<|end_header_id|> {p} <|eot_id|>"

    def _get_prompt(self):
        return f"<|begin_of_text|>{self._sys_prompt}<|start_header_id|>user<|end_header_id|>{self._usr_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    def setup_inputs(self):
        ctx_bin_dir_path = (
            self._notebook_config.export_dir / "artifacts/serialized_binaries"
        )
        backend_config_dir_path = self._notebook_config.export_dir
        model_name = ""
        if self._model_config.mode == modelconfig.Mode.weightsharing:
            model_name = "weight_sharing"
            backend_config_dir_path = backend_config_dir_path / "weightsharing"
        ctx_bins = []
        for f in ctx_bin_dir_path.iterdir():
            if (
                f.is_file()
                and f.name.startswith(model_name)
                and f.name.endswith(".serialized.bin")
            ):
                ctx_bins.append(str(f))
        self._context_binary_paths = sorted(ctx_bins)

        logger.info(f"Context binary paths: {self._context_binary_paths}")
