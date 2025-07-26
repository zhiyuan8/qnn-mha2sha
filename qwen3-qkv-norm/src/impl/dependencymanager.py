import distro
import os
import platform
import shutil
import subprocess
import sys
import yaml

from dependencies import deps
from pathlib import Path
from packaging.version import Version

from common.base.basedependencymanager import BaseDependencyManager, DependencyException
from common.base.defs import NotebookStep
from config.notebookconfig import NotebookConfig
from common.utilities.logging_util import logger


class DependencyManager(BaseDependencyManager):
    """Manages dependencies for this model. Dependencies are verified, and installed
    based on the ModelDeps.json. Each notebook is evaluated separately."""

    _step = None

    def __init__(self, step: NotebookStep, nb_cfg: NotebookConfig):
        self._step = step
        dep_config = deps.get_model_deps("deps.json")
        self.notebook_config = nb_cfg

        if step == NotebookStep.MODEL_COMPILE:
            self.dep_config = dep_config.compile_deps
        elif step == NotebookStep.MODEL_EXECUTE:
            self.dep_config = dep_config.execute_deps

    def _verify_qairt_sdk(self):
        """Reads the QAIRT SDK yaml file for validation"""
        required_qnn_version = Version(self.dep_config.qairt_sdk_version.version)
        if not os.path.exists(self.notebook_config.qnn_sdk_path):
            logger.error(
                "Could not find configured QNN SDK path, %s",
                self.notebook_config.qnn_sdk_path,
            )
        with open(
            f"{self.notebook_config.qnn_sdk_path}{os.path.sep}sdk.yaml",
            encoding="utf-8",
        ) as stream:
            try:
                sdk_yaml = yaml.safe_load(stream)
            except yaml.YAMLError as e:
                raise DependencyException(
                    "Failed to validate QAIRT SDK from yaml file."
                )

        logger.info(
            "<DependencyManager> Found QAIRT Version %s, recommended %s",
            sdk_yaml["version"],
            required_qnn_version,
        )
        if Version(sdk_yaml["version"]) < required_qnn_version:
            raise DependencyException(
                "<DependencyManager> QAIRT SDK version is insufficient"
            )
        return True

    def _verify_aimet(self):
        """Reads the version file in the aimet package for validation"""
        req_aimet_ver = self.dep_config.aimet_version
        aimet_path = Path(self.notebook_config.aimet_path, "packaging", "version.txt")
        try:
            with open(aimet_path, encoding="utf-8") as stream:
                aimet_version = stream.readline()
                logger.info(aimet_version)
        except FileNotFoundError as e:
            raise DependencyException("Failed to validate AIMET package.")

    def _verify_gpu(self):
        """Reads the gpu driver version as an output from nvcc for validatoin"""
        try:
            process = subprocess.run(
                [
                    "nvcc",
                    "--version",
                    "|",
                    "egrep",
                    "-o",
                    '"V[0-9]+.[0-9]+.[0-9]+"',
                    "|",
                    "cut -c2-",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            cuda_version, err = process.communicate()
            logger.info("CUDA Version found: %s", cuda_version)
            # TODO verify against config
            if self.dep_config.gpu_driver_version == cuda_version:
                logger.info("<DependencyManager> Found exact GPU driver match")
            else:
                logger.warning(
                    "<DependencyManager> Mismatch of GPU driver found on system and model "
                    "requirements. This may impact performance or functionality. "
                    "Found: %s, recommended:%s,",
                    cuda_version,
                    self.dep_config.gpu_driver_version,
                )
        except FileNotFoundError as e:
            raise DependencyException("Failed to validate GPU driver.")

    def _verify_os(self):
        """Wrapper around verification of the operating system"""
        try:
            system = platform.system()
            if system == "Linux":
                self._verify_linux()
            elif system == "Windows":
                self._verify_windows()
            else:
                logger.warning(
                    "<DependencyManager> Unable to dynamically determine OS, read: %s",
                    sys,
                )
        except Exception as e:  # TODO specific exception
            raise DependencyException("Failed to validate OS")

    def _verify_linux(self):
        """Verifies that this Linux host machine is compatible to execute this notebook"""
        if (
            distro.id().lower() == self.dep_config.os_version.os.lower()
            and distro.version() == self.dep_config.os_version.version
        ):
            logger.info(
                "<DependencyManager> Host OS was found to be compatible for this model, "
                "%s, %s",
                distro.id(),
                distro.version(),
            )
        else:
            logger.warning(
                "<DependencyManager> Host OS was not found to be an exact match."
                " Found:%s, %s, recommended: %s, %s",
                distro.id(),
                distro.version(),
                self.dep_config.os_version.os,
                self.dep_config.os_version.version,
            )

    @staticmethod
    def _verify_windows(self):
        """Verifies that this Windows host machine is compatible to execute this notebook"""
        return False

    def verify(self):
        """ "Verifies necessary notebook dependencies"""
        logger.info("Verifying Notebook %s software dependencies", self._step)
        try:
            self._verify_os()
            if self._step == NotebookStep.MODEL_PREPARE:
                self._verify_aimet()
                self._verify_gpu()
            elif self._step == NotebookStep.MODEL_COMPILE:
                self._verify_qairt_sdk()
            elif self._step == NotebookStep.MODEL_EXECUTE:
                self._verify_adb()
            logger.info("<DependencyManager> Dependency verification complete")
        except DependencyException as e:
            logger.exception(e)

    def _verify_adb(self):
        if os.path.exists(self.notebook_config.adb_executable):
            pass
        elif shutil.which("adb") is not None:
            self.notebook_config.adb_executable = shutil.which("adb")
        else:
            raise DependencyException(
                f"Could not call adb on this system, or at location:"
                f" {self.notebook_config.adb_executable}. Please either install adb or specify its"
                f" install location in config/notebookconfig.json"
            )
        logger.info(
            f"<DependencyManager> ADB found on system {self.notebook_config.adb_executable}"
        )

    def install(self):
        pass

    def _install_pip_deps(self):
        logger.info("Installing python requirements via pip...")
        deps = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        os.path.join(deps, "../Dependencies/requirements.txt")
        assert os.path.exists(deps)
        try:
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    "Dependencies/requirements.txt",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            logger.exception(e)
