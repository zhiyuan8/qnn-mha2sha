import abc


class DependencyException(Exception):
    pass


class BaseDependencyManager(metaclass=abc.ABCMeta):
    """Base class to manage dependencies, including installation and verification"""

    @abc.abstractmethod
    def __init__(self, nb_cfg):
        """Initialize dependency manager using configuration"""

    @abc.abstractmethod
    def install(self):
        """Enforces implementation of installation of dependencies"""

    @abc.abstractmethod
    def verify(self):
        """Enforces implementation of verification of dependencies"""

    @abc.abstractmethod
    def _install_pip_deps(self):
        """Base implementation that pip installs requirements.txt"""
