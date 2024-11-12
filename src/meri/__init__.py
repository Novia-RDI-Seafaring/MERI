from .core.meri import MERI
from .config.config_manager import ConfigManager

config_manager = ConfigManager()
MERI_CONFIGS_PATH = config_manager.get_layout_config_dir()

__all__ = ["MERI", "MERI_CONFIGS_PATH", "ConfigManager"]