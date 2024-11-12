from pathlib import Path
import yaml
import os

class ConfigManager:
    def __init__(self):
        self.base_config_dir = Path(__file__).parent.parent.parent.parent / 'configs'
        
    def get_config_path(self, config_name, config_type='default'):
        """Get full path for a config file"""
        if isinstance(config_name, Path):
            config_name = str(config_name)
            
        # Handle layout configs specially
        if config_name.startswith('layout/'):
            config_path = self.base_config_dir / config_type / 'layout' / config_name[7:]
        else:
            config_path = self.base_config_dir / config_type / config_name
            
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        return config_path
    
    def load_config(self, config_name, config_type='default'):
        """Load configuration with fallback to default"""
        config_path = self.get_config_path(config_name, config_type)
        with open(config_path) as f:
            return yaml.safe_load(f)