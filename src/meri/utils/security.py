from pathlib import Path
import yaml
from typing import Any, Dict
import subprocess

def safe_path_resolve(path: str) -> Path:
    """Safely resolve paths to prevent path traversal attacks."""
    return Path(path).resolve()

def safe_yaml_load(file_path: str) -> Dict[str, Any]:
    """Safely load YAML files."""
    with open(safe_path_resolve(file_path)) as f:
        return yaml.safe_load(f)

def safe_subprocess_run(cmd_args: list) -> subprocess.CompletedProcess:
    """Safely run subprocess commands."""
    return subprocess.run(
        cmd_args,
        shell=False,
        check=True,
        capture_output=True,
        text=True
    ) 