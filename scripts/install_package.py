import subprocess
import sys
from pathlib import Path

def install_package():
    project_root = Path(__file__).parent.parent
    
    # Install package in development mode
    subprocess.check_call([
        sys.executable, 
        "-m", 
        "pip", 
        "install", 
        "-e", 
        str(project_root)
    ])

if __name__ == "__main__":
    install_package() 