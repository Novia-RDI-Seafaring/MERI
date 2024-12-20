from pathlib import Path
import shutil

def verify_package_structure():
    base_path = Path(__file__).parent.parent
    src_path = base_path / 'src' / 'meri'
    
    # Ensure all necessary directories exist
    directories = [
        'core',
        'config',
        'extraction',
        'layout',
        'transformation',
        'utils'
    ]
    
    for dir_name in directories:
        dir_path = src_path / dir_name
        dir_path.mkdir(exist_ok=True)
        init_file = dir_path / '__init__.py'
        if not init_file.exists():
            init_file.touch()
    
    # Create main __init__.py
    init_content = '''
from .core.meri import MERI
from .config.config_manager import ConfigManager

__all__ = ['MERI', 'ConfigManager']
'''
    
    with open(src_path / '__init__.py', 'w') as f:
        f.write(init_content.strip())
    
    # Create core/meri.py if it doesn't exist
    meri_path = src_path / 'core' / 'meri.py'
    if not meri_path.exists():
        # Copy from old location if it exists
        old_meri = src_path / 'meri.py'
        if old_meri.exists():
            shutil.copy2(old_meri, meri_path)
            old_meri.unlink()  # Remove old file

    print("Package structure verified and fixed!")

if __name__ == "__main__":
    verify_package_structure() 