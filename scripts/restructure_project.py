import os
import shutil
from pathlib import Path
from datetime import datetime

def create_directory_structure(base_path):
    """Create the new directory structure"""
    directories = [
        'configs/default',
        'configs/custom',
        'configs/schema',
        'data',  # Existing
        'docker',  # Existing
        'docs/api',
        'docs/user_guide',
        'examples',
        'scripts',
        'src/meri/core',
        'src/meri/config',
        'src/meri/extraction',  # Existing
        'src/meri/layout',      # Existing
        'src/meri/transformation', # Existing
        'src/meri/utils',       # Existing
        'tests/unit',
        'tests/integration',
        'tests/data'
    ]
    
    for directory in directories:
        Path(base_path / directory).mkdir(parents=True, exist_ok=True)

def migrate_files(base_path):
    """Migrate files to their new locations"""
    # 1. Move configuration files
    old_config_path = base_path / 'src' / 'meri' / 'configs'
    new_config_path = base_path / 'configs' / 'default'
    
    if old_config_path.exists():
        # Move YAML configs
        for config_file in old_config_path.glob('*.yaml'):
            shutil.copy2(config_file, new_config_path)
        
        # Move layout configs
        if (old_config_path / 'layout').exists():
            layout_dest = new_config_path / 'layout'
            layout_dest.mkdir(exist_ok=True)
            for layout_file in (old_config_path / 'layout').glob('*'):
                if layout_file.is_file():
                    shutil.copy2(layout_file, layout_dest)

    # 2. Move demo files to examples
    demo_path = base_path / 'demo'
    examples_path = base_path / 'examples'
    if demo_path.exists():
        for demo_file in demo_path.glob('*.py'):
            new_name = demo_file.name.replace('demo_', '')
            shutil.copy2(demo_file, examples_path / new_name)

    # 3. Move core MERI class
    meri_source = base_path / 'src' / 'meri' / 'meri.py'
    meri_dest = base_path / 'src' / 'meri' / 'core' / 'meri.py'
    if meri_source.exists():
        shutil.copy2(meri_source, meri_dest)

    # 4. Create new config manager
    create_config_manager(base_path)

def create_config_manager(base_path):
    """Create the new config manager file"""
    config_manager_path = base_path / 'src' / 'meri' / 'config' / 'config_manager.py'
    
    config_manager_content = '''
from pathlib import Path
import yaml
import os

class ConfigManager:
    def __init__(self, base_config_dir=None):
        if base_config_dir is None:
            # Default to configs directory in project root
            self.base_config_dir = Path(__file__).parent.parent.parent.parent / 'configs'
        else:
            self.base_config_dir = Path(base_config_dir)
        
    def get_config_path(self, config_name, config_type='default'):
        """Get configuration file path with fallback to default"""
        custom_path = self.base_config_dir / 'custom' / config_name
        default_path = self.base_config_dir / 'default' / config_name
        
        if custom_path.exists():
            return custom_path
        return default_path

    def load_config(self, config_name, config_type='default'):
        """Load configuration with fallback to default"""
        config_path = self.get_config_path(config_name, config_type)
        with open(config_path) as f:
            return yaml.safe_load(f)
    '''
    
    with open(config_manager_path, 'w') as f:
        f.write(config_manager_content.strip())

def create_init_files(base_path):
    """Create __init__.py files in new directories"""
    init_directories = [
        'src/meri/core',
        'src/meri/config',
    ]
    
    for directory in init_directories:
        init_file = base_path / directory / '__init__.py'
        init_file.touch()

def backup_project(base_path):
    """Create a backup of the project inside the project directory"""
    # Create backup directory inside the project
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir = base_path / f"backup_{timestamp}"
    
    try:
        # Copy everything except the backup directory itself
        os.makedirs(backup_dir)
        for item in base_path.iterdir():
            if item != backup_dir and item.name != '.git':  # Skip .git directory to save space
                if item.is_dir():
                    shutil.copytree(item, backup_dir / item.name)
                else:
                    shutil.copy2(item, backup_dir)
        print(f"Backup created at: {backup_dir}")
        return True
    except Exception as e:
        print(f"Backup failed: {str(e)}")
        return False

def cleanup_old_structure(base_path):
    """Remove old directories and files after successful migration"""
    directories_to_remove = [
        'demo',                          # old demo directory
        'src/meri/configs',              # old configs directory
        'src/meri/meri.py',             # old meri.py file
        'layout_eval',                   # moving to evaluation/layout
        'extracted_parameters.json',     # should be in data directory
    ]
    
    for dir_path in directories_to_remove:
        path = base_path / dir_path
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path)
                print(f"Removed directory: {dir_path}")
            else:
                path.unlink()
                print(f"Removed file: {dir_path}")

def verify_migration(base_path):
    """Verify that critical files have been migrated successfully"""
    critical_paths = [
        'configs/default',
        'src/meri/core/meri.py',
        'src/meri/config/config_manager.py',
        'examples'
    ]
    
    all_exist = True
    for path in critical_paths:
        if not (base_path / path).exists():
            print(f"ERROR: {path} is missing!")
            all_exist = False
    
    return all_exist

def main():
    # Get the project root directory
    base_path = Path.cwd()
    
    # Create backup
    print("Creating backup...")
    if not backup_project(base_path):
        print("Aborting due to backup failure")
        return
    
    # Create new directory structure
    print("Creating new directory structure...")
    create_directory_structure(base_path)
    
    # Migrate files
    print("Migrating files...")
    migrate_files(base_path)
    
    # Create __init__.py files
    print("Creating __init__.py files...")
    create_init_files(base_path)
    
    # Verify migration
    print("\nVerifying migration...")
    if verify_migration(base_path):
        print("Migration verification successful!")
        
        # Cleanup old structure
        print("\nCleaning up old structure...")
        cleanup_old_structure(base_path)
        
        print("\nProject restructuring complete!")
        print("\nNext steps:")
        print("1. Review the migrated files")
        print("2. Update import statements in your code")
        print("3. Test the application")
        print(f"4. Your backup is available in the 'backup_[timestamp]' directory")
    else:
        print("\nMigration verification failed!")
        print("Please check the errors above and try again.")
        print("No cleanup was performed to preserve existing files.")

if __name__ == "__main__":
    main() 