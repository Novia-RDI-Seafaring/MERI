from pathlib import Path
import shutil
import yaml

def diagnose_and_fix_layout_configs():
    base_path = Path(__file__).parent.parent
    
    # Check old layout configs
    old_layout_path = base_path / 'src' / 'meri' / 'configs' / 'layout'
    new_layout_path = base_path / 'configs' / 'default' / 'layout'
    
    print("Checking configuration paths...")
    print(f"Old layout path exists: {old_layout_path.exists()}")
    print(f"New layout path exists: {new_layout_path.exists()}")
    
    # Create new layout directory if it doesn't exist
    new_layout_path.mkdir(parents=True, exist_ok=True)
    
    # List all files in old layout directory if it exists
    if old_layout_path.exists():
        print("\nFiles in old layout directory:")
        for file in old_layout_path.glob('*'):
            print(f"- {file.name}")
            if file.is_file() and file.suffix in ['.yaml', '.yml']:
                # Copy file to new location
                shutil.copy2(file, new_layout_path / file.name)
                print(f"Copied {file.name} to new location")
    
    # Check default config
    default_config_path = base_path / 'configs' / 'default' / 'meri_default.yaml'
    if default_config_path.exists():
        with open(default_config_path) as f:
            config = yaml.safe_load(f)
            print("\nCurrent layout config path in meri_default.yaml:")
            print(config.get('layout_analysis', {}).get('CONFIG_PATH', 'Not found'))
    
    # List all files in new layout directory
    print("\nFiles in new layout directory:")
    for file in new_layout_path.glob('*'):
        print(f"- {file.name}")

if __name__ == "__main__":
    diagnose_and_fix_layout_configs() 