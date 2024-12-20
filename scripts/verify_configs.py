from pathlib import Path
import yaml

def verify_configs():
    base_path = Path(__file__).parent.parent
    layout_path = base_path / 'configs' / 'default' / 'layout'
    default_config = base_path / 'configs' / 'default' / 'meri_default.yaml'
    
    print("Verifying configuration structure...")
    
    # Check layout directory
    print("\nLayout configuration files:")
    for file in layout_path.glob('*.yaml'):
        print(f"- {file.name}")
        # Verify each file is readable
        try:
            with open(file) as f:
                yaml.safe_load(f)
            print(f"  ✓ Valid YAML")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
    
    # Check default config
    print("\nDefault configuration:")
    try:
        with open(default_config) as f:
            config = yaml.safe_load(f)
            layout_path = config.get('layout_analysis', {}).get('CONFIG_PATH')
            print(f"Layout config path: {layout_path}")
            
            # Verify the referenced layout config exists
            if layout_path:
                full_path = base_path / 'configs' / 'default' / layout_path
                if full_path.is_file():
                    print("✓ Layout config file exists")
                else:
                    print("✗ Layout config file not found")
    except Exception as e:
        print(f"Error reading default config: {str(e)}")

if __name__ == "__main__":
    verify_configs() 