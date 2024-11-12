from pathlib import Path
import yaml
import os

def print_separator():
    print("\n" + "="*50 + "\n")

def debug_config_paths():
    try:
        # Get absolute path to project root
        base_path = Path(__file__).parent.parent
        print(f"Project root: {base_path}")
        print_separator()

        # List all directories in project root
        print("Project root contents:")
        for item in base_path.iterdir():
            print(f"- {item.name} ({'dir' if item.is_dir() else 'file'})")
        print_separator()

        # Check configs directory
        configs_dir = base_path / 'configs'
        print(f"Configs directory: {configs_dir}")
        print(f"Exists: {configs_dir.exists()}")
        if configs_dir.exists():
            print("\nConfigs directory contents:")
            for item in configs_dir.iterdir():
                print(f"- {item.name} ({'dir' if item.is_dir() else 'file'})")
        print_separator()

        # Check default config directory
        default_dir = configs_dir / 'default'
        print(f"Default config directory: {default_dir}")
        print(f"Exists: {default_dir.exists()}")
        if default_dir.exists():
            print("\nDefault directory contents:")
            for item in default_dir.iterdir():
                print(f"- {item.name} ({'dir' if item.is_dir() else 'file'})")
        print_separator()

        # Check main config file
        main_config_path = default_dir / 'meri_default.yaml'
        print(f"Main config file: {main_config_path}")
        print(f"Exists: {main_config_path.exists()}")
        print(f"Is file: {main_config_path.is_file()}")
        if main_config_path.is_file():
            try:
                with open(main_config_path) as f:
                    config = yaml.safe_load(f)
                print("\nMain config content:")
                print(yaml.dump(config, default_flow_style=False))
            except Exception as e:
                print(f"Error reading main config: {e}")
        print_separator()

        # Check layout directory
        layout_dir = default_dir / 'layout'
        print(f"Layout directory: {layout_dir}")
        print(f"Exists: {layout_dir.exists()}")
        print(f"Is directory: {layout_dir.is_dir()}")
        if layout_dir.is_dir():
            print("\nLayout directory contents:")
            for item in layout_dir.iterdir():
                print(f"- {item.name} ({'dir' if item.is_dir() else 'file'})")
                if item.is_file() and item.suffix in ['.yaml', '.yml']:
                    try:
                        with open(item) as f:
                            yaml.safe_load(f)
                        print(f"  ✓ Valid YAML")
                    except Exception as e:
                        print(f"  ✗ Error: {e}")

    except Exception as e:
        print(f"Error during debug: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_config_paths() 