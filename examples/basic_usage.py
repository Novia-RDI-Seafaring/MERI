from pathlib import Path
import deepdoctection as dd
from meri import MERI
# from meri.configs import MERI_CONFIGS_PATH
from meri.config.config_manager import ConfigManager

# Initialize config manager
config_manager = ConfigManager()
MERI_CONFIGS_PATH = config_manager.get_layout_config_dir()

import matplotlib.pyplot as plt


def main():
    try:     
        project_root = Path(__file__).parent.parent
        pdf_path = project_root / "data/demo_data/Alfa Laval LKH.pdf"
        
        # Use the config manager to get the correct path
        meri = MERI(
            pdf_path=str(pdf_path),
            config_yaml_path='meri_custom.yaml'  # Just use the filename, ConfigManager will handle the path
        )
        print("MERI initialized successfully!")
        
        dps, page_dicts = meri.layout_analysis()
        print("Layout analysis completed successfully!")
        print(f"Processed {len(page_dicts)} pages")
         
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()