import os
from pathlib import Path

def fix_deepdoctection_import():
    """
    Fix the deepdoctection import issue by replacing cached_download with hf_hub_download
    """
    site_packages = Path(__import__('deepdoctection').__file__).parent
    model_file = site_packages / 'extern/model.py'
    
    if model_file.exists():
        with open(model_file, 'r') as f:
            content = f.read()
        
        # Replace the import
        content = content.replace(
            'from huggingface_hub import cached_download, hf_hub_url',
            'from huggingface_hub import hf_hub_download as cached_download, hf_hub_url'
        )
        
        with open(model_file, 'w') as f:
            f.write(content)
        
        print("Fixed deepdoctection import!")

if __name__ == "__main__":
    fix_deepdoctection_import() 