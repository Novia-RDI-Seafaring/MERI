from pathlib import Path
import yaml

def fix_model_paths():
    base_path = Path(__file__).parent.parent
    config_dir = base_path / 'configs' / 'default' / 'layout'
    
    # Model path mappings
    model_mappings = {
        'layout/d2_model_0829999_layout_inf_only.pt': 'deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet_inference_only',
        'https://huggingface.co/deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet_inference_only/resolve/main/d2_model_0829999_layout_inf_only.pt': 'deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet_inference_only',
        'table/pubtables_detr.pt': 'deepdoctection/pubtables_detr'
    }
    
    # Process each yaml file
    for config_file in config_dir.glob('*.yaml'):
        print(f"Processing {config_file.name}")
        
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
            
            # Update weights path if present
            if config and isinstance(config, dict):
                if 'PT' in config and 'LAYOUT' in config['PT'] and 'WEIGHTS' in config['PT']['LAYOUT']:
                    old_path = config['PT']['LAYOUT']['WEIGHTS']
                    if old_path in model_mappings:
                        config['PT']['LAYOUT']['WEIGHTS'] = model_mappings[old_path]
                        print(f"Updated weights path in {config_file.name}")
                        
                        # Save updated config
                        with open(config_file, 'w') as f:
                            yaml.dump(config, f, default_flow_style=False)
            
        except Exception as e:
            print(f"Error processing {config_file.name}: {str(e)}")

if __name__ == "__main__":
    fix_model_paths() 