from pathlib import Path
import deepdoctection as dd
import yaml

def initialize_models():
    # Register layout model
    layout_profile = dd.ModelProfile(
        name="d2_model_0829999_layout_inf_only",
        type="object_detection",
        architecture="cascade_rcnn",
        provider="deepdoctection",
        repo_id="deepdoctection/d2_casc_rcnn_X_32xd4_50_FPN_GN_2FC_publaynet_inference_only",
        size=None,
        dependencies=["pytorch", "detectron2"],
        description="Layout detection model trained on PubLayNet"
    )
    
    # Register table model
    table_profile = dd.ModelProfile(
        name="pubtables_detr",
        type="object_detection",
        architecture="detr",
        provider="deepdoctection",
        repo_id="deepdoctection/pubtables_detr",
        size=None,
        dependencies=["pytorch"],
        description="Table detection model"
    )
    
    # Register the profiles
    dd.ModelCatalog.register("layout", layout_profile)
    dd.ModelCatalog.register("table", table_profile)
    
    print("Models registered successfully!")
    
    # Update configs to use registered names
    base_path = Path(__file__).parent.parent
    config_dir = base_path / 'configs' / 'default' / 'layout'
    
    # Update layout detector config
    layout_config = {
        'PT': {
            'LAYOUT': {
                'WEIGHTS': 'layout',
                'FILTER': []
            }
        },
        'LIB': 'PT',
        'DEVICE': 'cuda'
    }
    
    with open(config_dir / 'layout_detector_config.yaml', 'w') as f:
        yaml.dump(layout_config, f, default_flow_style=False)
    
    # Update table detector config
    table_config = {
        'PT': {
            'LAYOUT': {
                'WEIGHTS': 'table',
                'FILTER': ['table']
            }
        },
        'LIB': 'PT',
        'DEVICE': 'cuda'
    }
    
    with open(config_dir / 'table_detector_config.yaml', 'w') as f:
        yaml.dump(table_config, f, default_flow_style=False)
    
    print("Configuration files updated!")

if __name__ == "__main__":
    initialize_models() 