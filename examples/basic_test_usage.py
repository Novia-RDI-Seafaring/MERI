from meri import MERI
from pathlib import Path

def initialize_deepdoctection():
    # First register the model profile
    dd.ModelCatalog.register("microsoft/table-transformer-detection", dd.ModelProfile(
        name="microsoft/table-transformer-detection",
        description="Microsoft Table Transformer for table detection",
        size=[115511753],
        tp_model=False,
        config="microsoft/table-transformer-detection/config.json",
        preprocessor_config="microsoft/table-transformer-detection/preprocessor_config.json",
        categories={"1": dd.LayoutType.table},
        dl_library="PT",
        model_wrapper="HFDetrDerivedDetector"
    ))

    # Then initialize analyzer with the registered model
    analyzer = dd.get_dd_analyzer(
        reset_config_file=True, 
        config_overwrite=[
            "USE_LAYOUT=False",
            "PT.ITEM.WEIGHTS=microsoft/table-transformer-detection",
            "PT.ITEM.FILTER=['table']",
            "PT.ITEM.PAD.TOP=5",
            "PT.ITEM.PAD.RIGHT=5",
            "PT.ITEM.PAD.BOTTOM=5",
            "PT.ITEM.PAD.LEFT=5",
            "USE_OCR=True",
            "DEVICE='cuda'"
        ]
    )
    print("Deepdoctection initialized successfully!")
    return analyzer
def main():
    # Get paths relative to project root
    project_root = Path(__file__).parent.parent
    pdf_path = project_root / "data/demo_data/Alfa Laval LKH.pdf"
    config_path = project_root / "configs/default/meri_default.yaml"
    
    # Initialize MERI
    meri = MERI(
        pdf_path=str(pdf_path),
        config_yaml_path=str(config_path)
    )
    
    # Test basic functionality
    dps, page_dicts = meri.layout_analysis()
    print("Layout analysis completed successfully!")

if __name__ == "__main__":
    main()