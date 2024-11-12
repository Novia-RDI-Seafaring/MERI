from ..config.config_manager import ConfigManager
from ..extraction.extractor import JsonExtractor
from ..utils.format_handler import MarkdownHandler
from ..transformation.transformer import DocumentTransformer, Format
from ..layout.layout_detector import LayoutDetector
import deepdoctection as dd
import yaml
import os
from PIL import Image
import pdf2image
from pathlib import Path


class MERI:

    def __init__(self, pdf_path: str, config_yaml_path: str = None) -> None:
        self.config_manager = ConfigManager()
        
        # Load main config
        if config_yaml_path is None:
            config_yaml_path = 'meri_default.yaml'
        self.config = self.config_manager.load_config(config_yaml_path)
        
        self.layout_config = self.config['layout_analysis']
        self.transformer_config = self.config['transformer']
        self.extractor_config = self.config['extractor']
        
        # Get the layout config directory
        layout_config_dir = self.config_manager.base_config_dir / 'default' / 'layout'
        if not layout_config_dir.is_dir():
            raise NotADirectoryError(f"Layout config directory not found: {layout_config_dir}")
        
        self.layout_config_dir = layout_config_dir
        self.pdf_path = pdf_path
        self.temp_pdf_path = None

    def layout_analysis(self):
        from ..layout import LayoutDetector
        
        # Get the specific layout config file path from the config
        layout_config_file = self.layout_config['CONFIG_PATH']
        config_path = self.config_manager.get_config_path(layout_config_file)
        
        # Initialize detector with the layout config directory
        detector = LayoutDetector(pipeline_config_path=str(config_path))
        
        dps, page_dicts = detector.detect(self.pdf_path)
        return dps, page_dicts

    def transform_to_intermediate(self, dps):

         ## create intermedaite format
        annotations_to_merge = [dd.LayoutType.figure, dd.LayoutType.table]
        self.doc_transformer = DocumentTransformer(self.pdf_path, **self.transformer_config['KWARGS'])
        self.doc_transformer.merge_with_annotations(dps, annotations_to_merge)
        self.doc_transformer.docorate_unmatched_textblocks()

        intermediate_format = self.doc_transformer.transform_to(self.transformer_config['FORMAT'])

        return intermediate_format

    def populate_schema(self, json_schema_string, intermediate_format):
        
        if self.transformer_config['FORMAT'] == Format.MARKDOWN.value:
            format_handler = MarkdownHandler(intermediate_format) 
        else:
            raise NotImplementedError

        self.jsonExtractor = JsonExtractor(intermediate_format=format_handler, **self.extractor_config['KWARGS'])
        
        res = self.jsonExtractor.populate_schema(json_schema_string=json_schema_string)
        return res
    
    def run(self, json_schema_string):

        ## make detections
        dps, _ = self.layout_analysis()

        ## transform to intermediate format
        intermediate_format = self.transform_to_intermediate(dps)

        if self.extractor_config['METHOD'] == "populate_schema":
            return self.populate_schema(json_schema_string, intermediate_format)
        
