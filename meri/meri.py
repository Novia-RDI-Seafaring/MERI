from .extraction.extractor import JsonExtractor
from .layout.pipeline_components.utils import CONFIGS_PATH
from .utils.format_handler import MarkdownHandler
from .transformation.transformer import DocumentTransformer, Format
from .layout import LayoutDetector
import deepdoctection as dd
import json
import yaml
import os


class MERI:

    def __init__(self, pdf_path: str, config_yaml_path) -> None:

        assert os.path.exists(pdf_path)

        with open(config_yaml_path, 'r') as file:

            config = yaml.safe_load(file)

        self.layout_config = config['layout_analysis']
        self.transformer_config = config['transformer']
        self.extractor_config = config['extractor']

        self.pdf_path=pdf_path


    def layout_analysis(self):
        
        pipeline_config_path = os.path.abspath(self.layout_config['CONFIG_PATH'])
        detector = LayoutDetector(pipeline_config_path=pipeline_config_path)

        dps, page_dicts = detector.detect(self.pdf_path)

        return dps, page_dicts

    def transform_to_intermediate(self, dps):

         ## create intermedaite format
        annotations_to_merge = [dd.LayoutType.figure, dd.LayoutType.table]
        doc_transformer = DocumentTransformer(self.pdf_path, **self.transformer_config['KWARGS'])
        doc_transformer.merge_with_annotations(dps, annotations_to_merge)
        doc_transformer.docorate_unmatched_textblocks()

        intermediate_format = doc_transformer.transform_to(self.transformer_config['FORMAT'])

        return intermediate_format

    def populate_schema(self, json_schema_string, intermediate_format):
        
        if self.transformer_config['FORMAT'] == Format.MARKDOWN.value:
            format_handler = MarkdownHandler(intermediate_format) 
        else:
            raise NotImplementedError

        jsonExtractor = JsonExtractor(intermediate_format=format_handler, **self.extractor_config['KWARGS'])
        prompt = f""" 
        You are expert in understanding technical information from documents. You are provided with 
        a document as markdown and are required to extract specific information from the document.

        document as markdown: {intermediate_format}

        Return the results as JSON.
        """
        res = jsonExtractor.populate_schema(json_schema_string=json_schema_string,custom_prompt=prompt)
        return res
    
    def run(self, json_schema_string):

        ## make detections
        dps, _ = self.layout_analysis()

        ## transform to intermediate format
        intermediate_format = self.transform_to_intermediate(dps)

        if self.extractor_config['METHOD'] == "populate_schema":
            return self.populate_schema(json_schema_string, intermediate_format)
        
