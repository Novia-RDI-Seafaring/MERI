import os
import sys
import json
import fitz
import numpy as np
import yaml
from PIL import Image, ImageDraw
import gradio as gr

import deepdoctection as dd
from meri.layout.pipeline_components import (AddPDFInfoComponent, 
                        DummyDetectorComponent, 
                        LayoutDetectorComponent,
                        OCRComponent,
                        DrawingsDetectorComponent,
                        ImageDetectorComponent,
                        TableDetectorComponent,
                        WordUnionComponent,
                        NMSComponent,
                        TextDetectorComponent,
                        TablePlumberComponent)
from meri.layout.pipeline import Pipeline
from meri.layout.pipeline_components.utils import ProcessingService, CONFIGS_PATH
from meri.utils.format_handler import MarkdownHandler
from meri.extraction.extractor import JsonExtractor
from meri.transformation.transformer import DocumentTransformer, Format

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '../MERI')))

class DocumentProcessor:
    @staticmethod
    def yaml_to_markdown(yaml_content):
        def dict_to_markdown(d, level=0):
            markdown_str = ""
            indent = "  " * level
            for key, value in d.items():
                if isinstance(value, dict):
                    markdown_str += f"{indent}- **{key}**:\n"
                    markdown_str += dict_to_markdown(value, level + 1)
                elif isinstance(value, list):
                    markdown_str += f"{indent}- **{key}**:\n"
                    for item in value:
                        if isinstance(item, dict):
                            markdown_str += dict_to_markdown(item, level + 1)
                        else:
                            markdown_str += f"{indent}  - {item}\n"
                else:
                    markdown_str += f"{indent}- **{key}**: {value}\n"
            return markdown_str

        def list_to_markdown(lst, level=0):
            markdown_str = ""
            indent = "  " * level
            for item in lst:
                if isinstance(item, dict):
                    markdown_str += dict_to_markdown(item, level)
                else:
                    markdown_str += f"{indent}- {item}\n"
            return markdown_str

        try:
            yaml_content = yaml.safe_load(yaml_content)
            if isinstance(yaml_content, dict):
                markdown_content = dict_to_markdown(yaml_content)
            elif isinstance(yaml_content, list):
                markdown_content = list_to_markdown(yaml_content)
            else:
                return f"Invalid YAML content: Parsed content is neither a dictionary nor a list. It is {type(yaml_content)}."
            return markdown_content
        except yaml.YAMLError as e:
            return f"Error parsing YAML content: {e}"

    @staticmethod
    def display_yaml_file(use_default, file):
        if use_default:
            pipeline_config_path = os.path.abspath(os.path.join(CONFIGS_PATH, 'good_pipeline.yaml'))
            try:
                with open(pipeline_config_path, 'r') as f:
                    file_content = f.read()
            except Exception as e:
                return f"Error reading default pipeline file: {e}"
        else:
            if file is None:
                return "No file uploaded."
            try:
                with open(file.name, 'r') as f:
                    file_content = f.read()
            except Exception as e:
                return f"Error reading file: {e}"
        return DocumentProcessor.yaml_to_markdown(file_content)

    @staticmethod
    def select_im(images, annotated_images, page_id):
        if len(annotated_images) != len(images):
            return gr.update(value=images[page_id-1])   
        else:
            return gr.update(value=annotated_images[page_id-1])

    @staticmethod
    def upload_pdf_new(pdf_path, idx):
        images = []
        doc = fitz.open(pdf_path)
        for page in doc:
            rect = page.search_for(" ")
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
            pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(np.asarray(pil_image))
        return images, gr.update(visible=False), gr.update(visible=True), gr.update(value=images[idx-1]), gr.update(maximum=len(images))

    @staticmethod
    def markdown_to_dict(markdown_content):
        try:
            yaml_content = yaml.safe_load(markdown_content)
            print(f"YAML content: {yaml_content}")  # Debug print statement
            return yaml_content
        except yaml.YAMLError as e:
            print(f"Error parsing Markdown content: {e}")
            return None

    @staticmethod
    def analyze(pdf_path, use_default, file, page_id):
        if use_default:
            pipeline_config_path = os.path.abspath(os.path.join(CONFIGS_PATH, 'good_pipeline.yaml'))
            try:
                with open(pipeline_config_path, 'r') as f:
                    loaded_yaml = f.read()
            except Exception as e:
                return f"Error reading default pipeline file: {e}", None, None, None, None, None
        else:
            if file is None:
                return "No file uploaded.", None, None, None, None, None
            try:
                with open(file.name, 'r') as f:
                    loaded_yaml = f.read()
            except Exception as e:
                return f"Error reading file: {e}", None, None, None, None, None

        try:
            pipeline_config = yaml.safe_load(loaded_yaml)
        except yaml.YAMLError as e:
            return f"Error parsing YAML content: {e}", None, None, None, None, None

        if not isinstance(pipeline_config, dict) or 'COMPONENTS' not in pipeline_config:
            return "Invalid pipeline configuration", None, None, None, None, None

        pipeline = Pipeline()
        for comp in pipeline_config['COMPONENTS']:
            comp_class_name = comp['CLASS']
            comp_kwargs = comp['KWARGS']
            comp_class = globals().get(comp_class_name)
            if comp_class is not None:
                pipeline.add(comp_class(**comp_kwargs))
            else:
                return f"Component class {comp_class_name} not found", None, None, None, None, None

        pipeline.build()
        dps, page_dicts = pipeline.run(pdf_path)

        all_category_names = []
        dd_images = []
        dd_annotations = []
        for dp in dps:
            category_names_list = []
            bboxes = []
            anns = dp.get_annotation()
            for ann in anns:
                bboxes.append([int(cord) for cord in ann.bbox])
                category_names_list.append(ann.category_name.value)
            annotations = list(zip(bboxes, category_names_list))
            dd_images.append(dp.image_orig._image)
            dd_annotations.append(annotations)
            all_category_names += category_names_list

        return (dd_images, dd_images, dd_images[page_id-1], dd_annotations,
                gr.update(choices=np.unique(all_category_names).tolist()), 
                gr.update(visible=True), dps)

    @staticmethod
    def draw_bboxes_on_im(images, rel_labels, page_id, all_annotations):
        print('draw rects: ', rel_labels)
        annotated_images = []
        color_map = {
            'table': (0, 255, 0, 255), # Green
            'text': (255, 0, 0, 255), # Red
            'image': (0, 0, 255, 255), # Blue
            'figure': (255, 255, 0, 255), # Yellow
            'word': (0, 255, 255, 255), # Cyan
            'title': (255, 0, 255, 255), # Magenta
            'list': (255, 165, 0, 255), # Orange
        }
        for image, annotations in zip(images, all_annotations):
            pil_image = Image.fromarray(image)
            im_draw= ImageDraw.Draw(pil_image, mode='RGBA')
            for (bbox, label) in annotations:
                if label in rel_labels:
                    fill_c = list(color_map[label])
                    fill_c = [int(c*255) for c in fill_c]
                    fill_c[-1] = 80
                    outline_c = [int(c*255) for c in color_map[label]]
                    im_draw.rectangle(bbox, outline=tuple(outline_c), fill=tuple(fill_c), width=4)
            annotated_images.append(np.asarray(pil_image))
        return annotated_images, gr.update(value=annotated_images[page_id-1])

    @staticmethod
    def transform_structure(method, selected_elements, structured_format, pdf_path, dps):
        if method == "PDF_Plumber":
            table_method = 'pdfplumber'
        elif method == "LLMs":
            table_method = 'llm'
        elif method == "TATR":
            table_method = 'tatr'

        annotations_to_merge = [dd.LayoutType[element] for element in selected_elements]
        doc_transformer = DocumentTransformer(pdf_path, table_extraction_method=table_method)
        doc_transformer.merge_with_annotations(dps, annotations_to_merge)
        doc_transformer.docorate_unmatched_textblocks()

        if "Markdown" in structured_format:
            markdown_str = doc_transformer.transform_to(Format.MARKDOWN.value)
            return markdown_str, markdown_str
        return "No structured format selected.", ""

    @staticmethod
    def extract_parameters(json_file, markdown_str):
        if json_file is None:
            return "No JSON schema uploaded.", None

        if not markdown_str:
            return "No Markdown content available for extraction.", None

        try:
            with open(json_file.name, 'r') as f:
                parameter_schema = json.load(f)
        except Exception as e:
            return f"Error reading JSON schema: {e}", None

        format_handler = MarkdownHandler(markdown_str)
        json_extractor = JsonExtractor(intermediate_format=format_handler, chunk_overlap=0, chunks_max_characters=100000, model='gpt-4o')

        try:
            res = json_extractor.populate_schema(json_schema_string=json.dumps(parameter_schema))
            json_result_str = json.dumps(res, indent=2)  # For displaying in JSON format
            
            # Validate JSON
            try:
                json.loads(json_result_str)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON generated: {e}")
                return f"Invalid JSON generated: {e}", None

            output_file = 'extracted_parameters.json'
            with open(output_file, 'w') as f:
                f.write(json_result_str)

            return res, output_file
        except Exception as e:
            return f"Error extracting parameters: {e}", None
