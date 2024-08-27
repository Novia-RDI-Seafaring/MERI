import os
import sys
import yaml
import numpy as np
import json
base_path = os.path.abspath(os.path.join(os.getcwd(), '../MERI'))
sys.path.append(base_path)
import gradio as gr
from PIL import Image
from meri.utils.document_processing import DocumentProcessor
from meri.meri import MERI  # Import the MERI class


processor = DocumentProcessor()

# Custom CSS
custom_css = """
.scrollable-markdown {
    max-height: 400px;
    overflow-y: scroll;
    padding: 10px;
    border: 1px solid #ccc;
}
.scrollable-json .json-holder {
    max-height: 350px;
    overflow-y: scroll;
    overflow-x: scroll;
}
.scrollable-jsons .json-holder {
    max-height: 400px;
    overflow-y: scroll;
    overflow-x: scroll;
}
"""


# Define the default configuration file path (if you have one)
DEFAULT_CONFIG_PATH = "/workspaces/MERI/meri/configs/meri_default.yaml" # "/workspaces/MERI/meri/layout/config/good_pipeline.yaml"

def layout_analysis_interface(pdf, config_yaml_path, use_default):
    
    config_path = DEFAULT_CONFIG_PATH if use_default else config_yaml_path.name
    meri_instance = MERI(pdf_path=pdf.name, config_yaml_path=config_path)
    dps, page_dicts = meri_instance.layout_analysis()
    
    all_category_names = []
    dd_annotations = []
    for dp in dps:
        
        category_names_list = []
        bboxes = []
        anns = dp.get_annotation()
        for ann in anns:
            bboxes.append([int(cord) for cord in ann.bbox])
            category_names_list.append(ann.category_name.value)
        annotations = (list(zip(bboxes, category_names_list)), dp.image_orig._image.shape)
        dd_annotations.append(annotations)
        all_category_names += category_names_list
        
    return dps, dd_annotations, gr.update(choices=np.unique(all_category_names).tolist()), gr.update(visible=True)

def transform_to_intermediate_interface(pdf, config_yaml_path, dps, use_default):
    config_path = DEFAULT_CONFIG_PATH if use_default else config_yaml_path.name
    meri_instance = MERI(pdf_path=pdf.name, config_yaml_path=config_path)
    intermediate_format = meri_instance.transform_to_intermediate(dps)
    return intermediate_format, intermediate_format  # Return twice to store the markdown in state

def populate_schema_interface(pdf, config_yaml_path, json_schema_path, intermediate_format, use_default):
    config_path = DEFAULT_CONFIG_PATH if use_default else config_yaml_path.name
    meri_instance = MERI(pdf_path=pdf.name, config_yaml_path=config_path)
    
    with open(json_schema_path, 'r') as f:
        json_schema = json.load(f)
    json_schema_string = json.dumps(json_schema)  # Ensure json_schema is a string
    result = meri_instance.populate_schema(json_schema_string, intermediate_format)
    json_result_str = json.dumps(result, indent=2)
    
    json_result = json.loads(json_result_str)
   
    output_file = 'extracted_parameters.json'
    with open(output_file, 'w') as f:
        f.write(json_result_str)  # Write the string version, not the dict

    return json_result, output_file, result

def run_entire_pipeline(pdf, config_yaml_path, json_schema_path, use_default):
    config_path = DEFAULT_CONFIG_PATH if use_default else config_yaml_path.name
    meri_instance = MERI(pdf_path=pdf.name, config_yaml_path=config_path)
    with open(json_schema_path, 'r') as f:
        json_schema = json.load(f)
    json_schema_string = json.dumps(json_schema)
    result = meri_instance.run(json_schema_string)
    return result, result

with gr.Blocks(title="Information Extraction from Document", css=custom_css) as demo:
    images = gr.State([])
    annotations = gr.State([])
    annotated_images = gr.State([])
    dps = gr.State([])  # State to store dps
    intermediate_format = gr.State("")  # State to store the generated intermediate format string
    yaml_file = gr.State(None)  # State to store the uploaded YAML file
    json_file = gr.State(None)  # State to store the uploaded JSON file
    res = gr.State(None)

    with gr.Row():
        gr.Markdown("<h1><center>Document MERI Demo</center></h1>")

    with gr.Row():
        with gr.Column(elem_id='imageColumn'):
            with gr.Column(visible=True) as pdfUpload:
                pdf = gr.File(label="PDF")
            
            with gr.Row(visible=False, elem_classes='image_holder') as original_image_row:
                imm = gr.Image(Image.new('RGB', (1, 1)), show_label=False, container=False, elem_classes="originalImage")
            with gr.Column(visible=False, elem_classes='image_holder') as annotated_image_row:
                page_slider = gr.Slider(1, 1, value=1, step=1, interactive=True)
                anIm = gr.Image(show_label=False, container=False)
                
        with gr.Tab("Run individual components"):
            with gr.Column(elem_id='ControlColumn'):
                with gr.Accordion(label="Layout Analysis Pipeline", open=False):
                    with gr.Column(elem_id='ControlColumn'):
                        with gr.Row():
                            gr.Markdown("""
                                # Configure Analysis Pipeline
                                The document layout is analyzed select a number of components that are being
                                executed in sequential order by default. Or upload the config YAML file for the components the pipeline.
                                """)
                        with gr.Row():
                            default_pipeline = gr.Checkbox(label='Use Default Pipeline')
                        with gr.Row():
                            pipeline_comps = gr.File(label='Configuration YAML')
                        with gr.Row():
                            pipeline_btn = gr.Button("Run Pipeline", variant="primary")
                        with gr.Row(visible=False) as detectionResRow:
                            labelsOfInterest = gr.CheckboxGroup([], every=0.5, info='Select elements that should be displayed', label='Detected Layout Elements')
                            displayLabels = gr.Button("Show Elements", variant="primary")

                with gr.Accordion(label="Structured Format", open=False):
                    with gr.Column(elem_id='ControlColumn'):
                        with gr.Row():
                            gr.Markdown("""
                                # Intermediary Structure Transformation
                                This is transforming the document's structure into intermediary formats, 
                                such as Markdown.
                                """)
                        with gr.Row():
                            transform_btn = gr.Button("Transform to Intermediate", variant="primary")
                        with gr.Row():
                            markdown_result = gr.Markdown(label="Markdown Result", visible=True, elem_classes="scrollable-markdown")
                
                with gr.Accordion(label="Parameter Extraction", open=False):
                    with gr.Column():
                        with gr.Row():
                            gr.Markdown("""
                                # Configure Parameter Extraction
                                The parameters are extracted from the document based on the provided JSON configuration.
                                """)

                        with gr.Row():
                            json_input = gr.File(label="JSON Configuration")
                            json_schema = gr.JSON(label="JSON Schema Content", visible=True, elem_classes="scrollable-json")
                        with gr.Row():
                            extract_btn = gr.Button("Extract Parameters", variant="primary")
                        with gr.Row():
                            json_result = gr.JSON(label="JSON Parameter Result", visible=True, elem_classes="scrollable-jsons")
                        with gr.Row():
                            download_btn = gr.File(label="Download JSON")
                            
        with gr.Tab("Run entire pipeline"):
            with gr.Accordion(label="Run entire pipeline Extraction", open=False):
                with gr.Column(elem_id='ControlColumn'):
                    with gr.Row():
                        gr.Markdown("""
                            # Run Entire Pipeline
                            This is a one-click button to run the entire pipeline and component.
                            """)
                    with gr.Row():
                        default_pipeline2 = gr.Checkbox(label='Default Pipeline')
                    with gr.Row():
                        json_input2 = gr.File(label="JSON Configuration")
                    with gr.Row():
                        run_pipeline_btn = gr.Button("Run", variant="primary")
                    with gr.Row():
                        entire_pipeline_res = gr.JSON(label="Pipeline JSON Parameter Result", visible=True, elem_classes="scrollable-jsons")

    # Upload the PDF and display the images
    page_slider.change(processor.select_im, 
                       inputs=[images, annotated_images, page_slider], 
                       outputs=[anIm])
    pdf.upload(processor.upload_pdf_new, 
               inputs=[pdf, page_slider], 
               outputs=[images, pdfUpload, annotated_image_row, anIm, page_slider])
    
    # Assigning Gradio actions to each component using the MERI class and handling default configuration
    pipeline_btn.click(fn = layout_analysis_interface, 
               inputs=[pdf, pipeline_comps, default_pipeline], 
               outputs=[dps, annotations, labelsOfInterest, detectionResRow])         
    
    displayLabels.click(fn=processor.draw_bboxes_on_im, 
                        inputs=[images, labelsOfInterest, page_slider, annotations], 
                        outputs=[annotated_images, anIm])
     
    transform_btn.click(fn = transform_to_intermediate_interface, 
                        inputs=[pdf, pipeline_comps, dps, default_pipeline], 
                        outputs=[markdown_result, intermediate_format])

    
    json_input.upload(processor.display_json_schema, 
                      inputs=json_input, 
                      outputs=json_schema)
    
    extract_btn.click(fn = populate_schema_interface, 
                      inputs=[pdf, pipeline_comps, json_input, intermediate_format, default_pipeline], 
                      outputs=[json_result, download_btn, res])

    run_pipeline_btn.click(fn = run_entire_pipeline, 
                           inputs=[pdf, pipeline_comps, json_input2, default_pipeline2], 
                           outputs=[entire_pipeline_res, res])
    
    """
        Highlight Extracted DATA with Bounding Boxes
    """
    # UI Element to trigger highlight
    highlight_btn = gr.Button("Highlight Extracted Text")

    # Update the click event for highlighting,
    highlight_btn.click(
        fn= processor.highlight_extracted_text_on_pdf, #on_highlight_text_click,
        inputs=[images, res, page_slider], 
        outputs=[annotated_images, anIm]
    )

demo.launch()