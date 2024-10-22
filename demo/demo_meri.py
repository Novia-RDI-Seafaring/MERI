import os
import sys
import json
base_path = os.path.abspath(os.path.join(os.getcwd(), '../MERI'))
sys.path.append(base_path)

from PIL import Image, ImageDraw
import gradio as gr
from meri.utils.document_processing import DocumentProcessor


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

with gr.Blocks(title="Information Extraction from Document", css=custom_css) as demo:
    images = gr.State([])
    annotations = gr.State([])
    annotated_images = gr.State([])
    dps = gr.State([])  # State to store dps
    markdown_str = gr.State("")  # State to store the generated markdown string
    yaml_file = gr.State(None)  # State to store the uploaded YAML file
    json_file = gr.State(None)  # State to store the uploaded JSON file
    default_pipeline_state = gr.State(False)  # State to store the default_pipeline checkbox state
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
                            loaded_yaml_markdown = gr.Markdown(label='Loaded Configuration YAML', elem_classes="scrollable-markdown")
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
                        with gr.Row(elem_classes="intermediary_structure"):
                            with gr.Column():
                                displayMode = gr.Radio(["PDF_Plumber", "LLMs", "TATR"], info='Select the method you want', label='Method to run')
                            with gr.Column():
                                intermedia_comps = gr.CheckboxGroup(["Markdown"], info='Choose the intermediate format you want', label='Structured Format')
                        with gr.Row():
                            show_markdown_btn = gr.Button("Show content", variant="primary")
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
                        # default_schema = gr.Checkbox(label='Use Default Schema')
                    with gr.Row():
                        pipeline_comps2 = gr.File(label='Configuration YAML')
                        json_input2 = gr.File(label="JSON Configuration")
                    with gr.Column():
                        displayMode2 = gr.Radio(["PDF_Plumber", "LLMs", "TATR"], info='Select the method you want', label='Method to run')
                    with gr.Column():
                        intermedia_comps2 = gr.CheckboxGroup(["Markdown"], info='Choose the intermediate format you want', label='Structured Format')
            with gr.Row():
                run_pipeline_btn = gr.Button("Run", variant="primary")
            with gr.Row():
                entire_pipeline_res_markdown = gr.JSON(label="Pipeline JSON Parameter Result", visible=True, elem_classes="scrollable-jsons")

            
    """
    Upload PDF and Display Images and Annotated Images
    """
    # Upload the PDF and display the images
    page_slider.change(processor.select_im, inputs=[images, annotated_images, page_slider], outputs=[anIm])
    pdf.upload(processor.upload_pdf_new, inputs=[pdf, page_slider], outputs=[images, pdfUpload, annotated_image_row, anIm, page_slider])
    
    """
    Choose the components for the pipeline cinfiguration or use the default pipeline
    """
    # Pipeline components to detect the layout elements
    def on_pipeline_comps_change(use_default, file):
        return processor.display_yaml_file(use_default, file)
    
    def update_yaml_file(file):
        return file, file
    
    default_pipeline.change(fn=on_pipeline_comps_change, inputs=[default_pipeline, pipeline_comps], outputs=loaded_yaml_markdown)
    pipeline_comps.change(fn=on_pipeline_comps_change, inputs=[default_pipeline, pipeline_comps], outputs=loaded_yaml_markdown)

    """
    Layout Analysis Pipeline running and displaying the detected elements
    """
    def analyze(pdf, use_default, file, page_id):
        return processor.analyze(pdf.name, use_default, file, page_id)

    pipeline_btn.click(fn=analyze, inputs=[pdf, default_pipeline, pipeline_comps, page_slider], 
                       outputs=[annotations, labelsOfInterest, detectionResRow, dps])
    displayLabels.click(fn=processor.draw_bboxes_on_im, inputs=[images, labelsOfInterest, page_slider, annotations], outputs=[annotated_images, anIm])
    
    
    """
    Extracting the Intermediary Structure and Displaying the Markdown
    """
    # Structured Format Transformation
    def transform_structure_interface(method, selected_elements, structured_format, pdf, dps):
        return processor.transform_structure(method, selected_elements, structured_format, pdf.name, dps)

    show_markdown_btn.click(fn=transform_structure_interface, 
                            inputs=[displayMode, labelsOfInterest, intermedia_comps, pdf, dps], 
                            outputs=[markdown_result, markdown_str])

    """
    Parameter Extraction from the Document
    """
    # Parameter Extraction
    def extract_parameters_interface(json_file, markdown_str):
        return processor.extract_parameters(json_file, markdown_str)

    # Display the JSON schema content
    json_input.upload(processor.display_json_schema, inputs=json_input, outputs=json_schema)
    extract_btn.click(fn=extract_parameters_interface, inputs=[json_input, markdown_str], outputs=[json_result, download_btn,  res])
    
    
    """ ############################# Entire Pipeline Execution ################################## """
    """
    Entire Pipeline Execution
    """
    def on_run_pipeline_click(use_default, pipeline_file, json_file, pdf_file, method, structured_format, page_slider):
        json_result, res = DocumentProcessor.run_pipeline(use_default, pipeline_file, json_file, pdf_file, method, structured_format, page_slider)


        #json_result, res, _ = DocumentProcessor.run_pipeline(use_default, pipeline_file, json_file, pdf_file, method, structured_format, page_slider)
        return json_result, res
    json_input2.upload(processor.display_json_schema, inputs=json_input2, outputs=json_schema)
    run_pipeline_btn.click(
        on_run_pipeline_click,
        inputs=[default_pipeline2, pipeline_comps2, json_input2, pdf, displayMode2, intermedia_comps2, page_slider],
        outputs=[entire_pipeline_res_markdown, res]
    )
    
    """
    Highlight Extracted DATA with Bounding Boxes
    """
    # Function to highlight extracted text on PDF
    def on_highlight_text_click(res, pdf_images, page_slider):
        highlighted_images = processor.highlight_extracted_text_on_pdf(pdf_images, res, page_slider)
        return gr.update(value=highlighted_images)

    # UI Element to trigger highlight
    highlight_btn = gr.Button("Highlight Extracted Text")

    # Update the click event for highlighting,
    highlight_btn.click(
        fn= processor.highlight_extracted_text_on_pdf, #on_highlight_text_click,
        inputs=[images, res, page_slider], 
        outputs=[annotated_images, anIm]
    )

    
demo.launch()
