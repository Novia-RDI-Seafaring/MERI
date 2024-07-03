import os
import sys
# sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir, "meri")))
base_path = os.path.abspath(os.path.join(os.getcwd(), '../../MERI'))
sys.path.append(base_path)
print(base_path)

import gradio as gr
from PIL import Image

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


# import deepdoctection as dd
# from meri.layout.pipeline_components import (AddPDFInfoComponent, 
#                         DummyDetectorComponent, 
#                         LayoutDetectorComponent,
#                         OCRComponent,
#                         ImageDetectorComponent,
#                         DrawingsDetectorComponent,
#                         TablePlumberComponent,
#                         TableDetectorComponent)
# from meri.layout.pipeline import Pipeline

from matplotlib import pyplot as plt 
import matplotlib
import numpy as np
from PIL import Image, ImageDraw
import fitz

cmap = matplotlib.colormaps['nipy_spectral']


component_order_mapping = {
    AddPDFInfoComponent.__name__: 0,
    LayoutDetectorComponent.__name__: 1,
    DrawingsDetectorComponent.__name__: 1,
    ImageDetectorComponent.__name__: 1,
    TableDetectorComponent.__name__: 1,
    TablePlumberComponent.__name__: 1,
    OCRComponent.__name__: 2,
    dd.TextOrderService.__name__: 3
}

color_map = {}
n_labels = len(list(dd.LayoutType))
for i, x in enumerate(list(dd.LayoutType)):
    color_map[x.value] = cmap(i/n_labels)


pipeline_components_map = {
    'AddInfo': {
        'display_name': 'Add pdf info',
        'info': 'Extract info from nativ pdf structure',
        'class': AddPDFInfoComponent,
        'kwargs': {}
    },
    'ImageDetector': {
        'display_name': 'Fitz Image Detector',
        'info': 'Finds images based on info available in pdf',
        'class': ImageDetectorComponent,
        'kwargs': {}
    },
    'DrawingDetector': {
        'display_name': 'Fitz Drawing Detector',
        'info': 'Finds vector drawings based on info available in pdf',
        'class': DrawingsDetectorComponent,
        'kwargs': {}
    },
    'DLObjectDetector':{
        'display_name': 'DL Object Detector',
        'info': 'Applies object detection model to detect layout components like tables, figures, etc.',
        'class': LayoutDetectorComponent,
        'kwargs': {'cfg_path': 'layout_detector_config.yaml',
                   'method':'d2layout'}
    },
    'DLTableDetector':{
        'display_name': 'DL Table Detector',
        'info': 'Applies object detection model to detect tables.',
        'class': LayoutDetectorComponent,
        'kwargs': {'cfg_path': 'table_detector_config.yaml',
                   'method':'detr'}
    },
    'DocTrWordDetector':{
        'display_name': 'DocTr Word Detector',
        'info': 'Detect words with DocTr.',
        'class': LayoutDetectorComponent,
        'kwargs': {'cfg_path': 'doctr_config.yaml',
                   'method':'doctr_textdetector'}
    },
    'DocTrOCR':{
        'display_name': 'DocTr OCR',
        'info': 'Required DocTr word detector. Extract text from detected words.',
        'class': OCRComponent,
        'kwargs': {'cfg_path': 'doctr_config.yaml',
                   'method':'doctr'}
    },
    'TesseractOCR':{
        'display_name': 'Tesseract OCR',
        'info': 'Applies Tesseract OCR.',
        'class': OCRComponent,
        'kwargs': {'cfg_path': 'tesseract_config.yaml',
                   'method':'tesseract'}
    },
    'TextOrdering':{
        'display_name': 'Text Ordering',
        'info': 'Oders the extracted text. Requires OCR.',
        'class': dd.TextOrderService,
        'kwargs': {'text_container': dd.LayoutType.word},
    },
    'TablePlumber': {
        'display_name': 'Table plumber detecor',
        'info': 'Detect tables using pdfplumber library',
        'class': TablePlumberComponent,
        'kwargs': {}
    }
}


intermediary_structure_transformation_pipeline = [
    'Markdown',
    'LLMSTableDetector',
    'PDFPlumberTableDetector',
]


def show_markdown_result():
    # Simulate processing and showing Markdown result
    return "# This is the Markdown result"

def show_json_result(json_file):
    # Simulate processing and showing JSON result
    # In a real scenario, you would parse and process the JSON file here
    if json_file is not None:
        return {"key": "value", "another_key": "another_value"}
    return None

def select_im(images, annotated_images, page_id):
    if len(annotated_images) != len(images):
        return gr.update(value=images[page_id-1])   
    else:
        return gr.update(value=annotated_images[page_id-1])

def upload_pdf_new(pdf_path, idx):

    images = []
    doc = fitz.open(pdf_path)
    for page in doc:
        rect = page.search_for(" ")
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
        pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(np.asarray(pil_image))

    return images, gr.update(visible=False), gr.update(visible=True), gr.update(value=images[idx-1]), gr.update(maximum=len(images))


def compOrder(comp_key) -> int:
    return component_order_mapping[pipeline_components_map[comp_key]['class'].__name__]

def analyze(pdf_path, pipeline_comps, page_id):

    pipeline_comps.sort(key=compOrder)
    pipeline = Pipeline()
    
    for comp in pipeline_comps:
        comp_settings = pipeline_components_map[comp]
        comp_class = comp_settings['class']
        comp_kwargs = comp_settings['kwargs']
        pipeline.add(comp_class(**comp_kwargs))

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

    return dd_images, dd_images, dd_images[page_id-1], dd_annotations, gr.update(choices=np.unique(all_category_names).tolist()), gr.update(visible=True)

def draw_bboxes_on_im(images, rel_labels, page_id, all_annotations):

    print('draw rects: ', rel_labels)
    annotated_images = []
    for image, annotations in zip(images, all_annotations):
        pil_image = Image.fromarray(image)
        im_draw= ImageDraw.Draw(pil_image, mode='RGBA')

        for (bbox, label) in annotations:
            if label in rel_labels:
                fill_c = list(color_map[label])
                fill_c = [int(c*255) for c in fill_c]
                fill_c[-1] = 80

                outline_c = [int(c*255) for c in color_map[label]]
                im_draw.rectangle(bbox, outline=tuple(outline_c), fill=tuple(fill_c), width=4)#color_map[label]
        annotated_images.append(np.asarray(pil_image))
        
    return annotated_images, gr.update(value=annotated_images[page_id-1])


with gr.Blocks(title='Document Layout Analysis') as demo:

    def pdf_coords_to_img_coords(pdf_coords, pdf_height, pdf_width, im_width, im_height):
        '''PDF bbox coords as (x0, y0, x1, y1)
        to img coords in pixel coords (x0,y0, x1,y1)'''

        x0, y0, x1, y1 = pdf_coords

        x0_rel = x0/pdf_width
        x1_rel = x1/pdf_width

        y0_rel = y0/pdf_height
        y1_rel = y1/pdf_height

        rect_shape = [int(x0_rel*im_width+0.5),int(y0_rel*im_height+0.5), int(x1_rel*im_width+0.5), int(y1_rel*im_height+0.5)]

        return rect_shape
    
    annotations = gr.State([]) # one list for each page in pdf
    images=gr.State([]) # one img for each page in pdf
    annotated_images = gr.State([])

    with gr.Row():
        gr.Markdown("<h1><center>Document Layout Analysis Demo</center></h1>")


    with gr.Row():
        with gr.Column(elem_id='imageColumn'):
            with gr.Column(visible=True) as pdfUpload:
                pdf = gr.File(label="PDF")
                #upload_btn = gr.Button("Load Default PDF", variant="primary")

            with gr.Row(visible=False, elem_classes='image_holder') as original_image_row:
                imm = gr.Image(Image.new('RGB', (1, 1)), show_label=False, container=False, elem_classes="originalImage")
            with gr.Column(visible=False, elem_classes='image_holder') as annotated_image_row:
                page_slider = gr.Slider(1, 1, value=1, step=1, interactive=True)
                anIm = gr.Image(show_label=False, container=False)    
        
        with gr.Column(elem_id='ControlColumn'):                   
            with gr.Accordion(label="Layout Analysis Pipeline", open=False):
                with gr.Column(elem_id='ControlColumn'):
                    with gr.Row():
                        gr.Markdown(
                            """
                            # Configure Analysis Pipeline
                            The document layout is analyized through a number of components that are being
                            executed in sequential order. Select the components the pipeline should contain.
                            
                            """
                        )

                    with gr.Row():
                        pipeline_comps = gr.CheckboxGroup([key for (key, comp) in pipeline_components_map.items()],
                                                            info=' ', label='Pipeline Components')
                    with gr.Row():
                        pipeline_btn = gr.Button("Run Pipeline", variant="primary")

                    with gr.Row(visible=False) as detectionResRow:
                        labelsOfInterest = gr.CheckboxGroup([], every=0.5,
                                                            info='Select elements that should be displaid', label='Detected Layout Elements')
                        displayLabels = gr.Button("Show Elements", variant="primary")
                        
            with gr.Accordion(label="Intermediary Structure Transformation", open=False):
                with gr.Column(elem_id='ControlColumn'):
                    with gr.Row():
                        gr.Markdown(
                            """
                            # Intermediary Structure Transformation
                            This stage involves transforming the document's structure into intermediary formats, 
                            such as Markdown. These transformations help in structuring the document layout in 
                            a more readable and organized manner, facilitating further analysis and processing.

                            """
                        )

                    with gr.Row():
                        pipeline_comps = gr.CheckboxGroup(intermediary_structure_transformation_pipeline,
                                                            info=' ', label='Transformation Components')
                    with gr.Row():
                        pipeline_btn = gr.Button("Run Pipeline", variant="primary")

                    with gr.Row(visible=False) as detectionResRow:
                        labelsOfInterest = gr.CheckboxGroup([], every=0.5,
                                                            info='Select elements that should be displaid', label='Detected Layout Elements')
                        displayLabels = gr.Button("Show Elements", variant="primary")
            
            with gr.Accordion(label="Parameter Extraction", open=False):
                with gr.Column():
                    with gr.Row():
                        gr.Markdown(
                            """
                            # Configure Parameter Extraction
                            The parameters are extracted from the document based on the provided JSON configuration.
                            """
                        )

                    with gr.Row():
                        json_input = gr.File(label="JSON Configuration")
                    with gr.Row():
                        extract_btn = gr.Button("Extract Parameters", variant="primary")

                    with gr.Row(visible=False) as extractionResRow:
                        extraction_labels = gr.CheckboxGroup([], every=0.5,
                                                             info='Select parameters to display', label='Extracted Parameters')
                        displayExtraction = gr.Button("Show Parameters", variant="primary")
                        
            # Adding the Markdown result display
            with gr.Column():
                with gr.Row():
                    markdown_result = gr.Markdown(label="Markdown Result", visible=False)

            # Adding the JSON result display
            with gr.Column():
                with gr.Row():
                    json_result = gr.JSON(label="JSON Parameter Result", visible=False)



        page_slider.change(select_im, inputs=[images, annotated_images, page_slider], outputs=[anIm] )

        pdf.upload(upload_pdf_new, inputs=[pdf, page_slider], outputs=[images, pdfUpload, annotated_image_row, anIm, page_slider])
        #upload_btn.click(fn=upload_pdf, inputs=[],
        #      outputs=[imm, anIm, annotated_image_row, upload_button])

        pipeline_btn.click(fn=analyze, inputs=[pdf, pipeline_comps, page_slider], 
                           outputs=[annotated_images, images, anIm, annotations, labelsOfInterest, detectionResRow])
        
        displayLabels.click(fn=draw_bboxes_on_im, inputs=[images, labelsOfInterest, page_slider, annotations],
                            outputs=[annotated_images, anIm])

if __name__ == '__main__':
    demo.launch()