import torch
from torchvision import transforms
from transformers import AutoModelForObjectDetection
from PIL import Image, ImageDraw
from enum import Enum
from typing import List
import deepdoctection as dd
from .utils import scale_coords
from .tsr import table_transformer_utils as tsr_utils


#
# see https://huggingface.co/spaces/nielsr/tatr-demo/blob/main/app.py
#

class TableStructureLabels(Enum):

    TABLE = 'table'
    TABLE_ROW = 'table row'
    TABLE_COLUMN = 'table column'
    TABLE_SPANNING_CELL = 'table spanning cell'
    TABLE_COLUMN_HEADER = 'table column header'
    TABLE_PROJECTED_ROW_HEADER = 'table projected row header'
    


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
        
        return resized_image



# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    width, height = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
    return boxes


def outputs_to_objects(outputs, img_size, id2label, thr=0.5, filter_labels: List[TableStructureLabels]=[]):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object' and class_label not in [lab.value for lab in filter_labels]:
            if float(score)>=thr:
                objects.append({'label': class_label, 'score': float(score),
                                'bbox': [float(elem) for elem in bbox]})

    return objects

class TableStructureRecognizer:

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.structure_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all").to(self.device)
        self.structure_transform = transforms.Compose([
            MaxResize(1000),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def predict(self, image, thr=0.5, filter_labels: List[TableStructureLabels] = [], outline_color='red'):

        # prepare image for the model
        # pixel_values = structure_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = self.structure_transform(image).unsqueeze(0).to(self.device)

        # forward pass
        with torch.no_grad():
            outputs = self.structure_model(pixel_values)

        # postprocess to get individual elements
        id2label = self.structure_model.config.id2label
        id2label[len(self.structure_model.config.id2label)] = "no object"
        cells = outputs_to_objects(outputs, image.size, id2label, thr=thr, filter_labels=filter_labels)

        # visualize cells on cropped table
        draw = ImageDraw.Draw(image)

        for cell in cells:
            if cell['label'] == 'row':
                draw.rectangle(cell["bbox"], outline='green')
            else:
                draw.rectangle(cell["bbox"], outline=outline_color)
            
        return cells, image
    
class CellDetector:
    def __init__(self, weights='cell/d2_model_1849999_cell_inf_only.pt') -> None:
        self.config_path = dd.ModelCatalog.get_full_path_configs(weights)
        self.weights_path = dd.ModelDownloadManager.maybe_download_weights_and_configs(weights)
        self.profile = dd.ModelCatalog.get_profile(weights)
        self.categories = self.profile.categories

        self.cell_detector = dd.D2FrcnnDetector(
                    self.config_path, self.weights_path, self.categories, 
                    device='cuda', filter_categories=[]
                )

    def detect(self, np_img, threshold = 0.9):

        res = self.cell_detector.predict(np_img)
        
        res_arr = []
        for r in res:
            if r.score > threshold:
                res_arr.append({
                    'bbox': r.box,
                    'score': r.score,
                    'label': r.class_name.value
                })

        image = Image.fromarray(np_img)
        # visualize cells on cropped table
        draw = ImageDraw.Draw(image)

        for cell in res_arr:
            draw.rectangle(cell['bbox'], outline='red')
        return res_arr, image
    

class TSRBasedTableExtractor:

    def __init__(self, cell_thr=0.9, tsr_thr=0.9) -> None:
        
        self.cell_detector = CellDetector()
        self.cell_thr = cell_thr


        self.tsr_detector = TableStructureRecognizer()
        self.tsr_thr = tsr_thr
        # only intrested in rows and columns
        self.tsr_filter_labels = [TableStructureLabels.TABLE_COLUMN_HEADER, TableStructureLabels.TABLE_PROJECTED_ROW_HEADER,
                 TableStructureLabels.TABLE_SPANNING_CELL, TableStructureLabels.TABLE]

    def cell_based_extract(self, table_np_im, pdf_full_fitz_page, table_bbox):

        # detect cells
        table_cells, cells_im = self.cell_detector.detect(table_np_im, threshold=self.cell_thr)

        # get text for each cell from pdf
        source_height, source_width = table_np_im.shape[:2]
        target_height = pdf_full_fitz_page.get_textpage(clip=table_bbox).rect.height
        target_width = pdf_full_fitz_page.get_textpage(clip=table_bbox).rect.width

        margin_l, margin_top, x1, y1 = table_bbox
        margin_r = target_width - x1
        margin_bottom = target_height - y1

        for cell in table_cells:
            pdf_coords = scale_coords(cell['bbox'], source_height, source_width, target_height-margin_top-margin_bottom, target_width-margin_l-margin_r)
            pdf_coords_adj = [pdf_coords[0]+margin_l, pdf_coords[1]+margin_top,pdf_coords[2]+margin_l, pdf_coords[3]+margin_top ]
            words = pdf_full_fitz_page.get_textpage(clip=pdf_coords_adj).extractWORDS()
            cell['text'] = ' '.join([w[4]for w in words])

        # in image coordinates
        cell_tokens = [{'text': c['text'], 'bbox': c['bbox']} for c in table_cells]

        out_formats, tsr_im = self.extract(table_np_im, tokens=cell_tokens)
        return out_formats, tsr_im, cells_im

    def extract(self, table_np_im, tokens=None):
        """_summary_

        Args:
            table_np_im (_type_): numpy array representing table
            token (_type_): [{'text': <text>, 'bbox': [x0,y0,x1,y1]}] representing text in the table.
        """

        # detector rows and cols
        table_structure, tsr_im = self.tsr_detector.predict(Image.fromarray(table_np_im), thr=self.tsr_thr, filter_labels=self.tsr_filter_labels)

        
        if tokens is None:
            # could do OCR
            raise NotImplementedError
        

        # output to objects
        out_formats = {}
        
        out_formats['objects'] = table_structure        
        out_formats['objects'].append({
                                    'label': 'table',
                                    'score': 1,
                                    'bbox': [0, 0, table_np_im.shape[1], table_np_im.shape[1]]
                                        })
        
        for idx, token in enumerate(tokens):
            if 'span_num' not in token:
                token['span_num'] = idx
            if 'line_num' not in token:
                token['line_num'] = 0
            if 'block_num' not in token:
                token['block_num'] = 0

        # Further process the detected objects so they correspond to a consistent table
        tables_structure = tsr_utils.objects_to_structures(out_formats['objects'], tokens, {
            'table row': self.tsr_thr,
            'table column': self.tsr_thr,
            'table spanning cell': self.tsr_thr,
            'table projected row header': self.tsr_thr,
            'table column header': self.tsr_thr
        })

        # Enumerate all table cells: grid cells and spanning cells
        tables_cells = [tsr_utils.structure_to_cells(structure, tokens)[0] for structure in tables_structure]
        out_formats['cells'] = tables_cells

        # Convert cells to HTML
        tables_htmls = [tsr_utils.cells_to_html(cells) for cells in tables_cells]
        out_formats['html'] = tables_htmls

        # Convert cells to csv
        tables_csvs = [tsr_utils.cells_to_csv(cells) for cells in tables_cells]
        out_formats['csv'] = tables_csvs

        return out_formats, tsr_im