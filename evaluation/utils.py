
import os
import fitz
import json
import pandas as pd
from meri.utils import scale_coords
import pdfplumber
from typing import List
import json
import pickle

def store_table_csv(values: List[List], store_path:str):
    df = pd.DataFrame(values)
    
    df.to_csv(store_path, index=False, header=False, sep=';')

def store_as_pickle(values, store_path: str):
    with open(store_path, "wb") as fp:   #Pickling
        pickle.dump(values, fp)

def load_pickle(path):
    with open(path, "rb") as fp:   # Unpickling
        return pickle.load(fp)

def parse_table_csv(path):
    """ Reads table with ; as seperator
    """
    df = pd.read_csv(path, sep=';', header=None)
    df = df.fillna('')
    return df.values.tolist()

def coco_to_xyxy(coco_box):
    """ Convert coco (xywh) to xyxy
    """
    x,y,w,h = coco_box
    return [x, y, x+w, y+h] 

def parse_table_annotations(path_to_annotations_dir, path_to_pdf_dir, ann_json_name='annotation.json'):
    """ 

    path_to_annotations_dir (str): path to directory that holds table annotations as one .csv per table
    and json file with annotations.
    path_to_pdf_dir (str): path to directory that contains pdf files for evaluation

    Returns:
    List of fitz_pages, plumber_pages, bboxes, table_contents, references.
    """

    with open(os.path.join(path_to_annotations_dir, ann_json_name), 'r') as f:
        coco = json.load(f)
    
    bboxes = []
    fitz_pages = []
    plumber_pages = []
    table_contents = []
    references = []
    for i, ann in enumerate(coco['annotations']):

        # parse fitz page
        pdf_info = coco["pdfs"][ann["pdf_id"]-1]
        pdf_file = pdf_info["file_name"]
        pdf_doc = fitz.open(os.path.join(path_to_pdf_dir, pdf_file))
        page_i = ann["page_id"]-1
        fitz_page = pdf_doc[page_i]
        fitz_pages.append(fitz_page)

        # parse plumber page
        plumber_doc = pdfplumber.open(os.path.join(path_to_pdf_dir, pdf_file))
        plumber_pages.append(plumber_doc.pages[page_i])


        # parse bbox of table, convert from coco (xywh) to xyxy
        source_height, source_width = pdf_info['height'], pdf_info['width']
        target_height, target_width = fitz_page.rect.height, fitz_page.rect.width
        target_bbox = scale_coords(coco_to_xyxy(ann["bbox"]), source_height, source_width, target_height, target_width)
        bboxes.append(target_bbox)

        # parse table content as 2d array
        table_list = parse_table_csv(os.path.join(path_to_annotations_dir, ann["table_content_file"]))
        table_contents.append(table_list)

        # reference to doc and table
        ref = {
            'file': pdf_file,
            'page_id': ann["page_id"],
            'table_id': ann["table_id"]
            }
        references.append(ref)

    return fitz_pages, plumber_pages, bboxes, table_contents, references