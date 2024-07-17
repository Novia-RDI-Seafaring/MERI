from .utils import parse_table_annotations
import json
import os
from pathlib import Path
import glob

class TableDataset:
    """ Python iterator that iterates through the dataset. Each sample of the dataset contains:
    fitz_page, pdfplumber page, bounding box of table (as xyxy), gt table content, reference to .csv with gt table content.
    """

    def __init__(self, annotations_path, pdf_path, ann_json_name='annotation.json'):

        self.fitz_pages, self.plumber_pages, self.bboxes, self.table_contents, self.references = parse_table_annotations(
            annotations_path, 
            pdf_path,
            ann_json_name)

        assert len(self.fitz_pages) == len(self.bboxes) == len(self.table_contents) == len(self.plumber_pages)

        self.n = len(self.fitz_pages)
        self.current=0

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current <= self.n:
            return (self.fitz_pages[self.current-1],
                    self.plumber_pages[self.current-1],
                    self.bboxes[self.current-1],
                    self.table_contents[self.current-1],
                    self.references[self.current-1])

        raise StopIteration


class PopulatedSchemaDataset():
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, json_schema_dir, gt_populated_dir, datasheets_dir):
        gt_paths = glob.glob(os.path.join(gt_populated_dir, '*.json'))

        self.datasheet_paths = []
        self.schema_paths = []
        self.gt_paths = []


        for gt_path in gt_paths:
            datasheet_id = Path(gt_path).stem
            schema_id = datasheet_id.split('_')[0]

            # get schema path
            schema_path = os.path.join(json_schema_dir, f"{schema_id}.json")
            if not os.path.exists(schema_path):
                print(f"Skipping {datasheet_id} because path to schema {schema_path} is missing.")
                continue

            # get datasheet path
            datasheet_path = os.path.join(datasheets_dir, f"{datasheet_id}.pdf")
            if not os.path.exists(datasheet_path):
                print(f"Skipping datasheet {datasheet_id} because path to datasheet {datasheet_path} is missing.")
                continue

            
            self.datasheet_paths.append(datasheet_path)
            self.schema_paths.append(schema_path)
            self.gt_paths.append(gt_path)
        
        assert len(self.datasheet_paths) == len(self.schema_paths) == len(self.gt_paths)

    def __getitem__(self, idx):
        """_summary_

        Args:
            idx (_type_): _description_

        Returns:
            (str, dict, dict): Tuple of (path to datasheet, schema as dict, gt populated schema as dict)
        """

        with open(self.schema_paths[idx]) as f:
            schema_dict = json.load(f)

        with open(self.gt_paths[idx]) as f:
            gt_dict = json.load(f)

        return (self.datasheet_paths[idx], schema_dict, gt_dict)

    def __len__(self):
        return len(self.datasheet_paths)