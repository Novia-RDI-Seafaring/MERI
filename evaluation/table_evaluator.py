import fitz
import glob
import json
from pathlib import Path
import pandas as pd
import os
from evaluation.datasets import TableDataset
from evaluation.utils import parse_table_csv, store_table_csv, store_as_pickle, load_pickle
from meri.transformation.elements.table import Table as MeriTable
from meri.utils import pdf_to_im
from typing import List
import tqdm
from pathlib import Path
import pickle

def perfect_match(pred: str, gt: str):
    """ apply strip to remove spaces at beginning and end of string.
    """

    return pred.strip() == gt.strip()


class TableEvaluator:

    def __init__(self, annotations_path: str, pdfs_path: str, extraction_method: str, criteria, res_dir:str, ann_json_name='annotation.json'):

        self.annotations_path = annotations_path
        self.pdfs_path = pdfs_path
        self.criteria = criteria
        self.ann_json_name = ann_json_name

        # extracted tables are stored there and loaded if already present
        self.res_dir = res_dir

        # list of evaluation criteria to use
        self.extraction_method = extraction_method


    @classmethod
    def compare_table(cls, table_pred: List[List], table_gt: List[List], criteria):
        """Compares cells of two tables. 

        Args:
            table_pred (List[List]): predicted table as list of lists (rows) with each element being dict
            of {'text': <str>, 'bbox': [x0,y0,x1,y1]}.
            table_gt (List[List]): gt table as list of lists (rows).
            criteria (_type_): function that returns true if two cells match and false otherwise.

        Returns:
            dict:  n_cells, correct_cells, correct_cells/n_cells, incorrect
        }
        """
        # count correct cells
        correct_cells = 0
        # count all cells
        n_cells = 0
        # track incorrect matches {'gt': str, 'pred': str}
        incorrect = []
        for i, gt_row in enumerate(table_gt):

            if len(table_pred)==0:
                break

            # account for mismatch in number of rows
            if i>= len(table_pred):
                n_cells += len(gt_row)
                continue

            pred_row = table_pred[i]
            for j, gt_cell in enumerate(gt_row):
                n_cells += 1

                # account for mismatch is number of row entries
                if j >= len(pred_row):
                    # only count not empty cells as incorrect
                    if gt_cell.strip() == '':
                        correct_cells += 1
                    else:
                        incorrect.append({'gt': gt_cell, 'pred': "note: cell not detected"})
                    continue
                    

                pred_cell = pred_row[j]['text']
                if criteria(pred_cell, gt_cell):
                    correct_cells += 1
                else:
                    incorrect.append({'gt': gt_cell, 'pred': pred_cell})

        res = {
            'n_cells': n_cells,
            'correct_cells': correct_cells,
            'acc': correct_cells/n_cells if n_cells>0 else 0,
            'incorrect_matches': incorrect
        }
        
        return res



    def evaluate(self, existsOk=True):
        """Evaluates table extraction methods used in transformation module. Caches extracted tables in
        <self.res_dir>/table_extraction/<extraction method>/pred_<pdf_file_name>_<page_id>_<table_id>. Table id and page
        id start at idx 1 NOT 0.

        Arguments:
            existsOk (bool): if true will load extracted tables if already extracted

        Raises:
            NotImplementedError: if table extraction method is not implemented

        Returns:
            dict: dictionary with results
        """

        table_results = []

        for fitz_page, plumber_page, bbox, table_gt, ref in tqdm.tqdm(TableDataset(self.annotations_path, self.pdfs_path, self.ann_json_name)):

            res_path = os.path.join(self.res_dir, 'table_extraction', self.extraction_method, 'pred_{}_{}_{}.pickle'.format(
                Path(ref['file']).stem, ref["page_id"], ref["table_id"]))
            os.makedirs(os.path.dirname(res_path), exist_ok=True)
            res_exists = os.path.exists(res_path)
            
            try:
                if res_exists and existsOk:
                    table_pred = load_pickle(res_path)
                    print('Loaded cached results from {}'.format(res_path))
                else:
                    # extract data according to method
                    if self.extraction_method == 'llm':
                        table_pred = MeriTable.extract_table_llm(fitz_page, clip=bbox)
                    elif self.extraction_method == 'pdfplumber':

                        table_pred = MeriTable.extract_table_plumber(plumber_page, clip=bbox)
                    else:
                        raise NotImplementedError

                    assert len(table_pred) <= 1
                    table_pred = table_pred[0] if len(table_pred)>0 else []
                    store_as_pickle(table_pred, res_path)
                    print('Stored extracted table at: {}'.format(res_path))

                table_res = self.compare_table(table_pred, table_gt, self.criteria)
                table_res['ref'] = ref
                table_results.append(table_res)

            except Exception as e:
                print('Error occured while extracting tablular data from {}: {}'.format(ref, e))

        overall_res = {
            'total_tables': 0,
            'total_cells': 0,
            'correct_cells': 0,
            'avg_acc': 0, # average over tables
            'acc': 0, # computed over all cells
            'incorrect_matches': []
        }

        for table_res in table_results:
            overall_res['total_tables'] += 1
            overall_res['total_cells'] += table_res['n_cells']
            overall_res['incorrect_matches'] += table_res['incorrect_matches']
            overall_res['correct_cells'] += table_res['correct_cells']
            overall_res['avg_acc'] += table_res['acc']/len(table_results) if len(table_results) > 0 else 0

        overall_res['acc'] = overall_res['correct_cells']/overall_res['total_cells'] if overall_res['total_cells']>0 else 0

        return overall_res, table_results