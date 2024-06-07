import os
import sys
base_path = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))
sys.path.append(base_path)
import glob
import pandas as pd
from evaluation.table_evaluator import TableEvaluator, perfect_match

from evaluation.utils import parse_table_csv

def test_csv_format():
    """ Test if all .csvs in evaluation can be loaded
    """
    ann_path = os.path.join(base_path, 'data', 'table_extraction', 'annotations')
    assert os.path.exists(ann_path)
    csv_paths = glob.glob(os.path.join(ann_path, '**.csv'))
    for csv_path in csv_paths:
        try:
            df = parse_table_csv(csv_path)
        except Exception as e:
            assert False
    print(f'Loaded {len(csv_paths)} csvs')
    assert True

def test_csv_parser():
    """ check that csv parser for evaluation is working properly. Especially ; delimination
    """
    test_csv_values = parse_table_csv(os.path.join(base_path, 'tests/test_data/test_csv.csv'))
    test_csv_gt = [['cell1', 'cell2'],
                ['cell3', 'cell4']]
    assert test_csv_values == test_csv_gt

def test_table_comparison():
    """ Test compare table function
    """

    tab_gt = [[' val'], ['val'], ['vall']]
    tab_pred = [[{'text': 'val'}], [{'text': 'val'}], [{'text':'vaal'}]]
    res = TableEvaluator.compare_table(tab_pred, tab_gt, perfect_match)

    assert res['n_cells'] == 3
    assert res['correct_cells'] == 2
    assert res['acc'] == 2/3
    