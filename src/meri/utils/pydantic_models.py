import numpy as np
from pydantic import BaseModel
from pydantic import Field
from pydantic.config import ConfigDict
import json
import pandas as pd
import os
from typing import OrderedDict, Tuple, List, Self
import pickle
from pathlib import Path
import xml.etree.ElementTree as ET
import ast


class TableMetaDataModel(BaseModel):
    title: str = Field(..., description='Title of the table.')
    description: str = Field(..., description='Short description what the table is about.')

class TableCellContentModel(BaseModel):
    text : str = Field(..., description='content of the cell')
    isheader: bool = Field(..., description='True if cell is a header.')
    rowspan: int = Field(default=1, description='number of rows the cell spans over. Same as html table rowspan attribute.')
    colspan: int = Field(default=1, description='number of columns the cell spans over. Same as html table colspan attribute.')
    bbox: List[float] = Field(..., description='bbox of cells text as coco coordinates.')

class TableContentModel(BaseModel):
    metadata: TableMetaDataModel = Field(..., description='metadata on the table.')
    rows: List[List[TableCellContentModel]] = Field(..., description='Array of rows, where each row is an array of cells.')

    @classmethod
    def from_csv(cls, csv_path: str, title: str = '', description: str = '') -> Self:
        """Creates instance of this class from csv table. 
        Instanciates TableCellContentModel for each cell with isheader=False, rowspan=1, colspan=1 
        and bbox = [0.0, 0.0, 0.0, 0.0] because these information are not contained in .csv.

        Args:
            csv_path (str): path to csv. .csv deliminator needs to be ';'. Reads csv without header=None

        Returns:
            TableContentModel: instance of TableContentModel
        """
        
        assert os.path.exists(csv_path)

        df = pd.read_csv(csv_path, sep=';', header=None).fillna('NaN')
        
        # Initialize rows list
        rows = []

        # Add data rows
        for _, row in df.iterrows():
            data_row = [TableCellContentModel(
                text=str(cell), isheader=False, rowspan=1, colspan=1, bbox=[0.0, 0.0, 0.0, 0.0]
            ) for cell in row]
            rows.append(data_row)

        # Create metadata
        metadata = TableMetaDataModel(title=title, description=description)

        # Create TableContentModel instance
        return cls(metadata=metadata, rows=rows)

    def to_pickle(self, path: str):
       
        assert Path(path).suffix == '.pickle'

        model_dict = self.model_dump()

        with open(path, "wb") as fp: 
            pickle.dump(model_dict, fp)
    
    @classmethod
    def from_pickle(cls, path) -> Self:
        assert os.path.exists(path)

        with open(path, "rb") as fp:   # Unpickling
            model_dict = pickle.load(fp)

        return cls.model_validate(model_dict)

    def to_matrix_str(self):
        """Adds bbox of each cell as html comment to cell string

        Returns:
            _type_: _description_
        """

        # Determine the number of columns by considering colspan
        max_cols = 0
        for row in self.rows:
            col_count = sum(cell.colspan for cell in row)
            max_cols = max(max_cols, col_count)

        # Create a matrix to hold the final table structure with spans considered
        table_matrix = [["" for _ in range(max_cols)] for _ in range(len(self.rows) + sum(cell.rowspan - 1 for row in self.rows for cell in row))]

        # Fill the table matrix with cell content considering rowspan and colspan
        for row_idx, row in enumerate(self.rows):
            col_idx = 0
            for cell in row:
                while col_idx < max_cols and table_matrix[row_idx][col_idx]:
                    col_idx += 1

                for r in range(cell.rowspan):
                    for c in range(cell.colspan):
                        if row_idx + r < len(table_matrix) and col_idx + c < max_cols:
                            # empty cell as html block https://stackoverflow.com/questions/17536216/create-a-table-without-a-header-in-markdown
                            #table_matrix[row_idx + r][col_idx + c] = "{} {} ".format(f"<!-- Bounding box (x0,y0,x1,y1): {cell.bbox} -->", cell.text.strip() if cell.text != 'NaN' else '') 
                            table_matrix[row_idx + r][col_idx + c] = "{} ".format( cell.text.strip() if cell.text != 'NaN' else '') 

                col_idx += cell.colspan

        return table_matrix
    
    def compare_to(self, other_table: Self, criteria):
        """

        Args:
            other_table (Self): table to compared with, assumes other table is the correct one e.g. ground truth table

        """

        self_str_matrix = self.to_matrix_str()
        other_str_matrix = other_table.to_matrix_str()

        # count correct cells
        correct_cells = 0
        # count all cells
        n_cells = 0
        # track incorrect matches {'gt': str, 'pred': str}
        incorrect = []

        for i, other_row in enumerate(other_str_matrix):
            if len(self_str_matrix)==0:
                break
            # account for mismatch in number of rows
            if i>= len(self_str_matrix):
                n_cells += len(other_row)
                continue

            self_row = self_str_matrix[i]
            for j, gt_cell in enumerate(other_row):
                n_cells += 1

                # account for mismatch is number of row entries
                if j >= len(self_row):
                    # only count not empty cells as incorrect
                    if gt_cell.strip() == '':
                        correct_cells += 1
                    else:
                        incorrect.append({'gt': gt_cell, 'pred': "note: cell not detected"})
                    continue
                    

                pred_cell = self_row[j]
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
        
    def to_markdown(self, render_meta_data: bool = False) -> str:
        """Converts table to markdown. If colspan or rowspan >1 the cell value will be duplicated
        over all columns/rows it spans over. If header is not set in any cell, will treat first row as header.
        Cells with value 'NaN' will be replaces by html comment symbol '<!-- -->' (thus will not render).

        Args:
            render_meta_data (str): if true, will render metadata above table

        Returns:
            str: markdown string
        """
        metadata = self.metadata
        rows = self.rows

        # Initialize markdown output
        if render_meta_data:

            markdown = f"# {metadata.title}\n\n"
            markdown += f"{metadata.description}\n\n"
        else:
            markdown = ""

        # Determine the number of columns by considering colspan
        max_cols = 0
        for row in rows:
            col_count = sum(cell.colspan for cell in row)
            max_cols = max(max_cols, col_count)

        header_row_idx = None
        table_matrix = self.to_matrix_str()
        
        for row in table_matrix:
            for cell in row:
                if cell == 'NaN' or cell == '':
                    cell = '<!-- -->'

        # if no header 1st row is
        if header_row_idx is None:
            header_row_idx = 0

        # Convert table matrix to markdown format
        if header_row_idx is not None:
            header_row = table_matrix[header_row_idx]
            markdown += '| ' + ' | '.join(header_row) + ' |\n'
            markdown += '| ' + ' | '.join('---' for _ in header_row) + ' |\n'
            #markdown += '| ' + ' | '.join('---' if cell else '' for cell in header_row) + ' |\n'

            for row_idx, row in enumerate(table_matrix):
                if row_idx != header_row_idx:
                    markdown += '| ' + ' | '.join(row) + ' |\n'
                


        return markdown
    
class TableContentArrayModel(BaseModel):
    table_contents: List[TableContentModel] = Field(..., description='List of tables.')

class TableStructureModel(BaseModel):
    textual_description: str = Field(..., description='textual description of the stucutre of the table.')

######################## new ####################
class TableCellModel(BaseModel):
    text : str = Field(..., description='content of the cell')
    row_nums: List[int] = Field(..., description='list of row indices the row spans over. Length of this List equals the row span.')
    col_nums: List[int] = Field(..., description='list of column indices the column spans over. Length of this List equals the column span.')
    col_header: bool = Field(..., description='True if cell is column header.')
    bbox: List[float] = Field(..., description='bbox of cells text as coco coordinates.')

class TableModel2(BaseModel):
    metadata: TableMetaDataModel = Field(..., description='metadata on the table.')
    cells: List[TableCellModel] = Field(..., description='list of cells in the table')

    def to_markdown(self, render_meta_data: bool = False, add_bbox_as_attr=True):

        return self.to_html(add_bbox_as_attr)

    def to_html(self, add_bbox_as_attr=False):
        cells = [c.model_dump() for c in self.cells]
        cells = sorted(cells, key=lambda k: min(k['col_nums']))
        cells = sorted(cells, key=lambda k: min(k['row_nums']))

        table = ET.Element("table", attrib=self.metadata.model_dump())
        current_row = -1

        for cell in cells:
            this_row = min(cell['row_nums'])

            if add_bbox_as_attr:
                attrib = {
                    'bbox': cell['bbox']
                }
            else:
                attrib = {}
            colspan = len(cell['col_nums'])
            if colspan > 1:
                attrib['colspan'] = str(colspan)
            rowspan = len(cell['row_nums'])
            if rowspan > 1:
                attrib['rowspan'] = str(rowspan)
            if this_row > current_row:
                current_row = this_row
                if cell['col_header']:
                    cell_tag = "th"
                    row = ET.SubElement(table, "thead")
                else:
                    cell_tag = "td"
                    row = ET.SubElement(table, "tr")
            tcell = ET.SubElement(row, cell_tag, attrib=attrib)
            tcell.text = cell['text']

        return str(ET.tostring(table, encoding="unicode", short_empty_elements=False))
    
    def to_df(self):
        cells = [c.model_dump() for c in self.cells]

        if len(cells) > 0:
            num_columns = max([max(cell['col_nums']) for cell in cells]) + 1
            num_rows = max([max(cell['row_nums']) for cell in cells]) + 1
        else:
            return

        header_cells = [cell for cell in cells if cell['col_header']]
        if len(header_cells) > 0:
            max_header_row = max([max(cell['row_nums']) for cell in header_cells])
        else:
            max_header_row = -1

        table_array = np.empty([num_rows, num_columns], dtype="object")
        if len(cells) > 0:
            for cell in cells:
                for row_num in cell['row_nums']:
                    for column_num in cell['col_nums']:
                        table_array[row_num, column_num] = cell["text"]

        header = table_array[:max_header_row+1,:]
        flattened_header = []
        for col in header.transpose():
            flattened_header.append(' | '.join(OrderedDict.fromkeys(col)))
        df = pd.DataFrame(table_array[max_header_row+1:,:], index=None, columns=flattened_header)

        return df

    def to_csv(self, csv_path: str):
        df = self.to_df()
        df.to_csv(csv_path, index=None)

    @classmethod
    def from_tsr_cells(cls, cells):

        return cls(metadata=TableMetaDataModel(title='', description=''),
                    cells=[TableCellModel(
                                text=c['cell text'],
                                row_nums=c['row_nums'],
                                col_nums=c['column_nums'],
                                col_header=c['column header'],
                                bbox=c['bbox']) for c in cells])
    
    @classmethod
    def from_html(cls, html_str):

        # Wrap the HTML string in a single root element
        html_string = f"<root>{html_str}</root>"

        # Parse the HTML string
        root = ET.fromstring(html_string)

        table = root.find('.//table')
        title = table.attrib['title'] if 'title' in table.attrib.keys() else ''
        description = table.attrib['description'] if 'description' in table.attrib.keys() else ''

        table_metadata = TableMetaDataModel(title=title, description=description)

        table_cells = []
        # Iterate through the rows and cells
        for row_i, row in enumerate(table):
            if row.tag in ['tr', 'thead']:
                for col_i, cell in enumerate(row):
                    attrib = cell.attrib
                    row_nums = [row_i] if 'rowspan' not in attrib.keys() else list(range(row_i, ast.literal_eval(attrib['rowspan'])))
                    col_nums = [col_i] if 'colspan' not in attrib.keys() else list(range(col_i, ast.literal_eval(attrib['colspan'])))
                    col_header = (cell.tag == 'th')
                    bbox = ast.literal_eval(attrib['bbox']) if 'bbox' in attrib.keys() else [0,0,0,0]
                    text = cell.text if cell.text else ''

                    cell = TableCellModel(
                        text=text,
                        row_nums=row_nums,
                        col_nums=col_nums,
                        col_header=col_header,
                        bbox = bbox
                        )
                    table_cells.append(cell)

        return cls(metadata=table_metadata, cells= table_cells)


class TableArrayModel2(BaseModel):
    table_contents: List[TableModel2] = Field(..., description='List of tables.')

    @classmethod
    def from_html(cls, html_str):

        html_string = f"<root>{html_str}</root>"
        
        # Parse the HTML string
        root = ET.fromstring(html_string)

        # Find all table elements
        tables = root.findall('.//table')

        # Extract each table as an HTML string
        table_strings = []
        for table in tables:
            # Convert each table element back to a string
            table_string = ET.tostring(table, encoding='unicode', method='html')
            table_strings.append(table_string)

        table_models = []
        for table_html_str in table_strings:
            table_models.append(TableModel2.from_html(table_html_str))

        return cls(table_contents = table_models)

