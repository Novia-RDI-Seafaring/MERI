import numpy as np
from pydantic import BaseModel
from pydantic import Field
import pandas as pd
from typing import OrderedDict, List
from pathlib import Path
import xml.etree.ElementTree as ET
import ast

class TableCellModel(BaseModel):
    text : str = Field(..., description='content of the cell')
    row_nums: List[int] = Field(..., description='list of row indices the row spans over. Length of this List equals the row span.')
    col_nums: List[int] = Field(..., description='list of column indices the column spans over. Length of this List equals the column span.')
    col_header: bool = Field(..., description='True if cell is column header.')
    bbox: List[float] = Field(..., description='bbox of cells text as coco coordinates.')
    
class TableMetaDataModel(BaseModel):
    title: str = Field(..., description='Title of the table.')
    description: str = Field(..., description='Short description what the table is about.')

class TableModel(BaseModel):
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
