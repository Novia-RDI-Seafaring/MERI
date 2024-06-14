from pydantic import BaseModel
from pydantic import Field
from pydantic.config import ConfigDict
import json
import pandas as pd
import os
from typing import Tuple, List, Self

class TableMetaDataModel(BaseModel):
    title: str = Field(..., description='Title of the table.')
    description: str = Field(..., description='Short description what the table is about.')

class TableCellContentModel(BaseModel):
    text : str = Field(..., description='content of the cell')
    isheader: bool = Field(..., description='True if cell is a header.')
    rowspan: int = Field(..., description='number of rows the cell spans over. Same as html table rowspan attribute.')
    colspan: int = Field(..., description='number of columns the cell spans over. Same as html table colspan attribute.')
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

    def compare_to(self, Self):
        pass


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

        # Create a matrix to hold the final table structure with spans considered
        table_matrix = [["" for _ in range(max_cols)] for _ in range(len(rows) + sum(cell.rowspan - 1 for row in rows for cell in row))]
        header_row_idx = None

        # Fill the table matrix with cell content considering rowspan and colspan
        for row_idx, row in enumerate(rows):
            col_idx = 0
            for cell in row:
                while col_idx < max_cols and table_matrix[row_idx][col_idx]:
                    col_idx += 1

                for r in range(cell.rowspan):
                    for c in range(cell.colspan):
                        if row_idx + r < len(table_matrix) and col_idx + c < max_cols:
                            # empty cell as html block https://stackoverflow.com/questions/17536216/create-a-table-without-a-header-in-markdown
                            table_matrix[row_idx + r][col_idx + c] = cell.text if cell.text != 'NaN' else '<!-- -->'
                
                col_idx += cell.colspan

            # Identify the header row
            if header_row_idx is None and any(cell.isheader for cell in row):
                header_row_idx = row_idx

        # if no header 1st row is
        if header_row_idx is None:
            header_row_idx = 0

        # Convert table matrix to markdown format
        if header_row_idx is not None:
            header_row = table_matrix[header_row_idx]
            markdown += '| ' + ' | '.join(header_row) + ' |\n'
            markdown += '| ' + ' | '.join('---' if cell else '' for cell in header_row) + ' |\n'

            for row_idx, row in enumerate(table_matrix):
                if row_idx != header_row_idx:
                    markdown += '| ' + ' | '.join(row) + ' |\n'
                


        return markdown