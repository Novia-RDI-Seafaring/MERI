from ...utils import GPTExtractor, GPT_TOOL_FUNCTIONS
from ...utils import pil_to_base64, pdf_to_im, sub_coords_to_abs_coords #pil_to_base64, pdf_to_imm
from ...utils.pydantic_models import TableContentModel, TableCellContentModel, TableMetaDataModel, TableContentArrayModel, TableArrayModel2, TableCellModel, TableModel2
from .page_element import PageElement
from ...utils.table_structure_recognizer import TSRBasedTableExtractor
#from .old_llm_extractor import GPTLayoutElementExtractor, GPT_TOOL_FUNCTIONS
from PIL import Image
import PIL
import xml.etree.ElementTree as ET

import pdfplumber.page
import fitz
import numpy as np
from typing import Tuple, List, Dict, Union

class Table(PageElement):
    """ Content class for Table elements.
    """

    def __init__(self, pdf_bbox: Tuple[float], im: Image, fitz_page: fitz.Page, plumber_page: pdfplumber.page, method='llm', model='gpt-4o-mini') -> None:
        """
        - pdf_bbox: bounding box in pdf coordinates that outlines table. Given by deepdoctection pipeline
        - im: pil image of table as detected by detector.
        - page: fitz Page, helpful when applying find_tables methods from fitz library. clip page to pdf_bbox for table extraction
                based on pdf.
        """
        super().__init__(pdf_bbox, fitz_page.number)
        self.detectiion_im = im
        self.fitz_page = fitz_page
        self.plumber_page = plumber_page
        self.method = method
        self.model = model
        self.content: TableContentArrayModel = None

    @classmethod
    def extract_table_plumber(cls, plumber_page: pdfplumber.page, clip=None) -> TableArrayModel2:
        # Extracting potential tables using pdfplumber
        potential_tables = cls.extract_potential_tables_pdfplumber(plumber_page, clip)

        if len(potential_tables) > 0:
            tables = []
            for table in potential_tables:
                table_metadata = TableMetaDataModel(title='', description='')

                # Extract text from each block in the table and structure it into rows
                table_rows = []
                current_row = []
                last_top = int(round(table[0]['top']))  # yhe integer value

                for block in table:
                    top = int(round(block['top']))  # The integer value
                    if top != last_top:
                        if current_row:
                            table_rows.append(current_row)
                        current_row = []
                    current_row.append(TableCellModel(
                        text=block['text'],
                        col_header=False,
                        row_nums=[top],  # The integer value
                        col_nums=[int(round(block['x0']))],  # the integer value
                        bbox=[block['x0'], block['top'], block['x1'], block['bottom']]
                    ))
                    last_top = top
                if current_row:
                    table_rows.append(current_row)

                cells = [cell for row in table_rows for cell in row]
                tables.append(TableModel2(metadata=table_metadata, cells=cells))

            return TableArrayModel2(table_contents=tables)
        else:
            return []


    def contained_words(self):
        words = self.fitz_page.get_textpage(clip=self.outer_bbox).extractWORDS()
        words = [w[:5] for w in words]

        return words

    @classmethod
    def extract_table_llm(cls, fitz_page: fitz.Page, clip=None, custom_jinja_prompt=None, model='gpt-4o-mini') -> TableArrayModel2:
        table_im = pdf_to_im(fitz_page, cropbbox=clip)

        words = fitz_page.get_textpage(clip=clip).extractWORDS()
        words = [w[:5] for w in words]

        gpt_extractor = GPTExtractor(model=model)  # GPTLayoutElementExtractor()
        try:
            raw_tables = gpt_extractor.extract_content(GPT_TOOL_FUNCTIONS.EXTRACT_TABLE_CONTENT, table_im, words_arr=words, custom_jinja_prompt=custom_jinja_prompt)
        except Exception as e:
            print('Exception occurred when extracting table data with llm: ', e)
            raw_tables = []

        tables = []
        for raw_table in raw_tables:
            # Debug print to understand the structure of raw_table
            print("raw_table structure:", raw_table)

            # raw_table contains a list of TableContentModel instances
            for table_content in raw_table[1]:
                if isinstance(table_content, TableContentModel):
                    table_metadata = table_content.metadata
                    rows_data = table_content.rows

                    # Create metadata
                    table_metadata = TableMetaDataModel(title=table_metadata.title, description=table_metadata.description)

                    # Process rows and cells
                    cells = []
                    for row_index, row in enumerate(rows_data):
                        for col_index, cell in enumerate(row):
                            cells.append(
                                TableCellModel(
                                    text=cell.text,
                                    row_nums=[row_index],  # Use the actual row index
                                    col_nums=[col_index],  # Use the actual column index
                                    col_header=cell.isheader,
                                    bbox=cell.bbox
                                )
                            )

                    tables.append(TableModel2(metadata=table_metadata, cells=cells))
                else:
                    print("Unexpected table_content format:", table_content)

        return TableArrayModel2(table_contents=tables)


    @classmethod
    def extract_potential_tables_pdfplumber(cls, plumber_page: pdfplumber.page, clip = None):
        '''
        Extract potential tables from a page using pdfplumber
        Args:
            page: pdfplumber page object
        Returns:
            list of potential tables
        '''
        # print("Extracting potential tables from pdfplumber")
        crop_page = plumber_page.crop(clip)  # Crop the page to the table bounding box
        
        # Extract text blocks from the page
        text_blocks = crop_page.extract_words(keep_blank_chars=True)
        if not text_blocks:
            return []
        # Sort the text blocks by top(vertical) and x0(horizontal) left coordinates
        text_blocks_sorted = sorted(text_blocks, key=lambda b: (b['top'], b['x0']))
        potential_tables = []
        current_table = [text_blocks_sorted[0]]

        for block in text_blocks_sorted[1:]:
            last_block = current_table[-1]
            # If the current block is close to the last one vertically, consider it part of the same table
            if (block['top'] - last_block['bottom']) < 15:
                current_table.append(block)
            else:
                # If the current block is far from the last one vertically, consider it a new table
                if len(current_table) > 3:
                    potential_tables.append(current_table)
                current_table = [block]
        # Check the last accumulated table
        if len(current_table) > 3:
            potential_tables.append(current_table)

        return potential_tables
    
    @classmethod
    def extract_tables_tatr(cls, table_im, fitz_page, table_bbox):
        table_extractor = TSRBasedTableExtractor(tsr_thr=0.85)
        out_formats, _, _ = table_extractor.cell_based_extract(
            table_np_im=np.asarray(table_im),
            pdf_full_fitz_page=fitz_page,
            table_bbox=table_bbox
        )

        # cell bbox are in cropped im coordinates. need to scale to full page coordinates
        source_height, source_width = np.array(table_im).shape[:2]
        for table_cells in out_formats["cells"]:
            for cell in table_cells:
                cell["bbox"] = sub_coords_to_abs_coords(cell["bbox"] , source_height, source_width, table_bbox)

        # only one table
        tables = [TableModel2.from_tsr_cells(table_cells) for table_cells in out_formats['cells']]
        return TableArrayModel2(table_contents=tables)


    def get_content(self) -> TableArrayModel2 | Image.Image:

        if self.content is None:
            if self.method == 'pdfplumber':
                self.content = self.extract_table_plumber(self.plumber_page, clip=self.outer_bbox)
            elif self.method == 'llm':
                self.content = self.extract_table_llm(self.fitz_page, clip=self.outer_bbox, model=self.model)
            elif self.method == 'toimage':
                self.content = self.outer_image
            elif self.method == 'tatr':
                self.content = self.extract_tables_tatr(self.outer_image, self.fitz_page, self.outer_bbox)
            else:
                raise NotImplementedError
        else:
             print("Content already extracted")
        return self.content

    def as_markdown_str(self) -> str:
        """ Convert table content into a markdown string. Adds bbox of table element as html comment to markdown string
        """
        content: TableArrayModel2 | Image.Image = self.get_content()

        if not content:
            print("No content found.")
            return ""

        # if table is converted to image, just return base64 encoding
        if self.method == 'toimage':
            image_base64 = pil_to_base64(content)
            return '<div {} {}>{}</div>'.format(self.attribute_str, 
                                                  'className="image_wrapper"',
                                                  f'<img src="data:image/png;base64,{image_base64}"/>')
        
        # Generate the markdown for each table
        markdown_tables = []
        for table in content.table_contents:
            markdown_tables.append(table.to_markdown(render_meta_data=False, add_bbox_as_attr=True))

        print(f"Generated markdown tables: {markdown_tables}")
        print('<div {} {}>{}</div>'.format(self.attribute_str, 
                                            'className="table_wrapper"',
                                            " ".join(markdown_tables)))
        return '<div {} {}>{}</div>'.format(self.attribute_str, 
                                            'className="table_wrapper"',
                                            " ".join(markdown_tables))

    @property
    def outer_image(self):
        return pdf_to_im(self.fitz_page, self.outer_bbox)
    

