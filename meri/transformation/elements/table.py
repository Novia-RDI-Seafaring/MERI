import PIL.Image
from ...utils import GPTExtractor, GPT_TOOL_FUNCTIONS
from ...utils import pil_to_base64, pdf_to_im #pil_to_base64, pdf_to_imm
from ...utils.pydantic_models import TableContentModel, TableCellContentModel, TableMetaDataModel, TableContentArrayModel, TableArrayModel2, TableCellModel, TableModel2
from .page_element import PageElement
from ...utils.table_structure_recognizer import TSRBasedTableExtractor
#from .old_llm_extractor import GPTLayoutElementExtractor, GPT_TOOL_FUNCTIONS
from PIL import Image
import PIL
import pdfplumber.page
import fitz
import numpy as np
from typing import Tuple, List, Dict, Union

class Table(PageElement):
    """ Content class for Table elements.
    """

    def __init__(self, pdf_bbox: Tuple[float], im: Image, fitz_page: fitz.Page, plumber_page: pdfplumber.page, method='llm') -> None:
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
        self.content: TableContentArrayModel = None

    @classmethod
    def extract_table_plumber(cls, plumber_page: pdfplumber.page, clip = None) -> TableContentArrayModel:
        """_summary_

        Args:
            plumber_page (pdfplumber.page): _description_
            clip (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: list of detected tables. Each table is of type TableContentModel
            e.g. [
                [[cell1, cell2,...],    # row1
                [cell3, cell4]]         # row2
            ]
        """
        
        # Extracting potential tables using pdfplumber
        potential_tables = cls.extract_potential_tables_pdfplumber(plumber_page, clip)

        if len(potential_tables)>0: 
            # Extract the text from the potential tables
            tables = []
            for table in potential_tables:
                
                # initialize metadata as empty
                table_metadata = TableMetaDataModel(title='', description='')

                # Extract text from each block in the table and structure it into rows
                table_rows = []
                current_row = []
                last_top = table[0]['top']

                for block in table:
                    if block['top'] != last_top:
                        if current_row:
                            table_rows.append(current_row)
                        current_row = []
                    current_row.append(TableCellContentModel(text=block['text'], isheader=False, rowspan=1, colspan=1, bbox=[block['x0'], block['top'], block['x1'], block['bottom']]))
                    #current_row.append({'text': block['text'], 'bbox': [block['x0'], block['top'], block['x1'], block['bottom']]})
                    last_top = block['top']
                if current_row:
                    table_rows.append(current_row)

                tables.append(TableContentModel(metadata=table_metadata.model_dump(), rows=[[cell.model_dump() for cell in row] for row in table_rows]))
                #tables.append(table_text)
            
            return TableContentArrayModel(table_contents=[table.model_dump() for table in tables])
        else:
            return []

    @classmethod
    def extract_table_llm(cls, fitz_page: fitz.Page, clip = None, custom_jinja_prompt=None) -> TableContentArrayModel:
        table_im = pdf_to_im(fitz_page, cropbbox=clip)

        words = fitz_page.get_textpage(clip=clip).extractWORDS()
        words = [w[:5] for w in words]

        gpt_extractor = GPTExtractor() #GPTLayoutElementExtractor()
        try:
            tables = gpt_extractor.extract_content(GPT_TOOL_FUNCTIONS.EXTRACT_TABLE_CONTENT, table_im, words_arr=words, custom_jinja_prompt=custom_jinja_prompt)
        except Exception as e:
            print('Exception occured when extracting table data with llm: ', e)
            tables = []

        return tables

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

    def extract_tables_tatr(cls, table_im, fitz_page, table_bbox):
        table_extractor = TSRBasedTableExtractor(tsr_thr=0.85)
        out_formats, _, _ = table_extractor.cell_based_extract(
            table_np_im=np.asarray(table_im),
            pdf_full_fitz_page=fitz_page,
            table_bbox=table_bbox
        )

        # only one table
        tables = [TableModel2.from_tsr_cells(table_cells) for table_cells in out_formats['cells']]
        return TableArrayModel2(table_contents=tables)


    def get_content(self) -> TableContentArrayModel | Image.Image:

        if self.content is None:
            if self.method == 'pdfplumber':
                self.content = self.extract_table_plumber(self.plumber_page, clip=self.outer_bbox)
            elif self.method == 'llm':
                self.content = self.extract_table_llm(self.fitz_page, clip=self.outer_bbox)
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
        content: TableContentArrayModel | Image.Image = self.get_content()

        if not content:
            print("No content found.")
            return ""

        # if table is converted to image, just return base64 encoding
        if self.method == 'toimage':
            image_base64 = pil_to_base64(content)
            return f'![Figure](data:image/png;base64,{image_base64})'
        
        # Generate the markdown for each table
        markdown_tables = []
        for table in content.table_contents:
            markdown_tables.append(table.to_markdown(render_meta_data=False))

        print(f"Generated markdown tables: {markdown_tables}")
        return "{}".format(self.bbox_html_comment) + "\n\n".join(markdown_tables+['<br/>'])

    @property
    def outer_image(self):
        return pdf_to_im(self.fitz_page, self.outer_bbox)
    

