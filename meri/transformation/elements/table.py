from ...utils import pil_to_base64, pdf_to_im
from .page_element import PageElement
from PIL import Image
import pdfplumber.page
import fitz
from typing import Tuple, List, Dict

class Table(PageElement):
    """ Content class for Table elements.
    """

    def __init__(self, pdf_bbox: Tuple[float], im: Image, fitz_page: fitz.Page, plumber_page: pdfplumber.page) -> None:
        """
        - pdf_bbox: bounding box in pdf coordinates that outlines table. Given by deepdoctection pipeline
        - im: pil image of table as detected by detector.
        - page: fitz Page, helpful when applying find_tables methods from fitz library. clip page to pdf_bbox for table extraction
                based on pdf.
        """
        super().__init__(pdf_bbox)
        self.detectiion_im = im
        self.fitz_page = fitz_page
        self.plumber_page = plumber_page
        #self.content = None

    def extract_potential_tables(self):
        '''
        Extract potential tables from a page using pdfplumber
        Args:
            page: pdfplumber page object
        Returns:
            list of potential tables
        '''
        # print("Extracting potential tables from pdfplumber")
        crop_page = self.plumber_page.crop(self.pdf_bbox)  # Crop the page to the table bounding box
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

    def get_content(self) -> List[List[List[Dict[str, str | Tuple[float,float,float,float]]]]]:
        """ Extract and return content of table as a list of lists.
        """
        if self.content is None:
            # Extracting potential tables using pdfplumber
            potential_tables = self.extract_potential_tables()
            if potential_tables:
                # Extract the text from the potential tables
                tables = []
                for table in potential_tables:
                    # Extract text from each block in the table and structure it into rows
                    table_text = []
                    current_row = []
                    last_top = table[0]['top']

                    for block in table:
                        if block['top'] != last_top:
                            if current_row:
                                table_text.append(current_row)
                            current_row = []
                        current_row.append({'text': block['text'], 'bbox': [block['x0'], block['top'], block['x1'], block['bottom']]})
                        last_top = block['top']
                    if current_row:
                        table_text.append(current_row)

                    tables.append(table_text)

                self.content = tables
            else:
                print('Extract tables with fallback option fitz')
                # Fallback to fitz if pdfplumber fails
                table_rect = fitz.Rect(*self.pdf_bbox)
                table_text_page = self.fitz_page.get_textpage(clip=table_rect)
                table_text = table_text_page.extractText()

                table_content = []
                for line in table_text.split('\n'):
                    table_content.append([cell.strip() for cell in line.split(' ') if cell.strip()])

                self.content = [table_content]
        else:
            print("Content already extracted")

        return self.content

    def as_markdown_str(self) -> str:
        """ Convert table content into a markdown string.
        """
        content = self.get_content()

        if not content:
            print("No content found.")
            return ""

        # Generate the markdown for each table
        markdown_tables = []
        for table in content:
            table=[[cell['text'] for cell in row] for row in table]
            # If table is a list of lists of strings
            if not isinstance(table, list) or not all(isinstance(row, list) for row in table):
                continue

            # Determine the number of columns from the longest row
            num_columns = max(len(row) for row in table)
            # Normalize the rows to have the same number of columns
            normalized_content = [row + [''] * (num_columns - len(row)) for row in table]

            # Generate the header separator
            header_separator = " | ".join(["---"] * num_columns) + " |"

            # Generate the table rows
            markdown_table = [" | ".join(row) + " |" for row in normalized_content]

            # Combine everything into a markdown string
            markdown_str = "\n".join([markdown_table[0], header_separator] + markdown_table[1:])
            markdown_tables.append(markdown_str)

        print(f"Generated markdown tables: {markdown_tables}")
        return "\n\n".join(markdown_tables+['<br/>'])

    @property
    def outer_image(self):
        return pdf_to_im(self.fitz_page, self.outer_bbox)
    

