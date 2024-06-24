from typing import List
from .page_element import PageElement
import fitz

class TextBlock(PageElement):
    """ Class representing basic building elements of the pdf. pdf.get_text_blocks results in list of textblocks
    """

    def __init__(self, pdf_bbox, block_no, text_page: fitz.TextPage, page_idx) -> None:
        '''
        text_page: enables extraction of arbitrary detail of information for text 
            https://pymupdf.readthedocs.io/en/latest/textpage.html#textpage. 
            E.g. text_page.extractTEXT() gives text or text_page.extractDICT() gives detailed information
            like font, size, etc

        '''
        super().__init__(pdf_bbox, page_idx)

        self.seqno = block_no # determines reading order
        self.text_page = text_page

        # if add-font_info_to_textbocks is called, will set this property according to font size in relation
        # to the rest of the document
        self.text_type = None 

    def as_markdown_str(self):
        """ Converts text to markdown. Textblocks might have text_type property. Adds bbox of element as html comment to markdown string
        """
        content = self.get_content()
        
        if self.text_type is None:
            return '<br/>{} {} <br/>'.format(self.bbox_html_comment, content)
        
        text_type_list = self.text_type.split('_')

        assert len(text_type_list) == 2
        markdown_str=''
        if text_type_list[0] == 'h':
            markdown_str+= (int(text_type_list[1]) * '#' + ' ')

            markdown_str += content
            return markdown_str
        
        else:
            return '<br/>{} {} <br/>'.format(self.bbox_html_comment, content)
        

    def text(self):
        return self.text_page.extractText()

    def details(self):
        return self.text_page.extractDICT()

    def get_content(self):

        if self.content is not None:
            return self.content
        
        content = self.text()
        self.content = self.text()
        return content
    
    def get_order(self):
        return self.seqno

def get_font_sizes(tbs: List[TextBlock], setattrs=False):
    """ Computes dictionary with font sizes as values and the number of 
    letters having this font size as value.

    Returns:
    {<font_size>: <count>}
    """
    fontsizes = {}

    for tb in tbs:
        # assumes all content in textbox is same style
        tb_dict = tb.text_page.extractDICT()
        if len(tb_dict['blocks'])>0:
            first_span = tb_dict['blocks'][0]["lines"][0]['spans'][0]
            fontsz = round(first_span['size'])
            font = first_span['font']

            count = fontsizes.get(fontsz, 0) + len(tbs[0].text_page.extractTEXT().strip())
            fontsizes[fontsz] = count
        else:
            font = None
            fontsz = None
        if setattrs:
                setattr(tb, 'font', font)
                setattr(tb, 'fontsz', fontsz)

    return fontsizes

def add_font_info_to_textblocks(tbs: List[TextBlock], fontsizes_count_dict: dict) -> None:
    """ Inferes header and body tags for each textblock based on fontsizes_count_dict.
    Sets text_type attributes of textblock instances. 
    possible types: body_1, body_2, ..., h_1, h2, ...
    
    fontsizes_count_dict: {<font_size>: <count>, ...}
    """
    temp = sorted(
        [(k, v) for k, v in fontsizes_count_dict.items()],
        key=lambda i: i[1],
        reverse=True,
    )
    if temp:
        body_limit = temp[0][0]
    else:
        body_limit = 12

    text_type_mapper = {None: None}

    for i, size in enumerate(sorted(
                    [f for f in fontsizes_count_dict.keys() if f > body_limit], reverse=True
                )):
        text_type_mapper[size] = f"h_{i+1}"
    for i, size in enumerate(sorted(
                    [f for f in fontsizes_count_dict.keys() if f <= body_limit], reverse=True
                )):
        text_type_mapper[size] = f"body_{i+1}"

    for tb in tbs:
        setattr(tb, 'text_type', text_type_mapper[tb.fontsz])
