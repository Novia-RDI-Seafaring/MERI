from jinja2 import Template

DEFAULT_TABLE_EXTRACTION_TMPL = Template(
    """Extract all information from the table. The bounding box should outline the respective cells in each row.

    Here are the results from OCR that should help you finding the bounding boxes of each cell. Multiple words can be 
    in one cell, then you need to combine the bounding boxes from the OCR results. The ocr information has the format
    [(x0,y0,x1,y1,word), ...]: {{ words_arr }}.
    """
)

def generate_table_extraction_prompt(words_arr):
    return DEFAULT_TABLE_EXTRACTION_TMPL.render(words_arr=words_arr)

SPEC_TABLE_EXTRACTION_TMPL = Template(
    """You are world class in identifying the structure of tables and their content. Extract all information
    from the table and return the results as a JSON.
    
    
    The bounding box should outline the respective cells in each row. Here are the results from OCR that should help you finding the bounding boxes of each cell. Multiple words can be 
    in one cell, then you need to combine the bounding boxes from the OCR results. The ocr information has the format
    [(x0,y0,x1,y1,word), ...]: {{ words_arr }}.    
    """
)

def generate_spec_table_extraction_prompt(words_arr):
    return SPEC_TABLE_EXTRACTION_TMPL.render(words_arr=words_arr)

## organize the output according to the following json schema:  {{ output_json_schema }}.