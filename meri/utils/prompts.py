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
    from the table and return the results as a JSON. The provided data can contain multiple tables. Return tables seperated.
    
    
    The bounding box should outline the respective cells in each row. Here are the results from OCR that should help you finding the bounding boxes of each cell. Multiple words can be 
    in one cell, then you need to combine the bounding boxes from the OCR results. The ocr information has the format
    [(x0,y0,x1,y1,word), ...]: {{ words_arr }}.    
    """
)

def generate_spec_table_extraction_prompt(words_arr):
    return SPEC_TABLE_EXTRACTION_TMPL.render(words_arr=words_arr)

TABLE_STRUCTURE_RECOGNITION_TMPL = Template(
    """    
    """
)
def generate_tsr_prompt(words_arr):
    return TABLE_STRUCTURE_RECOGNITION_TMPL.render(words_arr=words_arr)


SELFSUPERVISED_SCHEMA_POPULATION_TMPL = Template(
    """ You are expert in understanding technical information from documents. You are provided with 
    a part of the document as markdown and are required to extract specific information from the document.
    Because you only get a part of the document some information might not be present. 

    Alongside the document snippet you get a dictionary that contains values that were already extracted from other snippets
    of the document. You are allowed to overwrite them, in case you think your document snippets provides
    other, better information in respect to this value.

    Already extraced information from other snippets: {{ some_dict }}.

    You separate numerical units from values into separate standard attributes but you also add a textual representation called text, 
    where value and unit are together.
    For each nested object in the JSON, provide the bounding box that outlines the property called 'text'.

    Return the results as JSON.
    """
    )

def generate_self_supervised_json_population_prompt(current_extracted_dict):
    return SELFSUPERVISED_SCHEMA_POPULATION_TMPL.render(some_dict = current_extracted_dict)