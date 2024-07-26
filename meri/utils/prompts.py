from jinja2 import Template

DEFAULT_TABLE_EXTRACTION_TMPL = Template(
    # """Extract all information from the table. The bounding box should outline the respective cells in each row.

    # Here are the results from OCR that should help you finding the bounding boxes of each cell. Multiple words can be 
    # in one cell, then you need to combine the bounding boxes from the OCR results. The ocr information has the format
    # [(x0,y0,x1,y1,word), ...]: {{ words_arr }}.
    # """
    
    """
        Task:

            - You are tasked with extracting all information from a provided table.
            - Accurately identify the structure of the table and extract the content of each cell.
            - For each cell, define a bounding box that outlines its location in the original table.
        
        Bounding Box Integration:

            - Information from Optical Character Recognition (OCR) will be provided to assist with bounding box creation.

            - The OCR data will be formatted as:

            [(x0, y0, x1, y1, word), ...]  // List of bounding boxes with words
                - (x0, y0): Top-left corner coordinates of the bounding box.
                - (x1, y1): Bottom-right corner coordinates of the bounding box.
                - word: Recognized text within the bounding box.
            - Combine overlapping bounding boxes from the OCR results ({{words_arr}}) to represent multi-word cells.

        Output:

            - Return the extracted data as a single JSON object with the following structure:
                {
                "data": [
                    [  // Array of rows (inner arrays represent cells)
                    {
                        "text": "Cell content",  // Extracted text content
                        "bbox": [x0, y0, x1, y1]   // Bounding box coordinates
                    },
                    // ... more cells in the row
                    ],
                    // ... more rows in the table
                ]
                }
    """
)

def generate_table_extraction_prompt(words_arr):
    return DEFAULT_TABLE_EXTRACTION_TMPL.render(words_arr=words_arr)

SPEC_TABLE_EXTRACTION_TMPL = Template(
    # """You are world class in identifying the structure of tables and their content. Extract all information
    # from the table and return the results as a JSON. The provided data can contain multiple tables. Return tables seperated.
    
    
    # The bounding box should outline the respective cells in each row. Here are the results from OCR that should help you finding the bounding boxes of each cell. Multiple words can be 
    # in one cell, then you need to combine the bounding boxes from the OCR results. The ocr information has the format
    # [(x0,y0,x1,y1,word), ...]: {{ words_arr }}.    
    # """
    
    """
        Context:
        
            - You are a highly skilled system capable of identifying table structure and extracting content from them.
            - Avoid false extractions by only extracting information where you are 99 percent confident in its accuracy.
            - You are not allow give any information you did not found in the table for the KEY VALUE pairs.

        Task:
        
            - You will be provided with data potentially containing multiple tables.
            - Extract all information from each table and return the results as separate JSON objects.
            - Accurately outline the bounding box for each cell within a row.
        
        Bounding Box Integration:

            - Information from Optical Character Recognition (OCR) will be provided in the format:
                [(x0, y0, x1, y1, word), ...]  // List of bounding boxes with words
                - (x0, y0): Top-left corner coordinates of the bounding box.
                - (x1, y1): Bottom-right corner coordinates of the bounding box.
                - word: Recognized text within the bounding box.
            - Combine overlapping bounding boxes from the OCR results ({{ words_arr }}) to represent multi-word cells.

        Output:

            - Return the extracted data for each table as a separate JSON object.

            - Each JSON object should represent the table structure with the following format:
                {
                "data": [
                    [  // Array of rows (inner arrays represent cells)
                    {
                        "text": "Cell content",  // Extracted text content
                        "bbox": [x0, y0, x1, y1]   // Bounding box coordinates
                    },
                    // ... more cells in the row
                    ],
                    // ... more rows in the table
                ]
                }

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
    # """ You are expert in understanding technical information from documents. You are provided with 
    # a part of the document as markdown and are required to extract specific information from the document.
    # Because you only get a part of the document some information might not be present. 

    # Alongside the document snippet you get a dictionary that contains values that were already extracted from other snippets
    # of the document. You are allowed to overwrite them, in case you think your document snippets provides
    # other, better information in respect to this value.

    # Already extraced information from other snippets: {{ some_dict }}.

    # You separate numerical units from values into separate standard attributes but you also add a textual representation called text, 
    # where value and unit are together.
    # For each nested object in the JSON, provide the bounding box that outlines the property called 'text'.

    # Return the results as JSON.
    # """
    """
        Context:
        
            -You are an expert system trained to understand and process technical information from documents.
            - Avoid false extractions by only extracting information where you are 99 percent confident in its accuracy.
            - You are not allow give any information you did not found in the table for the KEY VALUE pairs.
        Task:
        
            - You will be provided with:
                - A snippet of a document in markdown format.
                - A dictionary ({{ some_dict }}) containing previously extracted information from other parts of the document.
            - You are required to extract specific data points from the provided markdown snippet.
            - You may overwrite existing values in some_dict if the snippet offers more accurate information.
        
        Data Representation:
        
            - Extracted numeric values will be separated from their units.
                - A standard attribute will hold the numerical value (value).
                - Another attribute (unit) will hold the corresponding unit of measurement (e.g., "cm", "kg", "%").
            - A textual representation (text) will combine the value and unit for user-friendliness (e.g., "10 cm").
            - For nested objects in the resulting JSON, the text attribute will have a bounding box outlining its location in the original markdown snippet.
            
        Guidelines:
 
            - Minimize false extractions. Only extract information where you are 99 percent confident that it is correct. If you are not sure, dont populate the schema
                pop_schema with the parameters.
            - extracting the "value" might require simple computation based on the "text". e.g. if text is "3+4" the value should be
            - Dont rely on the information from technical drawings.
        
        Output:
        
            Return the extracted data in JSON format.
    """
    )

def generate_self_supervised_json_population_prompt(current_extracted_dict):
    return SELFSUPERVISED_SCHEMA_POPULATION_TMPL.render(some_dict = current_extracted_dict)