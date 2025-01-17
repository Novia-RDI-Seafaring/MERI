from jinja2 import Template

SELFSUPERVISED_SCHEMA_POPULATION_TMPL = Template(
    """
        Context:
            - You are an expert system trained to understand and process technical information from documents.
            - Avoid false extractions by only extracting information where you are 99 percent confident in its accuracy.
        
        Task:
            - You will be provided with:
                - The document as a list of html elements The html elements contain attributes regarding the bounding box as well as page indexes the
                content of the element is lcoated in the origianl document.
            - A dictionary ({{ some_dict }}) containing previously extracted information from other parts of the document.
            - You are required to extract specific data points from the provided markdown snippet.
        
        Data Representation:
            - Extracted numeric values will be separated from their units.
                - A standard attribute will hold the numerical value (value).
                - Another attribute (unit) will hold the corresponding unit of measurement (e.g., "cm", "kg", "%").
            - If the datapoint is only a string and does not have a unit:
                - use the string as the value
                - use None/null for the unit
            - the provided schema will contain information about the "desiredUnit" (target unit). In the html document
                the value might be given in another unit (source unit). In this case you have to convert the value from the source unit to the target unit. 
                For example: if the source value is 100 and the source unit mm and the desiredUnit (target unit) as specified in the schema is cm, then
                the correct value for the extracted parameter is 10 and the unit cm.
            - each html element contains a reference to the page (page index) and rectancle (bounding box) where the elements content is located in the original document.
                For some parameters the required information might be scattered around in the document, in those cases provide the reference to all relevant information for
                infering the parameter.
            
        Guidelines:
            - Minimize false extractions. Only extract information where you are 99 percent confident that it is correct.
            - extracting the "value" might require simple computation based on the "text". e.g. if text is "3+4" the value should be 7 OR 3x 4 then 12.
        
        Output:
            Return the extracted data in JSON format and pay attention to the provided json schema.
    """
    )

def generate_self_supervised_json_population_prompt(current_extracted_dict):
    return SELFSUPERVISED_SCHEMA_POPULATION_TMPL.render(some_dict = current_extracted_dict)