import openai
import deepdoctection as dd
from PIL import Image
from ...utils import chat_completion_request, pil_to_base64
import os
from enum import Enum
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv('.env'))


class GPT_TOOL_FUNCTIONS(Enum):
    """ values need to match function names in tools
    """
    EXTRACT_TABLE_CONTENT = 'extract_table_content'


class GPTLayoutElementExtractor:

    def __init__(self, api_key: str = None) -> None:
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.Client(api_key=api_key)
    
    def extract_content(self, tool_func: GPT_TOOL_FUNCTIONS, pil_im: Image, words_arr):
        """ uses multimodal gpt to extract information from an layout element. Leverages
        gpt function calls to force gpt response to follow a certain schema.

        tool_func: function_name of function in tools array that should be applied
        pil_im: pil image of the layout element. Is passed to gpt alongside the prompt
        words_arr: [(x0,y0,x1,y1,word), ...] words in the pil image

        """
        # select prompt in dependence to layout type
        if tool_func == GPT_TOOL_FUNCTIONS.EXTRACT_TABLE_CONTENT:
            prompt = f"""Extract all information from the table. The bounding box should outline the respective cells in each row.

                    Here are the results from OCR that should help you finding the bounding boxes of each cell. Multiple words can be 
                    in one cell, then you need to combine the bounding boxes from the OCR results. The ocr information has the format
                    [(x0,y0,x1,y1,word), ...]: {words_arr} 
                    """
        else:
            raise NotImplementedError

        messages = [
            {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": pil_to_base64(pil_im, raw=False)}},
            ],
            }
        ]

        chat_response = chat_completion_request(
            self.client, messages, tools=tools, tool_choice={"type": "function", "function": {"name": tool_func.value}}
        )
        tool_calls = chat_response.choices[0].message.tool_calls
        if tool_calls:
            kwargs = eval(tool_calls[0].function.arguments)
            if tool_func == GPT_TOOL_FUNCTIONS.EXTRACT_TABLE_CONTENT:
                return extract_table_content(**kwargs)
        else: 
            # Model did not identify a function to call, result can be returned to the user 
            print(chat_response.choices[0].message.content) 

def extract_table_content(rows, unmatchedData):
    print('rows: ', rows)
    print('umatcheddata: ', unmatchedData)

    # some postprocessing if multiple tables are detected
    return [rows], unmatchedData

tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_table_content",
            "description": "Extract all information from a table",
            "parameters": {
                "type": "object",
                "properties": {
                    "rows": {
                        "description": "List of rows of the table. First row is header",
                        "type": "array",
                        "items": {
                            "description": "List of cells representing the row",
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                        "text": {
                                            "type": "string"
                                        },
                                        "bbox": {
                                            "type": "array",
                                            "items": {
                                                "type": "number"
                                            },
                                            "minItems": 4,
                                            "maxItems": 4
                                            }
                                    },
                                    "required": [
                                        "text",
                                        "bbox"
                                ]
                            }
                        }
                    },
                    "unmatchedData": {
                        "description": "Contains information that is in the document but does not fit to any of the other specified fields",
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string"
                                },
                                "bbox": {
                                            "type": "array",
                                            "items": {
                                                "type": "number"
                                            },
                                            "minItems": 4,
                                            "maxItems": 4
                                            }
                            },
                            "required": [
                                "text",
                                "bbox",
                            ]
                        }
                    }
                },
                "required": ["rows", "unmatchedData"],
            },
        }
    },
]