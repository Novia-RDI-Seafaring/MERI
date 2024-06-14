import openai
import deepdoctection as dd
from PIL import Image
from ...utils import chat_completion_request, pil_to_base64, generate_table_extraction_prompt
from ...utils.pydantic_models import TableContentModel
import os
from enum import Enum
from typing import List
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
    
    def extract_content(self, tool_func: GPT_TOOL_FUNCTIONS, pil_im: Image, words_arr) -> List[TableContentModel]:
        """ uses multimodal gpt to extract information from an layout element. Leverages
        gpt function calls to force gpt response to follow a certain schema.

        tool_func: function_name of function in tools array that should be applied
        pil_im: pil image of the layout element. Is passed to gpt alongside the prompt
        words_arr: [(x0,y0,x1,y1,word), ...] words in the pil image

        Returns:
            List of TableContentModel: List of TableContentModel instances. one pil_im might contain multiple tables
        """
        # select prompt in dependence to layout type
        if tool_func.value == GPT_TOOL_FUNCTIONS.EXTRACT_TABLE_CONTENT.value:
            prompt = generate_table_extraction_prompt(words_arr)
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

        # check if message is complete, else JSON is incorrect
        if chat_response.choices[0].finish_reason == 'length':
            raise RuntimeError('GPT finished generation with finish reason length.')
        
        tool_calls = chat_response.choices[0].message.tool_calls
        if tool_func.value == GPT_TOOL_FUNCTIONS.EXTRACT_TABLE_CONTENT.value:

            table = TableContentModel.model_validate_json(tool_calls[0].function.arguments)
            return [table]
            
        else:
            raise NotImplementedError()

tools = [
    {
        "type": "function",
        "function": {
            "name": "extract_table_content",
            "description": "Get content of the table",
            "parameters": TableContentModel.model_json_schema()
            }
    }
]