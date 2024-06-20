import json
import os
import openai
from meri.utils.llm_utils import chat_completion_request
from meri.utils.format_handler import MarkdownHandler, BasicFormatHandler

def create_openai_tools_arr(func_name, func_desc, output_schema):


    tools = [{
        "type": "function",
        "function": {
            "name": func_name,
            "description": func_desc,
            "parameters": output_schema
            }
    }]
    return tools

class JsonExtractor:

    def __init__(self, intermediate_format: BasicFormatHandler, model='gpt-4o', api_key: str = None) -> None:
        
        self.intermediate_format = intermediate_format # markdown or html
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        self.client = openai.Client(api_key=api_key)
        self.model = model

    def populate_schema(self, json_schema_string: str, custom_prompt: str):
        """Populates json file based on provided json_schema

        Args:
            json_schema_string (str): _description_
        """

        prompt = custom_prompt
        tools = create_openai_tools_arr(func_name='populate_json_schema', 
                                        func_desc='populate a json schema',
                                        output_schema=json.loads(json_schema_string))
        

        chunks = self.intermediate_format.chunk()
        results = []
        for chunk in chunks:
            chunk_messages = self.intermediate_format.prepare_gpt_message_content(chunk)

            # construct messages array by iterating through it, until base64 is reached, put as type test
            # then base 64 as type image_url then again text ... First message is then instruction
            messages = [
                {
                "role": "user",
                "content": [{"type": "text", "text": prompt}] + chunk_messages,
                }
            ]
            print(messages)
            chat_response = chat_completion_request(client=self.client,
                                    messages=messages,
                                    tools=tools,
                                    tool_choice={"type": "function", "function": {"name": "populate_json_schema"}},
                                    model=self.model)



            # check if message is complete, else JSON is incorrect
            if chat_response.choices[0].finish_reason == 'length':
                raise RuntimeError('GPT finished generation with finish reason length.')
            
            tool_calls = chat_response.choices[0].message.tool_calls
            results.append(json.loads(tool_calls[0].function.arguments))

        return results