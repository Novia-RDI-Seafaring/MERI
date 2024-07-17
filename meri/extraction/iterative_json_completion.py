from enum import Enum
from typing import List, Dict
from meri.utils.llm_utils import chat_completion_request
from meri.utils.prompts import generate_self_supervised_json_population_prompt
import json
import os
import openai
import tqdm

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


class IterativePopulationStrategies(Enum):

    # one parameters equals one extraction. Each parameter is only extracted
    # once. If parameter is found in part of document, we dont continue looking for it
    ONE2ONE = 'one2one'

    # one parameters equals multiple extractions. Each paramter can be extracted multiple
    # times, eventhough is has already been found in part of the document
    ONE2MANY = 'one2many'

    # LLM gets full schema and already filled parameters. LLM can decide itself if sth. is wrong
    # or should be left as is
    SELFSUPERVISED = 'selfsupervised'


class IterativeJsonPopulator:

    def __init__(self, json_schema_str: str, strategy: IterativePopulationStrategies, model = 'gpt-4o', api_key: str = None) -> None:
        self.json_schema_str = json_schema_str
        self.population_strategy = strategy
        self.model = model

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")

        self.client = openai.Client(api_key=api_key)

    def complete(self, content_chunks: List[List[Dict]]):
        
        if self.population_strategy == IterativePopulationStrategies.ONE2ONE.value:
            results = self.one2one_completion(content_chunks)

        elif self.population_strategy == IterativePopulationStrategies.ONE2MANY.value:
            results = self.one2many_completion(content_chunks)
        
        elif self.population_strategy == IterativePopulationStrategies.SELFSUPERVISED.value:
            results = self.selfsupervised_completion(content_chunks)
        else:
            print(self.population_strategy)
            raise NotImplementedError

        return results            
    
    
    def selfsupervised_completion(self, content_chunks):
        populated_dict = {}
        tools = create_openai_tools_arr(func_name='populate_json_schema', 
                                        func_desc='populate a json schema',
                                        output_schema=json.loads(self.json_schema_str))

        for c_chunk in tqdm.tqdm(content_chunks, total=len(content_chunks), desc='Processing content chunks'):

            try:
                prompt = generate_self_supervised_json_population_prompt(populated_dict)

                # construct messages array by iterating through it, until base64 is reached, put as type test
                # then base 64 as type image_url then again text ... First message is then instruction
                messages = [
                    {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}] + c_chunk,
                    }
                ]

                chat_response = chat_completion_request(client=self.client,
                                    messages=messages,
                                    tools=tools,
                                    tool_choice={"type": "function", "function": {"name": "populate_json_schema"}},
                                    model=self.model,
                                    log_token_usage=True)

                # check if message is complete, else JSON is incorrect
                if chat_response.choices[0].finish_reason == 'length':
                    print('GPT finished generation with finish reason length.')
                    #raise RuntimeError('GPT finished generation with finish reason length.')

                tool_calls = chat_response.choices[0].message.tool_calls

                # update populated dict
                populated_dict = json.loads(tool_calls[0].function.arguments)
            except Exception as e:
                print('Could finish schema population iteration: ', e)
        

        return populated_dict

    def one2one_completion(self, message_chunks):

        populated_json_dict = {}
        stripped_json_schema_str = self.json_schema_str
        for messages in message_chunks:
            # update populated_json_dict
            pass
            

    def one2many_completion(self, message_chunks):
        pass
