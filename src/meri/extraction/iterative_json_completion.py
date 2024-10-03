from enum import Enum
from typing import List, Dict
from meri.utils.llm_utils import chat_completion_request
from meri.utils.prompts import generate_self_supervised_json_population_prompt
import json
import os
import openai
import tqdm

def create_openai_response_format(name, schema):
    """_summary_

    Args:
        name (str): name of the response format
        schema (_type_): json schema as dict

    Returns:
        _type_: response_format parameter for openai API request
    """
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": f"{name}",
            "strict": True,
            "schema": schema,
        }
    }

    return response_format

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

    def __init__(self, json_schema_str: str, strategy: IterativePopulationStrategies, n_rounds=3, model = 'gpt-4o-mini', api_key: str = None) -> None:
        self.json_schema_str = json_schema_str
        self.population_strategy = strategy
        self.model = model
        self.n_rounds = n_rounds

    def get_response_format(self):
        schema = json.loads(self.json_schema_str)
        return create_openai_response_format(name="populate_json_schema", schema=schema)

    def complete(self, content_chunks: List[List[Dict]]):
        strategy_method = {
            IterativePopulationStrategies.ONE2ONE.value: self.one2one_completion,
            IterativePopulationStrategies.ONE2MANY.value: self.one2many_completion,
            IterativePopulationStrategies.SELFSUPERVISED.value: self.selfsupervised_completion,
        }
        
        if self.population_strategy not in strategy_method:
            raise NotImplementedError(f"Strategy {self.population_strategy} is not implemented.")
        
        return strategy_method[self.population_strategy](content_chunks)

    def process_completion(self, messages, populated_dict, response_format=None, tools=None):
        try:
            prompt = generate_self_supervised_json_population_prompt(populated_dict)
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}] + messages}]
            chat_response = chat_completion_request(
                messages=messages,
                tools=tools,
                response_format = response_format,
                tool_choice={"type": "function", "function": {"name": "populate_json_schema"}},
                model=self.model,
                log_token_usage=False
            )
            if chat_response.choices[0].finish_reason == 'length':
                print('GPT finished generation with finish reason length.')
            
            tool_calls = chat_response.choices[0].message.tool_calls
            return json.loads(tool_calls[0].function.arguments)
        except Exception as e:
            print('Error during schema population iteration: ', e)
            return populated_dict

    def selfsupervised_completion(self, content_chunks):
        populated_dict = {}
        tools = create_openai_tools_arr('populate_json_schema', 'populate a json schema', json.loads(self.json_schema_str))

        for i in range(self.n_rounds):
            print('Round: ', i)
            for c_chunk in tqdm.tqdm(content_chunks, total=len(content_chunks), desc='Processing content chunks'):
                populated_dict = self.process_completion(c_chunk, populated_dict, tools=tools)
        return populated_dict

    def one2one_completion(self, content_chunks):
        populated_json_dict = {}
        tools = create_openai_tools_arr('populate_json_schema', 'populate a json schema', json.loads(self.json_schema_str))
        for c_chunk in tqdm.tqdm(content_chunks, total=len(content_chunks), desc='Processing content chunks'):
            for msg in c_chunk:
                populated_json_dict = self.process_completion([msg], populated_json_dict, tools)
        return populated_json_dict

    def one2many_completion(self, content_chunks):
        populated_json_dict = {}
        tools = create_openai_tools_arr('populate_json_schema', 'populate a json schema', json.loads(self.json_schema_str))
        for c_chunk in tqdm.tqdm(content_chunks, total=len(content_chunks), desc='Processing content chunks'):
            for msg in c_chunk:
                populated_json_dict = self.process_completion([msg], populated_json_dict, tools)
        return populated_json_dict