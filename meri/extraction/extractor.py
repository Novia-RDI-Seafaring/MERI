import json
import os

import openai
from .iterative_json_completion import (
    IterativeJsonPopulator,
    IterativePopulationStrategies,
)

from meri.utils.format_handler import BasicFormatHandler, MarkdownHandler
from meri.utils.llm_utils import chat_completion_request


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

    def __init__(self, intermediate_format: BasicFormatHandler, chunks_max_characters=1000, chunk_overlap=2,  n_rounds=1, model='gpt-4o-mini', api_key: str = None) -> None:
        
        self.intermediate_format = intermediate_format # markdown or html
        # if api_key is None:
        #     api_key = os.getenv("OPENAI_API_KEY")
        
        # self.client = openai.Client(api_key=api_key)
        self.client = openai.Client(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model

        self.chunks_max_characters = chunks_max_characters
        self.chunk_overlap = chunk_overlap
        self.n_rounds = n_rounds

    def populate_schema(self, json_schema_string: str):
        """Populates json file based on provided json_schema

        Args:
            json_schema_string (str): _description_
        """
        chunks = self.intermediate_format.chunk(character_threshold=self.chunks_max_characters, overlap=self.chunk_overlap)
        content_chunks = [self.intermediate_format.prepare_gpt_message_content(chunk) for chunk in chunks]
        
        populator = IterativeJsonPopulator(json_schema_string, IterativePopulationStrategies.SELFSUPERVISED.value, n_rounds=self.n_rounds, model = 'gpt-4o')
        results = populator.complete(content_chunks)

        return results