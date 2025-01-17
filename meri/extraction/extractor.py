from .iterative_json_completion import (
    IterativeJsonPopulator,
    IterativePopulationStrategies,
)

from meri.intermediate_format.format_handler import BasicFormatHandler

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

    def __init__(self, intermediate_format: BasicFormatHandler, chunks_max_characters=450000, chunk_overlap=1, n_rounds=1, model='gpt-4o-mini', model_temp=0.3) -> None:
        
        self.intermediate_format = intermediate_format # markdown or html
        self.model = model

        self.chunks_max_characters = chunks_max_characters
        self.chunk_overlap = chunk_overlap
        self.n_rounds = n_rounds
        self.temp = model_temp

    def populate_schema(self, json_schema_string: str):
        """Populates json file based on provided json_schema

        Args:
            json_schema_string (str): _description_
        """
        chunks = self.intermediate_format.chunk(character_threshold=self.chunks_max_characters, overlap=self.chunk_overlap)
        print(f"Number of chunks: {len(chunks)}")
        content_chunks = [self.intermediate_format.prepare_gpt_message_content(chunk) for chunk in chunks]
        print(f"Number of content chunks: {len(content_chunks)}")
        populator = IterativeJsonPopulator(json_schema_string, IterativePopulationStrategies.SELFSUPERVISED.value, n_rounds=self.n_rounds, model = self.model,
                                           temp=self.temp)
        results = populator.complete(content_chunks)

        return results