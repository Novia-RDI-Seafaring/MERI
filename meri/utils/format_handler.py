from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import re

class BasicFormatHandler(ABC):

    @abstractmethod
    def add():
        pass

    @abstractmethod
    def split():
        pass

    @abstractmethod
    def chunk():
        pass

    @abstractmethod
    def prepare_gpt_message_content():
        pass
    
class MarkdownHandler(BasicFormatHandler):
    
    def __init__(self, markdown_str: str = '', seperator="\n\n") -> None:
        self.seperator = seperator
        self.markdown_str = markdown_str

    def add(self, x: str | List[str]):
        to_be_joined = [self.markdown_str, x] if isinstance(x, str) else [self.markdown_str] + x
        self.markdown_str = self.seperator.join(to_be_joined)

    @classmethod
    def find_first_base64_substring(cls, string):
        """Finds base64 string that is enclosed in "(" ")"

        Args:
            string (_type_): _description_

        Returns:
            _type_: _description_
        """
        pattern = r'\(data:[^)]+\)'

        # Use re.search to find the match
        match = re.search(pattern, string)

        if match:
            result = match.group(0)[1:]  # Remove the leading parenthesis '('
        else:
            print("No match found.")

        return result
    
    def split(self) -> List[str]:

        return self.markdown_str.split('\n\n')

    def split_add_type(self) -> List[Tuple[str, str]]:
        """splits markdown based on seperator and returns components as tuples (<type>, <markdown_string>) where
        type can be text or image.

        Returns:
            List[Tuple[str, str]]: _description_
        """
        parts = self.split()
        types = []
        for markdown_part in parts:
            if markdown_part.startswith("![Figure]"):
                t = 'image'
            else:
                t = 'text'

            types.append(t)

        return list(zip(types, parts))
    
    def chunk(self, character_threshold=1000, overlap=2) -> List[List[Tuple[str, str]]]:
        """Chunks markdown into smaller overlapping bits

        Args:
            character_threshold (int, optional): Threshold for text characters in each chunk. Images
            are not count as any characters. Defaults to 1000.
            overlap (int, optional): Overlap between chunks. Makes sure to not loose important context. Defaults to 2.
            ignore_images (bool, optional): if true, images are excluded from chun. Defaults to False.

        Returns:
            _type_: _description_
        """
        markdown_parts = self.split_add_type()
        chunks = []

        current_chunk = []
        character_counter = 0
        for (type, cont) in markdown_parts:
            if type == 'text':
                current_chunk.append((type, cont))
                character_counter += len(cont)
            elif type == 'image':
                current_chunk.append((type, cont))
            else:
                current_chunk.append((type, cont))

            if character_counter >= character_threshold:
                chunks.append(current_chunk)

                # always add previous markdown part as overlap
                current_chunk = [*current_chunk[-overlap:]] if len(current_chunk)>0 else []
                character_counter = 0

        return chunks
    
    def prepare_gpt_message_content(self, chunk: List[Tuple[str, str]], image_detail='high'):
        """_summary_

        Args:
            chunk (List[Tuple[str, str]]): List of tuple (<type>, <markdown_string>) where <type> is text or image
            image_detail (str, optional): Determines image processing in gpt. Defaults to 'high'.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: List of contents for gpt messages parameter. Elements are either for text: {"type": "text", "text": <some text>} 
            or for images {"type": "image_url", "image_url": {"url": <base64image>, "detail": <image_detail>}}
        """
        message_contents = []

        for (type, cont) in chunk:
            if type == 'text':
                if len(message_contents)==0 or message_contents[-1]['type'] != "text":
                    message_contents.append({"type": "text", "text": ''})
                message_contents[-1]["text"] = self.seperator.join([message_contents[-1]["text"], cont])
            elif type == 'image':
                message_contents.append({"type": "image_url", "image_url": {"url": self.find_first_base64_substring(cont), "detail": image_detail}})
            else:
                raise NotImplementedError
        
        return message_contents

class HTMLHandler(BasicFormatHandler):

    def __init__(self) -> None:
        super().__init__()

    def add():
        pass
    
    def split():
        pass