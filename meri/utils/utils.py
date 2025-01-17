import fitz
from PIL import Image
import json
from litellm import completion, litellm
import base64
import io

def scale_coords(source_coords, source_height, source_width, target_height, target_width):
    '''Transforms source coordinates (x0, y0, x1, y1)
    to target coordinates (x0,y0, x1,y1)'''

    x0, y0, x1, y1 = source_coords

    x0_rel = x0/source_width
    x1_rel = x1/source_width

    y0_rel = y0/source_height
    y1_rel = y1/source_height

    #rect_shape = [int(x0_rel*target_width+0.5),int(y0_rel*target_height+0.5), int(x1_rel*target_width+0.5), int(y1_rel*target_height+0.5)]
    rect_shape = [int(x0_rel*target_width),int(y0_rel*target_height), int(x1_rel*target_width), int(y1_rel*target_height)]

    return rect_shape

def pdf_to_im(page, cropbbox=None):
    """ Converts pdf to image and if provided crops image by cropbox
    """

    pix = page.get_pixmap()
    pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    if cropbbox is None:
        return pil_image
    cropped_im = pil_image.crop(cropbbox)
    return cropped_im 

def pil_to_base64(pil_image: Image, raw=True):
    """ Converts PIL to base64 string
    """
    # Convert PIL Image to bytes
    with io.BytesIO() as buffer:
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

    # Convert bytes to base64 string
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    if raw:
        return base64_image
    else: 
        return f'data:image/png;base64,{base64_image}'
    
def load_json(json_path):
    with open(json_path) as file:
        json_content = json.load(file)

    return json_content


""" 

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


def chat_completion_request(messages, tools=None, tool_choice=None, response_format=None, model="gpt-4o-mini", log_token_usage=False, temp=0.3, top_p=1.0):

    try:
        response = completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            max_tokens=4096,
            temperature=temp,
        )

        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
"""