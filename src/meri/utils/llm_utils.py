import tiktoken
from litellm import completion, litellm
import os

#litellm.set_verbose=True

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def count_messages(messages, encoding_name='o200k_base'):
    """Only count tokens for type text i.e. images are not count

    Args:
        messages (_type_): message array of form [
                                                    {
                                                    "role": "user",
                                                    "content": [{"type": "text", "text": prompt}, ...],
                                                    }
                                                ]
        encoding_name (str, optional): _description_. Defaults to 'o200k_base'.

    Returns:
        _type_: _description_
    """
    count = 0
    for m in messages:
        for m_c in m['content']:
            if m_c['type'] == 'text':
                count += num_tokens_from_string(m_c['text'], encoding_name)
    
    return count

def get_litellm_baseurl(model: str):
    """ Returns base URL for model. If model is a gpt model, base url will be none. Else 
    the model will have the form e.g. "ollama/llava:3", then the base_url will be loaded
    from .env file (e.g. LITELLM_BASE_URL_ollama)
    """

    model_comps = model.split("/")
    if len(model_comps) == 1:
        return None
    elif len(model_comps) == 2:
        return os.environ.get(f"LITELLM_BASE_URL_{model_comps[0]}")
    else:
        raise Exception


def chat_completion_request(messages, tools=None, tool_choice=None, response_format=None, model="gpt-4o-mini", log_token_usage=False, temp=0.3, top_p=1.0):

    base_url = get_litellm_baseurl(model)
    print("USING BASE URL: ", base_url)
    if log_token_usage:
        input_token = count_messages(messages)
        print('Estimated input token: ', input_token)

    try:
        response = completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            max_tokens=4096,
            temperature=temp,
            base_url=base_url,

        )

        if log_token_usage:
            print('Actual Token Usage {}: {}'.format(model, *response.usage))
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e