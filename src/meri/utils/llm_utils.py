import tiktoken

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

def chat_completion_request(client, messages, tools=None, tool_choice=None, response_format=None, model="gpt-4o-mini", log_token_usage=False, temp=0.3, top_p=1.0):

    if log_token_usage:
        input_token = count_messages(messages)
        print('Estimated input token: ', input_token)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            max_tokens=4096,
            temperature=temp,
        )
        if log_token_usage:
            print('Actual Token Usage {}: {}'.format(model, *response.usage))
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e