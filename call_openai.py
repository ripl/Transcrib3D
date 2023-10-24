import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_openai(user_prompt, model='gpt-4', temperature=0, top_p=0.1, system_message=''):
    pretext = [{"role": "system", "content": system_message}]
    user_message = [{"role": "user", "content": user_prompt}]
    completion = openai.ChatCompletion.create(
        model= model,
        messages=pretext + user_message,
        temperature=temperature,
        top_p=top_p,
    )
    assistant_response = completion.choices[0].message
    return assistant_response