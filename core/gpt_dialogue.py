import os
import json
import datetime
from copy import deepcopy
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
from openai import OpenAI
client = OpenAI()

# HUGGINGFACE_MODELS = {
#     'meta-llama/Llama-2-7b-chat-hf',
#     'meta-llama/Llama-2-13b-chat-hf',
#     'meta-llama/Llama-2-70b-chat-hf',
#     'codellama/CodeLlama-7b-Instruct-hf',
#     'codellama/CodeLlama-13b-Instruct-hf',
#     'codellama/CodeLlama-34b-Instruct-hf',
#     'mistralai/Mistral-7B-Instruct-v0.1',
# }


class Dialogue:
    def __init__(self, model='gpt-4', temperature=0, top_p=0.0, max_tokens=10, system_message='', load_path=None, save_path='chats', debug=False):
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.system_message = system_message
        self.save_path = save_path
        self.debug = debug
        if load_path is not None:
            self.load_pretext(load_path)
        else:
            self.pretext = [{"role": "system", "content": self.system_message}]

        if 'llama' in self.model:
            from hf_conversational import HuggingfaceConversational
            from transformers import Conversation
            self.conversational = HuggingfaceConversational(
                model_name=self.model,
                temperature=self.temperature,
                top_p=self.top_p,
                max_length=self.max_tokens
            )

    def load_pretext(self, load_path):

        def load_json(load_path):
            with open(load_path) as json_file:
                return json.load(json_file)
            
        self.pretext = []
        if isinstance(load_path, list):
            for path in load_path:
                self.pretext += load_json(path)
        elif isinstance(load_path, str):
            self.pretext = load_json(load_path)
        else:
            raise Exception('load_path must be a list of strings or a string')

    def get_pretext(self):
        return self.pretext

    # def save_pretext(self, save_path, timestamp):
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     json_path = os.path.join(save_path, 'dialogue_' + timestamp + '.json')
    #     json_object = json.dumps(self.get_pretext(), indent=4)
    #     with open(json_path, 'w') as f:
    #         f.write(json_object)

    def save_pretext(self, save_folder_path, file_name):
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)
        json_path = os.path.join(save_folder_path, file_name)
        json_object = json.dumps(self.get_pretext(), indent=4)
        with open(json_path, 'w') as f:
            f.write(json_object)
    
    def print_pretext(self, print_system_and_user_first_prompt=True, to_print_out=True):
        # determine whether to print system message and user's first prompt
        pretext=deepcopy(self.pretext)
        if not print_system_and_user_first_prompt:
            pretext=pretext[2:]
        printed_pretext=''
        # print pretext
        for piece in pretext:
            if to_print_out:
                print('----------------->\tROLE: '+piece['role']+'\t<-----------------')
                print('CONTENT: '+piece['content'])
            printed_pretext=printed_pretext+'----------------->\tROLE: '+piece['role']+'\t<-----------------\n'
            printed_pretext=printed_pretext+'CONTENT: '+piece['content']+'\n'
        self.printed_pretext=printed_pretext

    def call_llm(self, user_prompt):
        """
        Call LLM with user prompt, get the response, append the user prompt and the response to pretext(dialogue history), and return the current response and token usage.

        Parameters:
            user_prompt (str): The user prompt.

        Returns:
            str: The response content.
            int: The token usage in this round of conversation.
        """
        user_message = [{"role": "user", "content": user_prompt}]
        messages = self.pretext + user_message
        # print('messages: ', messages)
        if 'gpt' in self.model:
            completion = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                seed=42,
            )
            raw_response_message = completion.choices[0].message
            assistant_response_message = {'role': raw_response_message.role, 'content': raw_response_message.content}
            # print('assistant_response_message: ', assistant_response_message)
            token_usage = completion.usage.total_tokens
        elif 'llama' in self.model:
            chat_completion_messages,token_usage = self.conversational(messages)
            assistant_response_message = chat_completion_messages.messages[-1]
        else:
            raise Exception('model name {} not supported'.format(self.model))
        
        self.pretext = self.pretext + user_message + [assistant_response_message]
        
        return assistant_response_message['content'], token_usage


if __name__ == '__main__':

    config = {
        # 'model': 'gpt-4-1106-preview',
        'model': 'gpt-4',
        # 'model': 'gpt-3.5-turbo',
        # 'model': 'meta-llama/Llama-2-7b-chat-hf',
        'temperature': 0,
        'top_p': 0.0,
        'max_tokens': 8192,
        'system_message': '',
        # 'load_path': 'chats/dialogue_an apple.json',
        'save_path': 'chats',
        'debug': False
    }

    dialogue = Dialogue(**config)
    print('======================Instructions======================')
    print('Type "exit" to exit the dialogue')
    print('Type "reset" to reset the dialogue')
    print('Type "pretext" to see the current dialogue history')
    print('Type "config" to see the current config')
    print('Type "save" to save the current dialogue history')
    print('====GPT Dialogue Initialized, start asking your questions====')

    while True:
        user_prompt = input('You: ')
        if user_prompt == 'exit':
            break
        elif user_prompt == 'reset':
            dialogue = Dialogue(**config)
            print('====GPT Dialogue Initialized, start asking your questions====')
            continue
        elif user_prompt == 'pretext':
            print('===Pretext===')
            for message in dialogue.get_pretext():
                print(message)
            # dialogue.print_pretext()
            print('===Pretext===')
            continue
        elif user_prompt == 'config':
            print('===Config===')
            print(config)
            print('===Config===')
            continue
        elif user_prompt == 'save':
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            dialogue.save_pretext(config['save_path'], timestamp)
            print('Pretext saved to', os.path.join(
                config['save_path'], 'dialogue_' + timestamp + '.json'))
            continue
        else:
            assistant_response_message, token_usage = dialogue.call_llm(user_prompt)
            # response = assistant_response_message['content']
            # print('Bot:', response)
            print('Bot:', assistant_response_message)
