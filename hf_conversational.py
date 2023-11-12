import os
from huggingface_hub import login

hf_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=hf_token)
os.environ['TRANSFORMERS_CACHE'] = '/share/data/2pals/fjd/.cache/huggingface'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, Conversation

from functools import reduce

def setup_pipeline(model_name, temperature=1e-7, top_p=1e-7, max_length=4096):
    # set up resources
    # free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        # load_in_4bit=True,
        load_in_8bit=True,
        # torch_dtype=torch.bfloat16, 
        use_flash_attention_2=True,
        max_memory=max_memory
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    conversational_pipeline = pipeline(
        "conversational",
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        temperature=temperature+1e-7,
        top_p=top_p+1e-7,
        max_length=max_length,
    )

    return model, tokenizer, conversational_pipeline

class HuggingfaceConversational:
    def __init__(self, model_name, temperature=1e-7, top_p=1e-7, max_length=4096):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_length = max_length
        self.model, self.tokenizer, self.conversational_pipeline = setup_pipeline(model_name, temperature, top_p, max_length)

    def __call__(self, conversation):
        chat_completion = self.conversational_pipeline(conversation)
        token_length_list = [len(self.tokenizer(message['content'])['input_ids']) for message in chat_completion]
        total_token_length = reduce(lambda x, y: x + y, token_length_list)
        return chat_completion, total_token_length
        

if __name__=="__main__":
    # check out https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212 to see what's the best format for llama2 chat model
    model_name = 'meta-llama/Llama-2-7b-chat-hf'
    # model_name = 'meta-llama/Llama-2-7b-hf' # not good for chat
    # model_name = 'meta-llama/Llama-2-13b-chat-hf'
    # model_name = 'meta-llama/Llama-2-13b-hf' # not good for chat
    # model_name = 'meta-llama/Llama-2-70b-chat-hf'
    # model_name = 'meta-llama/Llama-2-70b-hf' # not good for chat
    # model_name = 'codellama/CodeLlama-7b-Instruct-hf'
    # model_name = 'codellama/CodeLlama-13b-Instruct-hf'
    # model_name = 'codellama/CodeLlama-34b-Instruct-hf' 
    # model_name = 'codellama/CodeLlama-7b-Python-hf' # not good for chat
    # model_name = 'codellama/CodeLlama-13b-Python-hf' # not good for chat
    # model_name = 'codellama/CodeLlama-34b-Python-hf' # not good for chat
    # model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    # model_name = 'mistralai/Mistral-7B-v0.1' # not good for chat
    model = HuggingfaceConversational(model_name)
    # print(model('What is the captial of France?'))
    # print(model('hi how are you?'))
    # conversation_1 = Conversation("Going to the movies tonight - any suggestions?")
    # conversation_2 = Conversation("What's the last book you have read?")
    # answer = model(conversation_1)
    chat = [
        {"role": "system", "content": "Hello, I'm a llama. I'm an AI assistant that can help you with your work."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
        {"role": "user", "content": "I'd like to show off how chat templating works!"},
    ]
    answer, total_token_length = model(chat)
    print(answer.messages)
    print('token length: ', total_token_length)

