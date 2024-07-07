import os, datetime, sys
from io import StringIO
from contextlib import redirect_stdout
import traceback
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

from core.gpt_dialogue import Dialogue

class CodeInterpreter(Dialogue):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call_llm_with_code_interpreter(self, user_prompt, namespace_for_exec={}, token_usage_total=0):
        """
        Call LLM with user prompt, using the code interpreter mechanism. 
        If the LLM response contains python code snippets, they will be executed locally, and the conversation will continue with the execution result appended.
        If an error occurs during code execution, the error message will be appended instead.
        If the LLM response does not contain code, the function will return. Note that only the last response content will be returned, and the entire dialogue is stored in self.pretext variable.

        Parameters:
            user_prompt (str): The user prompt.
            namespace_for_exec (dict): Namespace used in the exec() function. This enables LLM to use variables or functions it defined in previous rounds of conversation.
            token_usage_total (int): Token usage in total in current non-recursive call of this function.

        Returns:
            str: The response content.
            int: The token usage in this round of conversation.
        """
        assistant_response, token_usage = self.call_llm(user_prompt)
        token_usage_total += token_usage

        if self.debug:
            print('response_content: ', assistant_response)

        # check if the response contains code snippet
        response_splits = assistant_response.split('```python')
        if len(response_splits) <= 1:
            # if the LLM response contains no code, the function will return.
            if self.debug:
                print('no code snippet found, return the raw response')
            return assistant_response,token_usage_total
        else:
            # if code snippet is found, then execute the code
            code_snippet=""
            for split in response_splits:
                if '```' in split:
                    code_snippet+=split.split('```')[0]
            f = StringIO()
            code_exec_success=True
            
            with redirect_stdout(f):
                try:
                    exec(code_snippet,namespace_for_exec)
                    code_exe_result = f.getvalue()
                except Exception as e:
                    code_exec_success=False
                    traceback_message_lines=traceback.format_exc().splitlines()
                    code_exe_result = '\n'.join(traceback_message_lines[-4:])

            if code_exec_success:
                code_exe_msg='code execution result:\n' + str(code_exe_result)
            else:
                code_exe_msg = "An error was raised when executing the code you write: %s"%code_exe_result
            print(code_exe_msg)
            return self.call_llm_with_code_interpreter(code_exe_msg, namespace_for_exec, token_usage_total)
        
if __name__ == '__main__':

    config = {
        'model': 'gpt-4',
        # 'model': 'gpt-3.5-turbo',
        'temperature': 0,
        'top_p': 0.0,
        'max_tokens': 'inf',
        'system_message': "Imagine you are an artificial intelligence assitant with a python interpreter. So when answering questions, you can choose to generate python code (for example, when there is need to do quantitative evaluation). The generated code should always print out the result. The code should be written in python and should be able to run in the python environment with the following packages installed: numpy, math. The generated code should be complete and always include proper imports. Each generated code piece should be independent and NOT rely on previous generated code. When answer step by step, stop whenever you feel there is need to generate python code (for example, where there is need to do quantitative evaluation) and wait for the result from the code execution. When the answewr is complete, add 'Now the answer is complete.' to the end of your answer.",

        # 'load_path': '',
        'save_path': 'chats',
        'debug': False
    }

    dialogue = CodeInterpreter(**config)
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
            dialogue = CodeInterpreter(**config)
            print('====GPT Dialogue Initialized, start asking your questions====')
            continue
        elif user_prompt == 'pretext':
            print('===Pretext===')
            for message in dialogue.get_pretext():
                print(message)
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
            # response = dialogue.call_openai(user_prompt)['content']
            response = dialogue.call_llm_with_code_interpreter(user_prompt)['content']
            print('Bot:', response)
            counter = 0
            while not response.endswith('Now the answer is complete.') and counter < 10:
                response = dialogue.call_llm_with_code_interpreter('')['content']
                print('Bot:', response)
                counter += 1
