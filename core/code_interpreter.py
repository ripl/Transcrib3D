import os, datetime, sys
from io import StringIO
from contextlib import redirect_stdout
import traceback
import openai
from gpt_dialogue import Dialogue
openai.api_key = os.getenv("OPENAI_API_KEY")

class CodeInterpreter(Dialogue):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call_openai_with_code_interpreter(self, user_prompt,namespace_for_exec={},token_usage_total=0):
        # 如果gpt回复的内容包含python代码，则把代码的执行结果发送给gpt，继续等待其回复
        # 如果gpt回复的内容不包含python代码，则此函数返回全部结果
        # 每次递归统计使用的token数，最终返回总的token数
        assistant_response,token_usage = self.call_openai(user_prompt)
        token_usage_total+=token_usage

        # check if response contain code snippet
        response_content = assistant_response['content']
        if self.debug:
            print('response_content: ', response_content)
        response_splits = response_content.split('```python')
        if len(response_splits) <= 1:
            # no code snippet found, return the raw response
            if self.debug:
                print('no code snippet found, return the raw response')
            return assistant_response,token_usage_total
        else:
            # code snippet found, execute the code
            # code_snippet = response_splits[-1].split('```')[0]
            # print('code snippet: ', code_snippet)
            code_snippet=""
            for split in response_splits:
                if '```' in split:
                    code_snippet+=split.split('```')[0]
            f = StringIO()
            # sys.stdout = f
            code_exec_success=True
            
            with redirect_stdout(f):
                try:
                    exec(code_snippet,namespace_for_exec)
                    code_exe_result = f.getvalue()
                except Exception as e:
                    code_exec_success=False
                    traceback_message_lines=traceback.format_exc().splitlines()
                    code_exe_result = '\n'.join(traceback_message_lines[-4:])
            # code_exe_result = f.getvalue()
            # f.close()
            # sys.stdout = sys.__stdout__
            
            #############利用保存文件的方式####################
            # # 将代码片段保存到 code_snippet.py 文件
            # with open("code_snippet.py", "w") as file:
            #     file.write(code_snippet)

            # # 执行 code_snippet.py 并将输出重定向到临时文件
            # os.system("python code_snippet.py > output.txt")

            # # 从临时文件中读取结果
            # with open("output.txt", "r") as file:
            #     code_exe_result = file.read()
            ##################################################
            if code_exec_success:
                code_exe_msg='code execution result:\n' + str(code_exe_result)
            else:
                code_exe_msg = "An error was raised when executing the code you write: %s"%code_exe_result
            # code_exe_msg = 'Execution result of the above code is: ' + str(code_exe_result)
            print(code_exe_msg)
            return self.call_openai_with_code_interpreter(code_exe_msg,namespace_for_exec,token_usage_total)
        
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
            response = dialogue.call_openai_with_code_interpreter(user_prompt)['content']
            print('Bot:', response)
            counter = 0
            while not response.endswith('Now the answer is complete.') and counter < 10:
                response = dialogue.call_openai_with_code_interpreter('')['content']
                print('Bot:', response)
                counter += 1
