import re

def extract_int_lists(text):
    # 匹配方括号内的内容
    pattern = r'\[([^\[\]]+)\]'
    matches = re.findall(pattern, text)
    
    int_lists = []
    
    for match in matches:
        elements = match.split(',')
        int_list = []
        
        for element in elements:
            element = element.strip()
            try:
                int_value = int(element)
                int_list.append(int_value)
            except ValueError:
                pass
        
        if len(int_list) == len(elements):
            int_lists = int_lists + int_list
    
    return int_lists

input_text = "一些文字 [1, 2, 3] 另一些文字 [4, 5, 6] 其他一些文字 [7, 8, id]"
result = extract_int_lists(input_text)
print(result)
