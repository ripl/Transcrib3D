import re

def exact_dict_from_text(text):
    # 使用正则表达式匹配文字中的字典
    match = re.search(r'{\s*(.*?)\s*}', text)
    
    if match:
        # 获取匹配到的字典内容
        dict_str = match.group(1)
        
        # 将字典字符串转换为实际的字典对象
        try:
            result_dict = eval('{' + dict_str + '}')
            return result_dict
        except Exception as e:
            print(f"Error converting string to dictionary: {e}")
            return None
    else:
        print("No dictionary found in the given text.")
        return None

# 示例用法
text = "This is some text with a dictionary: {'table':[9,13,20],'kitchen cabinet':[6], 'trash can':[1], 'instrument case':[6], 'tv':[7], 'refrigerator':[21], 'room':[15,22,23], 'chair':[0]}"
result = exact_dict_from_text(text)

if result:
    print("Extracted dictionary:")
    print(result)
