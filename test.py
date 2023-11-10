# Here's a simple Python code snippet that reads a JSON file.
import json

# JSON file path
json_file_path = 'object_filter_pretext.json'

# Read JSON file
with open(json_file_path) as file:
    data = json.load(file)

# Output the data to check
print(data)
for d in data:
    print(d['role'])
    print(d['content'])
