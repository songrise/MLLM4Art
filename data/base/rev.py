import json

# Read the JSON file
with open('/root/autodl-tmp/MLLM4Art/data/style_1k.json', 'r') as f:
    data = json.load(f)

# Modify image paths in place
# for i, item in enumerate(data, start=1):
#     item['reference_image'] = f'./style_reference/{i}.jpg'

# Modify reference_image paths in place
# for key in data:
#     data[key]['reference_image'] = f'./style_reference/{key}.jpg'

# Write back to the same file
with open('/root/autodl-tmp/MLLM4Art/data/style_1k.json', 'w') as f:
    json.dump(data, f, indent=4)

print("Done")