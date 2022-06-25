import os
import re

CLASS_NAMES = ['fire']
CLASS_REAL_NAMES = ['fire']
CLASS_NAME_DICT = {
    'CA001': 'fire'
}

dir_path = './'
pattern = re.compile('"label": "([A-Z]{2}[0-9]{3}(?:.+)?)",')
class_ids = []
for file in os.listdir(dir_path):
    if os.path.splitext(file)[-1] != '.json':
        continue
    with open(os.path.join(dir_path, file), 'r+', encoding='utf-8') as f:
        content = f.read()
        image_class_ids = pattern.findall(content)
        for id in image_class_ids:
            if id not in class_ids:
                if len(id) > 5:
                    print("Find invalid id !!")
                    content = content.replace(id, id[:5])
                    with open(os.path.join(dir_path, file), 'w', encoding='utf-8') as f:
                        f.write(content)
                else:
                    class_ids.append(id)
print('一共有{}种class'.format(len(class_ids)))
print('分别是')
index = 1
for id in class_ids:
    print('"{}",'.format(id), end="")
    index += 1
print()
index = 1
for id in class_ids:
    print('"{}":{},'.format(id, index))
    index += 1

for id in class_ids:
    print("'{}',".format(CLASS_NAME_DICT[id]),end="")



