
# coding: utf-8

# In[29]:

import os
files = [files for root, directories, files in os.walk('.')][0]
files = [file for file in files if ('.py' in file) & (file != 'release.py')]
print(files)
for filename in files:
    lines = []    
    with open(filename, encoding='utf-8-sig') as f:
        for line in f:        
            if ('magic' in line) & ('#' not in line):
                to_app = '#' + line
            elif 'PRINT = True' in line:
                to_app = line.replace('True', 'False')
            else:
                to_app = line
            lines.append(to_app)

    with open(filename, 'w', encoding='utf-8-sig') as fw:
        for line in lines:
            fw.write(line)
        fw.close()

