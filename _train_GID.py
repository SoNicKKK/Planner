
# coding: utf-8

# In[117]:

def nice_time(t):
    return time.strftime(time_format, time.localtime(t)) if t > 0 else ''


# In[127]:

print('Excel-файлы с данными из ГИДа должны находиться в папке ./resources/others\n')


# In[118]:

import os
import time
import sys
import traceback

time_format = '%b %d, %H:%M'

files = [files for root, directories, files in os.walk('./resources/others')][0]
even = {}
odd = {}
os.chdir('./resources/others')
try:
    for f in files:        
        if ('чет' in f) & ('отправление' in f) & ('нечет' not in f): even[f] = int(os.path.getmtime(f))
        if ('нечет' in f) & ('отправление' in f): odd[f] = int(os.path.getmtime(f))
    if even != {}:
        even_filename = max(even, key=lambda k: even[k])
        even_date_modified = even[even_filename]
        print('Данные об отправлении четных поездов из Иркутска взяты из файла "%s" (дата изменения %s)'     
              % (even_filename, nice_time(even_date_modified)))
    else: 
        even_filename == None
        print('Не удалось загрузить данные по отправлению четных поездов')
    if odd != {}:
        odd_filename = max(odd, key=lambda k: odd[k])
        odd_date_modified = odd[odd_filename]
        print('Данные об отправлении нечетных поездов из Иркутска взяты из файла "%s" (дата изменения %s)'     
              % (odd_filename, nice_time(odd_date_modified)))
    else: 
        odd_filename == None
        print('Не удалось загрузить данные по отправлению нечетных поездов')
    os.chdir('..')
    os.chdir('..')
    
except Exception: 
    os.chdir('..')
    os.chdir('..')
    print('Ошибка: не удалось загрузить данные по отправлению поездов')    
    traceback.print_exc(file=sys.stdout)
    
print('\n')


# In[124]:

from subprocess import Popen, PIPE
def create_report(filename):
    print('Построение отчета для поездов из файла', filename)
    cmd = 'python _sub_train_GID.py %s' % ('./resources/others/' + filename)
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    print(stdout.decode('ascii', 'ignore'))


# In[126]:

create_report(even_filename)
create_report(odd_filename)
print('Отчеты созданы')

