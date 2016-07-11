
# coding: utf-8

# In[13]:

import pandas as pd
import numpy as np
import time
#filename = 'gid_vost_polygon_20150508_test.xml'
#filename = 'gid_vost_polygon_20150508.xml'
filename = 'gid_vost_polygon_20160610.xml'
start_time = time.time()


# In[14]:

from xml.etree import ElementTree

with open(filename, 'rt', encoding='utf-8-sig') as f:
    tree = ElementTree.parse(f)

print(tree)


# In[15]:

slot_id = ''
slot_st_from = ''
slot_st_to = ''
nodes = []
for node in tree.iter():    
    if node.tag == 'NTrain':
        sid = node.attrib.get('ID')
        s = node.attrib.get('ID').split()
        route = s[1].split('-')
        slot_number = int(s[0])
        slot_st_from = route[0][1:]
        slot_st_to = route[1][:-1]
        part = int(node.attrib.get('IDpart'))
    if node.tag == 'Rp':
        st_esr = int(node.attrib.get('Esr'))
        time_dep = node.attrib.get('Ot')
        time_arr = node.attrib.get('Pr')        
        nodes.append([sid, slot_number, part, slot_st_from, slot_st_to, st_esr, time_arr, time_dep])
        
df = pd.DataFrame(nodes, columns=['slot_id', 'slot_number', 'part_id', 'slot_st_from', 'slot_st_to', 
                                  'st', 'time_arr', 'time_dep'])
df.head(20)


# In[16]:

mapping = pd.read_csv('mapping_parks_gid.csv', encoding='utf-8-sig', sep=';').dropna(axis=1, how='all')
a = pd.DataFrame()
for col in list(mapping.columns.difference(['esr_st', 'nazv_st'])):
    a = pd.concat([a, mapping[[col, 'esr_st']].dropna(subset=[col]).rename(columns={col:'esr_park'})])
park_mapping = a.drop_duplicates().copy(deep=True)
park_mapping.head()


# In[17]:

df['mapping_st'] = df.st.map(park_mapping.set_index('esr_park').esr_st)
df[df.mapping_st.isnull() == False].head()


# In[18]:

st_names = pd.read_csv('resources\station.csv').drop_duplicates('esr').set_index('esr')
df['corr_st'] = df.mapping_st.fillna(df.st)

# А еще можно вот так:
# df['corr_st'] = np.where(df.mapping_st.isnull(), df.st, df.mapping_st)

df['st_name'] = df.corr_st.map(st_names.name)
df1 = df.dropna(subset=['st_name']).copy(deep=True)
df1.head(20)


# In[19]:

df1['st_next'] = df1.corr_st.shift(-1)
df1['st_next_name'] = df1.st_name.shift(-1)
df1['time_start'] = df1['time_dep']
df1['time_end'] = df1['time_arr'].shift(-1)
cols = ['slot_id', 'part_id', 'st_name', 'st_next_name', 'time_dep', 'time_arr', 'time_start', 'time_end']
df1['link'] = list(zip(df1.st_name, df1.st_next_name))
slots = df1[(df1.time_end != '') & (df1.time_start != '')]


# In[20]:

links = pd.read_csv('resources\link.csv')
stations = pd.read_csv('resources\station.csv')
st_names = stations.drop_duplicates('station').set_index('station')
links['st_from_name'] = links.st_from.map(st_names.name)
links['st_to_name'] = links.st_to.map(st_names.name)
links['link'] = list(zip(links.st_from_name, links.st_to_name))


# In[21]:

big_stations = stations[stations.norm_time != 0]
slot_number_mask = ((slots.slot_number >= 1000) & (slots.slot_number <= 3999))                    | ((slots.slot_number >= 9201) & (slots.slot_number <= 9799))
a = slots[slot_number_mask & (slots.link.isin(links.link)) & (slots.st_name.isin(big_stations.name))]        .groupby(['st_name', 'st_next_name']).slot_id.count()


# In[23]:

import sys
XLS = False
if len(sys.argv) > 1:
    if 'xls' in sys.argv[1].lower():
        XLS = True

if XLS:
    output_filename = 'slot_stats.xls'
    a.to_frame().reset_index().to_excel(output_filename)
else:
    output_filename = 'slot_stats.log'
    f = open(output_filename, 'w')
    for st in big_stations.drop_duplicates('name').name:
        try:
            b = a[st]    
            f.write('Нитки от станции %s:' % st)    
            f.write(b.to_string() + '\n\n')        
        except:
            print('Ошибка в обработке станции %s' % st)
    f.close()

print('Статистика по ниткам добавлена в файл %s' % output_filename)
print('Время выполнения: %.2f сек.' % (time.time() - start_time))


# In[ ]:

slots[(slots.st_next_name == 'ТОРЕЯ')]
#slot_id = 1371
#part_id = 1
#slots[slots.slot_id == str(slot_id)]
slots[slots.st_name == 'ШИЛКА-ТОВАРНАЯ']
sid = '990 [94314-94910]'
df[df.slot_id == sid].dropna(subset=['st_name'])


# In[ ]:

st = 'ФЕВРАЛЬСК'
#slots[slot_number_mask & (slots.st_name == st)].st_next_name.value_counts()
slots[(slots.st_next_name == st)].st_name.value_counts()

