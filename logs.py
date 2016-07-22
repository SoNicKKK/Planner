
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


# In[21]:

lines = []
trains, locos, teams, others = [], [], [], []
with open('./input/20160719_172108/event.log') as f:
    for line in f:
        a = line[:-1].split('\t')
        if 'Train' in a[0]:
            train_id = a[0].split(' ')[1]
            route_ind = a[1]
            b = a[2].split('{')
            event_type = b[0]
            event_info = b[1][:-1]
            sp = event_info.split(', ')
            if len(sp) == 1:
                step = 'pred'
                times = sp[0]
                loco, team = -1, -1
            else:
                step = sp[0]
                times = sp[1]
                if len(sp) == 2:
                    loco, team == -1, -1
                elif len(sp) == 3:
                    loco, team = sp[2][2:], -1
                else:
                    loco, team = sp[2][2:], sp[3][2:]
            spsp = [i.split('@') for i in times.split(' ? ')]
            if len(spsp[0]) == 2:
                st_from, time_start = spsp[0][0], spsp[0][1]
            else:
                st_from, time_start = spsp[0][0], -1
            if len(spsp) == 2:
                if len(spsp[1]) == 2:
                    st_to, time_end = spsp[1][0], spsp[1][1]
                else:
                    st_to, time_end = spsp[1][0], -1
            else:
                st_to, time_end = -1, -1
                
            trains.append([train_id, route_ind, event_type, step, st_from, time_start, st_to, time_end, loco, team])
        elif 'Loco' in a[0]:
            locos.append(a)
        elif 'Team' in a[0]:
            teams.append(a)
        else:
            others.append(a)
        
print(trains[:10])


# In[24]:

df = pd.DataFrame(trains, columns = ['train', 'ind', 'event', 'step', 'st_from', 'time_start', 'st_to', 'time_end', 'loco', 'team'])


# In[28]:

df.head()


# In[47]:

lines = []
with open('input/20160719_172108/System.out.log.2') as f:
    for line in f:        
        sp = line[:-1].split(' - ')
        sp1 = sp[0].strip().find(' ')
        t = sp[0].strip()[:sp1]
        code = sp[0].strip()[sp1+1:]
        text = sp[1]
        lines.append([t, code, text])
        
df = pd.DataFrame(lines, columns=['level', 'src', 'text'])
df.head()


# In[48]:

df.level.value_counts()


# In[49]:

pd.set_option('display.max_colwidth', 100)
df[df.level == 'ERROR']


# In[57]:

df['fail_init'] = df.text.apply(lambda x: x.split()[3] if 'инициализировать' in x else -1)
df[df.fail_init != -1].fail_init.value_counts()


# In[60]:

df['fail_loco_reg'] = df.text.apply(lambda x: x.split()[1] if 'отнесён' in x else -1)
df[df.fail_loco_reg != -1].head()


# In[62]:

df[df.text.apply(lambda x: '200021105168' in x)]

