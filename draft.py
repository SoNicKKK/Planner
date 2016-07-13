
# coding: utf-8

# In[179]:

FOLDER = 'resources/'

import numpy as np
import pandas as pd
import time, datetime
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

get_ipython().magic('matplotlib inline')
plt.style.use('fivethirtyeight')
plt.rc('font', family='Times New Roman')

pd.set_option('max_rows', 50)

time_format = '%b %d, %H:%M'

start_time = time.time()
current_time = pd.read_csv(FOLDER + 'current_time.csv').current_time[0]
twr          = pd.read_csv(FOLDER + 'team_work_region.csv', converters={'twr':str})
links        = pd.read_csv(FOLDER + 'link.csv')
stations     = pd.read_csv(FOLDER + 'station.csv', converters={'station':str})
train_info   = pd.read_csv(FOLDER + 'train_info.csv', converters={'train': str, 'st_from':str, 'st_to':str, 'oper_location':str,
                                                                 'st_from':str, 'st_to':str})
train_plan   = pd.read_csv(FOLDER + 'slot_train.csv', converters={'train': str, 'st_from':str, 'st_to':str})
loco_info    = pd.read_csv(FOLDER + 'loco_attributes.csv', converters={'train':str, 'loco':str, 'depot':str,
                                                                      'st_from':str, 'st_to':str})
loco_plan    = pd.read_csv(FOLDER + 'slot_loco.csv', converters={'train':str, 'loco':str, 'st_from':str, 'st_to':str})
team_info    = pd.read_csv(FOLDER + 'team_attributes.csv', converters={'team':str,'depot':str, 'oper_location':str,                                                                  'st_from':str, 'st_to':str, 'loco':str, 'depot_st':str})
team_plan    = pd.read_csv(FOLDER + 'slot_team.csv', converters={'team':str,'loco':str, 'st_from':str, 'st_to':str})
loco_series  = pd.read_csv(FOLDER + 'loco_series.csv')

team_info.regions = team_info.regions.apply(literal_eval)
st_names = stations[['station', 'name', 'esr']].drop_duplicates().set_index('station')
print('Planning start time: %s (%d)' % (time.strftime(time_format, time.localtime(current_time)), current_time))


# In[180]:

# Мержим таблицы _plan и _info для поездов, локомотивов и бригад
# Добавляем во все таблицы названия станций на маршруте и времена отправления/прибытия в читабельном формате

def add_info(df):    
    if 'st_from' in df.columns:
        df['st_from_name'] = df.st_from.map(st_names.name)
    if 'st_to' in df.columns:
        df['st_to_name'] = df.st_to.map(st_names.name)
    if 'time_start' in df.columns:
        df['time_start_norm'] = df.time_start.apply(lambda x: time.strftime(time_format, time.localtime(x)))
    if 'time_end' in df.columns:
        df['time_end_norm'] = df.time_end.apply(lambda x: time.strftime(time_format, time.localtime(x)))
    if 'oper_location' in df.columns:
        df['oper_location_name'] = df.oper_location.map(st_names.name)    
        df.oper_location_name.fillna(0, inplace=True)
    if ('oper_location' in df.columns) & ('st_from' in df.columns) & ('st_to' in df.columns):        
        df['loc_name'] = df.oper_location_name
        df.loc[df.loc_name == 0, 'loc_name'] = df.st_from_name + ' - ' + df.st_to_name
    
add_info(train_plan)
add_info(loco_plan)
add_info(team_plan)
add_info(loco_info)
add_info(team_info)
add_info(train_info)
train_plan = train_plan.merge(train_info, on='train', suffixes=('', '_info'), how='left')
loco_plan = loco_plan.merge(loco_info, on='loco', suffixes=('', '_info'), how='left')
team_plan = team_plan.merge(team_info, on='team', suffixes=('', '_info'), how='left')
team_plan['team_type'] = team_plan.team.apply(lambda x: 'Реальная' if str(x)[0] == '2' else 'Фейковая')


# In[252]:

def nice_time(t):
    #if not time_format: time_format = '%b %d, %H:%M'
    return time.strftime(time_format, time.localtime(t)) if t > 0 else ''

def nice_print(s, **kwargs):    
    num = kwargs['num'] if 'num' in kwargs.keys() else False
    cols = kwargs['cols'] if 'cols' in kwargs.keys() else s.columns
    if num:
        print(s.reset_index()[cols].to_string())
    else:
        print(s[cols].to_string(index=False))


# In[182]:

train_plan['train_type'] = train_plan.train.apply(lambda x: int(str(x)[0]))
train_plan.loc[train_plan.train_type == 9, 'weight'] = 3500


# In[183]:

cols = ['train', 'number', 'weight', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'loco', 'ser_name', 'sections']
train_plan['train_time'] = list(zip(train_plan.train, train_plan.time_start))
loco_plan['train_time'] = list(zip(loco_plan.train, loco_plan.time_start))
train_plan['loco'] = train_plan.train_time.map(loco_plan.drop_duplicates('train_time').set_index('train_time').loco)
loco_info['ser_name'] = loco_info.series.map(loco_series.set_index('ser_id').ser_name)
train_plan['ser_name'] = train_plan.loco.map(loco_info.set_index('loco').ser_name)
train_plan['sections'] = train_plan.loco.map(loco_info.set_index('loco').sections)
a = train_plan[(train_plan.weight > 6000) & (train_plan.weight < 6300)].dropna(subset=['loco'])[cols].drop_duplicates(['train', 'loco'])
a.groupby(['ser_name', 'sections']).loco.count()


# In[184]:

print(nice_time(current_time))
sec4 = train_plan[(train_plan.sections == 3)].drop_duplicates(['train', 'loco'])
sec4[(sec4.weight < 3500) & (sec4.time_start >= current_time)].sort_values('weight', ascending=False)[cols]


# In[185]:

train_plan[train_plan.train == '210256460752'][cols]


# In[186]:

loco_plan[loco_plan.loco == '200200062937'][['loco', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'train', 'state']]


# In[187]:

loco_info[loco_info.tts < 86400].sort_values('tts')[['loco', 'number', 'tts', 'ser_name', 'sections']]


# In[188]:

link = pd.read_csv(FOLDER + 'link.csv', dtype={'st_from':str, 'st_to':str})
link['link'] = list(zip(link.st_from, link.st_to))


# In[189]:

slot = pd.read_csv(FOLDER + 'slot.csv', dtype={'st_from':str, 'st_to':str})
#add_info(slot)
slot['link'] = list(zip(slot.st_from, slot.st_to))
slot['time'] = slot.time_end - slot.time_start
link['slot_tt'] = link.link.map(slot.groupby('link').time.mean())
link.slot_tt.fillna(link.time, inplace=True)
link.slot_tt = link.slot_tt.apply(int)
add_info(link)


# In[190]:

import networkx as nx

all_stations = pd.Series(np.concatenate([link.st_from_name.unique(), link.st_to_name.unique()])).drop_duplicates().values
g = nx.DiGraph()
g.add_nodes_from(all_stations)
g.add_weighted_edges_from(list(zip(link.st_from_name, link.st_to_name, link.slot_tt))) # names
#g.add_weighted_edges_from(list(zip(cost.st_from_name, cost.st_to_name, cost.cost))) # id


# In[191]:

st_from, st_to = 'МАРИИНСК', 'ИРКУТСК-СОРТИРОВОЧНЫЙ'
st_to1 = 'БОГОТОЛ'
path = nx.dijkstra_path(g, st_from, st_to)
length = nx.dijkstra_path_length(g, st_from, st_to)
print(length, path)


# In[192]:

all_paths = nx.all_pairs_dijkstra_path(g)
all_lengths = nx.all_pairs_dijkstra_path_length(g)
print(len(all_paths))


# In[193]:

all_lengths[st_from][st_to1]


# In[194]:

'''
    Examples:
    al = pd.read_csv(FOLDER + '/mandatory/travel_times_all_pairs.csv', sep=';')
    get_longest_pair(['МАРИИНСК', 'ИЛАНСКАЯ', 'ИРКУТСК-СОРТИРОВОЧНЫЙ', 'РЕШОТЫ'], al.set_index(['st_from_name', 'st_to_name']))
    
    => Out[460]: ('МАРИИНСК', 'ИРКУТСК-СОРТИРОВОЧНЫЙ', 94170)
    
    get_longest_pair(['МАРИИНСК', 'ИЛАНСКАЯ', 'РЕШОТЫ', 'ИРКУТСК-СОРТИРОВОЧНЫЙ'], nx.all_pairs_dijkstra_path_length(g))
    
    => Out[460]: ('МАРИИНСК', 'ИРКУТСК-СОРТИРОВОЧНЫЙ', 94170)    
'''

def get_longest_pair(st_list, lengths):
    sm1, sm2 = '', ''
    m = 0
    for s1 in st_list:
        for s2 in st_list:
            if type(lengths) == dict:
                l = lengths[s1][s2]
            elif type(lengths) == pd.DataFrame:
                l = lengths.ix[s1, s2].values[0]
            else: l = 0
            if l > m:
                m = l
                sm1, sm2 = s1, s2
    return (sm1, sm2, m)


# In[195]:

#stations.groupby('loco_region')['name'].unique()
d = dict(list(stations.groupby('loco_region')['name']))
res = []
for key in d.keys():
    sts = d[key]
    m = 0
    sm1, sm2 = '', ''
    for s1 in sts:
        for s2 in sts:
            m1 = all_lengths[s1][s2]
            if m1 > m: 
                m = m1
                sm1, sm2 = s1, s2
    #print(key, sm1, sm2, np.round(m / 3600, 2))
    res.append([key, sm1, sm2, np.round(m / 3600, 2)])
    
reg_lens = pd.DataFrame(res, columns = ['region', 'st_from', 'st_to', 'max_tt']).sort_values('max_tt', ascending=False)


# In[196]:

regions_stoplist = [2002119322, 2002119323, 2002119314, 2002119316, 2002119297]
reg_lens = reg_lens[reg_lens.region.isin(regions_stoplist) == False]
loco_reg_names = pd.read_csv(FOLDER + 'mandatory/loco_reg_names.csv')
reg_lens['reg_name'] = reg_lens.region.map(loco_reg_names.set_index('region').reg_name_str)
reg_lens


# In[197]:

res_start = train_plan[train_plan.train_type == 8].drop_duplicates('train')
res_end = train_plan[train_plan.train_type == 8].drop_duplicates('train', keep='last')
res = res_start[['train', 'st_from_name', 'time_start', 'time_start_norm']].set_index('train').join(res_end[['train', 'st_to_name', 'time_end', 'time_end_norm']].set_index('train'))
res['tt'] = res.time_end - res.time_start
sns.set(style='whitegrid', context='notebook')
#sns.kdeplot(res.tt / 3600, shade=True)
#l = len(res.tt.index)


# In[198]:

res['slot_tt'] = res.apply(lambda row: all_lengths[row.st_from_name][row.st_to_name], axis=1)
res['above'] = res.tt / res.slot_tt - 1
a = res[res.slot_tt > 16 * 3600]
a.sort_values('slot_tt', ascending=False)[['st_from_name', 'st_to_name', 'slot_tt', 'above']].drop_duplicates(subset=['st_from_name', 'st_to_name'])


# In[200]:

serv = pd.read_csv(FOLDER + 'service_station.csv', dtype={'station':str})
serv['st_name'] = serv.station.map(st_names.name)
serv_ac = serv[serv.ptype == 'ac']
serv_ac[serv_ac.st_name == 'МАГДАГАЧИ'].duration.mean()
serv[serv.ptype.isin(['ac', 'diesel'])].groupby(['ptype', 'priority', 'st_name']).duration.mean().apply(lambda x: np.round(x))


# In[201]:

loco_plan['tt'] = loco_plan.time_end - loco_plan.time_start
loco_plan[loco_plan.state == 4].groupby('st_from_name').loco.count()


# In[202]:

loco_plan['ser_name'] = loco_plan.series.map(loco_series.set_index('ser_id').ser_name)
cols = ['loco', 'series', 'sections', 'ser_name', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'state', 'train']
loco_plan[(loco_plan.state == 4) & (loco_plan.st_from_name == 'МАГДАГАЧИ')][cols].head()


# In[203]:

cols = ['loco', 'ser_name', 'sections', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'state', 'train']
with pd.option_context('display.max_colwidth', 15):
    nice_print(loco_plan[loco_plan.loco == '200200036395'], cols, num=False)


# In[204]:

def get_reg_name(l):
    l_big = [st for st in l if st in big_st]
    if len(l_big) == 2:
        ret = l_big
    elif len(l_big) > 2:
        st1, st2, length = get_longest_pair(l_big, all_lengths)
        ret = [st1, st2]
    else:
        st1, st2, length = get_longest_pair(l, all_lengths)
        ret = [st1, st2]
    return ret[0] + ' - ' + ret[1]        

team_region = pd.read_csv(FOLDER + 'team_region.csv', dtype={'st_from':str, 'st_to':str, 'depot':str})
add_info(team_region)
big_st = stations[stations.norm_time > 0].name.unique()
team_region['depot_name'] = team_region.depot.map(st_names.name)
team_region['reg_name'] = team_region.team_region                            .map(team_region.groupby('team_region').st_from_name.unique().apply(get_reg_name))

cols_tracks = ['team_region', 'asoup', 'depot', 'depot_name', 'st_from_name', 'st_to_name', 'reg_name']
cols_times = ['team_region', 'asoup', 'depot', 'depot_name', 'time_f', 'time_b', 'time_wr']


# In[273]:

# добавим локомотивам атрибут power_type

def get_power_type(x):
    if x == '-1':
        return -1
    elif 'ТЭ' in x:
        return 'diesel'
    elif ('ВЛ8' in x) | ('ЭС5К' in x):
        return 'ac'
    else:
        return 'dc'

def get_mess(row):
    regs = literal_eval(row.regions)
    regions = ''
    for r in regs:
        regions += 'id(%s),' % r
    if len(regions) > 0: regions=regions[:-1]
    s = '+loco_attributes(id(%s),attributes([series(%s),loco_regions([%s]),depot(station(%s)),sections(%d),type(%d),power_type(%s)]))'         % (row.loco, row.series, regions, row.depot, row.sections, row.ltype, row.power_type)
    return s
    
loco_info.ser_name.fillna('-1', inplace=True)
loco_info['power_type'] = loco_info.ser_name.apply(get_power_type)
loco_info[['loco', 'ser_name', 'power_type']]
a = loco_info[loco_info.ltype == 1].groupby('power_type').ser_name.unique()
for t in a.index:
    print(t, sorted(a[t]))
    
loco_info['message'] = loco_info.apply(get_mess, axis=1)
cols = ['loco', 'series', 'regions', 'depot', 'sections', 'ltype', 'power_type', 'message']
loco_info[cols]
with open(FOLDER + 'others/loco_info_power_type.txt', 'w') as f:
    for m in loco_info.message.values:
        f.write(m)
        f.write('\n')
f.close()


# In[274]:

slot

