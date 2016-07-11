
# coding: utf-8

# In[574]:

FOLDER = 'resources/'

import numpy as np
import pandas as pd
import time, datetime
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

#%matplotlib inline
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


# In[575]:

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


# In[576]:

def nice_time(t):
    #if not time_format: time_format = '%b %d, %H:%M'
    return time.strftime(time_format, time.localtime(t)) if t > 0 else ''

def nice_print(s, cols, num=True):
    if num:
        print(s.reset_index()[cols].to_string())
    else:
        print(s[cols].to_string(index=False))


# In[577]:

st_name = 'ИРКУТСК-СОРТИРОВОЧНЫЙ'
team_plan['depot_name'] = team_plan.depot.map(st_names.name)
cols = ['team', 'number', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'state', 'loco']
team_plan[(team_plan.st_from_name == st_name) & (team_plan.state.isin([0, 1]))
          & (team_plan.time_start >= current_time) & (team_plan.time_start < current_time + 24 * 3600)].sort_values('time_start')[cols]

team_plan[team_plan.number == 9205004408][cols]


# In[578]:

team_info['depot_name'] = team_info.depot.map(st_names.name)
team_info['in_plan'] = team_info.team.isin(team_plan[team_plan.state == 1].team)
team_info['oper_time_f'] = team_info.oper_time.apply(nice_time)
cols = ['team', 'number', 'depot_name', 'state']
a = team_info[(team_info.ttype == 1) & (team_info.loc_name == 'СЛЮДЯНКА I') 
          #& (team_info.depot_name.isin(['СЛЮДЯНКА I']))
          & (team_info.oper_time < current_time + 24 * 3600)]
a.depot_name.value_counts()


# In[579]:

b = a[a.in_plan == True]
#cols = ['team', 'number', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'state', 'loco']
#team_plan[team_plan.team.isin(b.team)][cols]
cols = ['team', 'regions', 'number', 'depot_name', 'ready_type', 'depot_st', 'depot_time', 'return_st', 'oper_time_f', 'state']
b = a[(a.in_plan) & (a.regions.apply(lambda x: '2002118236' in x))].sort_values('oper_time')[cols]
print(b.team.count()) # всего 30 слюдянковских бригад, которые могут ездить до Иркутска
# Еще 29 бригад из Зимы и Иркутска - они тоже могут ехать в нечетную сторону
team_plan[(team_plan.team.isin(b.team)) & (team_plan.st_from_name == 'СЛЮДЯНКА I') & (team_plan.state.isin([0, 1]))].st_to_name.value_counts()


# In[580]:

twr = pd.read_csv(FOLDER + 'team_work_region.csv')
twr['link'] = twr.link.apply(literal_eval)
twr['st_from_name'] = twr.link.apply(lambda x: x[0]).map(st_names.name)
twr['st_to_name'] = twr.link.apply(lambda x: x[1]).map(st_names.name)
twr[twr.twr == 2002118236]


# In[581]:

train_plan['train_type'] = train_plan.train.apply(lambda x: x[0])
train_plan[(train_plan.st_from_name == 'СЛЮДЯНКА I') & (train_plan.st_to_name == 'СЛЮДЯНКА II')
          & (train_plan.time_start >= current_time) & (train_plan.time_start < current_time + 24 * 3600)].train_type.value_counts()

# Всего 96 поездов в нечетную сторону из Слюдянки!!!
# Из них всего 15 локомотивов резервом и 81 настоящий поезд


# Итого на Слюдянку надо 96 бригад. А есть только 59 (на начало планирования)
# Надо где-то найти еще 37. 
# Еще 10 бригад едут от Иркутска в Слюдянку на начало планирования. Осталось 27.

# In[582]:

cols = ['team', 'number', 'depot_name', 'depot_st', 'depot_time', 'state', 'loc_name', 'oper_time_f']
team_info['link'] = list(zip(team_info.st_from, team_info.st_to))
links = pd.read_csv(FOLDER + 'link.csv', dtype={'st_from':str, 'st_to':str})
links['link'] = list(zip(links.st_from, links.st_to))
team_info['dir'] = team_info.link.map(links.set_index('link')['dir'])
team_info[(team_info.depot_name.isin(['ЗИМА', 'ИРКУТСК-СОРТИРОВОЧНЫЙ'])) & (team_info.state.isin(['2','3','4']) == False)
         & (team_info['dir'] == 0)].sort_values('oper_time')[cols]


# In[583]:

cols = ['team', 'number', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'state', 'loco']
team_plan[team_plan.team == '200200035170'][cols]


# In[584]:

cols = ['team', 'number', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'state', 'loco']
team_plan[(team_plan.depot_name == st_name) & (team_plan.st_to_name == 'ГОНЧАРОВО')
          & (team_plan.time_start >= current_time) & (team_plan.time_start < current_time + 8 * 3600)
         & (team_plan.state == 1) & (team_plan.st_from_name == st_name)].sort_values('time_start')[cols]


# In[585]:

team_info[(team_info.depot_st == '-1') & (team_info.depot_time == -1)].team.count() / team_info.team.count()


# In[586]:

train_info['in_plan'] = train_info.train.isin(train_plan.train)
train_info[train_info.in_plan == False].train.count() / train_info.train.count()


# In[587]:

print(nice_time(current_time))
train_info['oper_time_f'] = train_info.oper_time.apply(nice_time)
train_info[train_info.in_plan == False][['train', 'number', 'ind434', 'joint', 'oper_time_f', 'loc_name']]


# In[588]:

def get_station(name):
    s = stations[stations.name.apply(lambda x: name.upper() in x)]
    return s.station.unique()[0], s.name.unique()[0]    


# In[589]:

print(nice_time(current_time))
st, st_name = get_station('чернышев')
team_info['in_plan'] = team_info.team.isin(team_plan[team_plan.state == 1].team)
team_info[(team_info.oper_location == st)           
          & (team_info.regions.apply(lambda x: '200290072545' in x)) & (team_info.depot_time != -1)]\
            [['team', 'regions', 'depot_name', 'loc_name', 'oper_time_f', 'in_plan']].sort_values(['depot_name', 'oper_time_f'])
#a = [2002118291, 200290072545]
#twr[twr.twr.isin(a)]


# In[590]:

loco_plan['loco_time'] = list(zip(loco_plan.loco, loco_plan.time_start))
team_plan['loco_time'] = list(zip(team_plan.loco, team_plan.time_start))
loco_plan['team'] = loco_plan.loco_time.map(team_plan.drop_duplicates('loco_time').set_index('loco_time').team)
loco_plan[loco_plan.loco == '200215585260'][['loco', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'team']]


# In[591]:

slot = pd.read_csv(FOLDER + 'slot.csv', dtype={'st_from':str, 'st_to':str})
add_info(slot)
slot[(slot.st_from == st) & (slot.time_start > current_time) & (slot.time_start < current_time + 24 * 3600) 
    & (slot.st_to_name == 'КУЭНГА')].sort_values('time_start')\
[['slot', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm']]


# In[592]:

print(nice_time(current_time))
slot[slot.time_start_norm.apply(lambda x: 'Jul 04' in x)].slot.drop_duplicates().count()


# In[593]:

not_used = [9205004609, 9205007593, 9205000742, 9205002994, 9205007639, 9205004113, 9205004564, 9205031367, 9205031292, 
            9205000564, 9205002681, 9205008860, 9205004041, 9205002635, 9205008056, 9205000264, 9205003316, 9205008377, 
            9205007941, 9205002097, 9205007883, 9205030603, 9205003779, 9205003550, 9205030823, 9205003873, 9205004359, 
            9205008112, 9205000326, 9205005114, 9205007141, 9205001325, 9205002009, 9205000629, 9205002708, 9205004902, 
            9205007023, 9205007920, 9205004884, 9205031354, 9205002345, 9205007837, 9205001021, 9205002942, 9205004656, 
            9205008012, 9205007263, 9205007663, 9205000842, 9205000608]
team_info['depot_time_f'] = team_info.depot_time.apply(nice_time)
cols = ['team', 'ttype', 'number', 'regions', 'depot_st', 'depot_time_f', 'return_st', 'return_time', 'oper_time_f', 'loc_name', 'state', 'loco']
team_info[(team_info.number.isin(not_used)) 
          & (team_info.state == '3') 
         ].sort_values('depot_time')[cols]
#twr[twr.twr == 2002118233].sort_values('st_from_name')


# In[594]:

st, st_name = get_station('иркутск-с')
cols = ['team', 'depot_name', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'state']
team_plan['depot_name'] = team_plan.team.map(team_info.set_index('team').depot_name)
ts = team_plan[team_plan.state.isin([0, 1])].drop_duplicates('team')
team_plan['fake_depot_name'] = team_plan.team.map(ts.set_index('team').st_from_name)
team_plan.depot_name.fillna(team_plan.fake_depot_name, inplace=True)
team_plan['team_type'] = team_plan.team.apply(lambda x: int(x[0]))
team_plan[(team_plan.st_from == st) & (team_plan.depot_name == st_name) & (team_plan.team_type == 7)
          & (team_plan.time_start >= current_time) 
          & (team_plan.time_start < current_time + 24 * 3600) & (team_plan.state.isin([0, 1]))].sort_values('time_start')[cols]


# In[595]:

cols = ['train', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm']
train_plan[(train_plan.st_from == st) 
           & (train_plan.time_start >= current_time) 
           & (train_plan.time_start < current_time + 24 * 3600)].sort_values('time_start')[cols].st_to_name.value_counts()


# In[596]:

loco_plan[(loco_plan.st_from == st) & (loco_plan.state.isin([0, 1]))
           & (loco_plan.time_start >= current_time) 
           & (loco_plan.time_start < current_time + 24 * 3600)].sort_values('time_start')[cols].st_to_name.value_counts()


# In[597]:

team_plan[(team_plan.st_from == st) & (team_plan.state.isin([0, 1])) & (team_plan.depot_name == st_name)
           & (team_plan.time_start >= current_time) 
           & (team_plan.time_start < current_time + 24 * 3600)].sort_values('time_start').st_to_name.value_counts()


# In[598]:

team_plan['all_states'] = team_plan.team.map(team_plan.groupby('team').state.unique())
cols = ['team', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'all_states', 'state_info']
team_plan[(team_plan.st_from == st) & (team_plan.state.isin([0])) & (team_plan.depot_name == st_name)
           & (team_plan.time_start >= current_time) 
           & (team_plan.time_start < current_time + 24 * 3600)].sort_values('time_start')[cols]


# In[599]:

slot['link'] = list(zip(slot.st_from, slot.st_to))
a = slot.groupby(['link', 'time_start']).slot.count()
a[a > 1]
b = slot.set_index(['link', 'time_start']).join(a, rsuffix='_').reset_index()[['link', 'st_from_name', 'st_to_name', 'time_start_norm', 'slot_']]


# In[600]:

files = [files for root, directories, files in os.walk('./resources/others')][0]
times = {}
os.chdir('./resources/others')
try:
    for f in files:
        if 'Бригады_УТХ' in f:
            times[f] = int(os.path.getmtime(f))    

    if times != {}:
        uth_filename = max(times, key=lambda k: times[k])
        date_modified = times[uth_filename]
    else:
        uth_filename = 'Бригады_УТХ' + '.xls'
        date_modified = 0
    print('Данные об УТХ-бригадах взяты из файла %s (дата изменения %s)' % (uth_filename, nice_time(date_modified)))
    os.chdir('..')
    os.chdir('..')
except:
    os.chdir('..')
    os.chdir('..')


# In[601]:

import os
files = [files for root, directories, files in os.walk('./input')][0]
files = [file for file in files if '20160703' in file]
files


# In[602]:

#import zipfile
#for file in files:
#    zip_ref = zipfile.ZipFile('./input/' + file, 'r')
#    zip_ref.extractall('./input')
#    zip_ref.close()
#    %run read.py
#    slot = pd.read_csv(FOLDER + 'slot.csv', dtype={'st_from':str, 'st_to':str})
#    slot['time_start_f'] = slot.time_start.apply(nice_time)
#    slot['is_jul_03'] = slot.time_start_f.apply(lambda x: 'Jul 03' in x)
#    slot['is_jul_04'] = slot.time_start_f.apply(lambda x: 'Jul 04' in x)
#    slot['is_jul_05'] = slot.time_start_f.apply(lambda x: 'Jul 05' in x)
#    slot.head(10)
#    print('Jul 03: %d\nJul 04: %d\nJul 05: %d' % 
#          (slot[slot.is_jul_03].slot.count(), slot[slot.is_jul_04].slot.count(), slot[slot.is_jul_05].slot.count()))


# In[603]:

slot = pd.read_csv(FOLDER + 'slot.csv', dtype={'st_from':str, 'st_to':str})
#slot['time_start_f'] = slot.time_start.apply(nice_time)
add_info(slot)
slot[(slot.st_from_name == 'ИРКУТСК-СОРТИРОВОЧНЫЙ') & (slot.time_start_norm.apply(lambda x: 'Jul 05' in x))].st_to_name.value_counts()


# In[604]:

print('Время начала планирования: %s' % (time.strftime(time_format, time.localtime(current_time))))
team_info['dt_norm'] = team_info.depot_time.apply(lambda x: time.strftime(time_format, time.localtime(x)) if x > 0 else x)
team_info['rt_norm'] = team_info.return_time.apply(lambda x: time.strftime(time_format, time.localtime(x)) if x > 0 else x)
team_info['rst_norm'] = team_info.rest_time.apply(lambda x: time.strftime(time_format, time.localtime(x)) if x > 0 else x)
team_info['rest_dep_delta'] = np.round(((team_info.rest_time - team_info.depot_time) / 3600), 2)
team_info['return_dep_delta'] = np.round(((team_info.return_time - team_info.depot_time) / 3600), 2)

dep_less_rest = team_info[(team_info.depot_time < team_info.rest_time) & (team_info.dt_norm != -1)]
info_cols = ['team', 'ttype', 'number', 'dt_norm', 'rst_norm', 'rest_dep_delta', 'ready_type', 'state']
print('Всего %d бригад, у которых время последней явки в депо намного меньше (на 12+ часов) переданного времени начала отдыха. Примеры:' 
          % dep_less_rest[dep_less_rest.rest_dep_delta > 12].team.drop_duplicates().count())
print(dep_less_rest[dep_less_rest.rest_dep_delta > 12][info_cols].sort_values('rest_dep_delta', ascending=False).head(10).to_string(index=False))

dep_less_return = team_info[(team_info.depot_time < team_info.return_time) & (team_info.dt_norm != -1)]
info_cols = ['team', 'ttype', 'number', 'dt_norm', 'rst_norm', 'return_dep_delta', 'ready_type', 'state']
print('\nВсего %d бригад, у которых время последней явки в депо намного меньше (на 18+ часов) времени явки в пункте оборота. Примеры:' 
          % dep_less_return[dep_less_return.return_dep_delta > 12].team.drop_duplicates().count())
print(dep_less_return[dep_less_return.return_dep_delta > 12][info_cols].sort_values('return_dep_delta', ascending=False).head(10).to_string(index=False))


# In[605]:

nice_time(team_info.oper_time.max())


# In[606]:

nice_time(current_time)


# In[607]:

inds = ['9379-342-9861', '8626-125-9861', '8622-692-9861', '8630-837-9861']
train_info['oper_time_f'] = train_info.oper_time.apply(nice_time)
#print(train_info[train_info.ind434 == ind][['train', 'ind434', 'number', 'oper_time_f', 'loc_name']].to_string(index=False))
for ind in inds:
    a = train_plan[train_plan.ind434 == ind][['train', 'number', 'ind434', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm']]
    print(a.to_string(index=False), '\n')


# In[608]:

teams = [9608009639, 9608003284, 9608000026, 9608010886]
for team in teams:
    print(team_plan[team_plan.number == team][['team', 'number', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'loco', 'state']].to_string(index=False), '\n')


# In[609]:

a['link'] = list(zip(a.st_from_name, a.st_to_name))
slot['link'] = list(zip(slot.st_from_name, slot.st_to_name))
slot['tt'] = slot.time_end - slot.time_start
slot[slot.link.isin(a.link)].groupby('link').tt.mean()
slot[slot.link == ('БОЛЬШОЙ ЛУГ', 'ГОНЧАРОВО')].sort_values('time_start')[['slot', 'link', 'time_start_norm', 'tt']]


# In[610]:

print(nice_time(current_time))


# In[611]:

print(train_info[train_info.ind434.isin(inds)].sort_values('ind434').reset_index()[['train', 'number', 'ind434', 'oper', 'oper_time_f', 'loc_name']].to_string())


# In[612]:

[ind for ind in inds if ind not in train_info.ind434.unique()]


# In[613]:

cols = ['team', 'number', 'loc_name', 'state', 'oper_time_f']
nice_print(team_info[(team_info.state == '0') & (team_info.link.isin(links.link) == False)], cols)


# In[614]:

ind = '8642-798-9857'
train_info[train_info.ind434 == ind][['train', 'number', 'weight', 'oper_time_f', 'loc_name']]


# In[615]:

loco_info['ser_name'] = loco_info.series.map(loco_series.set_index('ser_id').ser_name)
loco_plan['ser_name'] = loco_plan.loco.map(loco_info.set_index('loco').ser_name)
loco_info[loco_info.train == '210217266516'][['loco', 'number', 'ser_name']]
loco_info[loco_info.number == 2680]
loco_plan[loco_plan.loco == '200200093671'][['loco', 'number', 'ser_name', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'state', 'train']]


# In[617]:

nice_time(current_time)


# In[637]:

st_name = 'ИРКУТСК-СОРТИРОВОЧНЫЙ'
cols = ['team', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'state', 'loco', 'all_states']
irk = team_plan[(team_plan.st_from_name == st_name) & (team_plan.state.isin([0, 1]))
          & (team_plan.depot_name == st_name)
          & (team_plan.time_start >= current_time) & (team_plan.time_start < current_time + 24 * 3600)]
irk.team.count()


# In[634]:

cols = ['train', 'train_type', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm']
a = train_plan[(train_plan.st_from_name == st_name)
          & (train_plan.time_start >= current_time) & (team_plan.time_start < current_time + 24 * 3600)].sort_values('time_start')[cols]
a.train_type.value_counts()


# In[672]:

b = team_plan[(team_plan.st_from_name == st_name) & (team_plan.state.isin([0, 1]))          
          & (team_plan.time_start >= current_time) & (team_plan.time_start < current_time + 24 * 3600)].groupby('st_to_name').depot_name.value_counts()
q = team_plan[(team_plan.st_from_name == st_name) & (team_plan.state.isin([0, 1]))          
          & (team_plan.time_start >= current_time) & (team_plan.time_start < current_time + 24 * 3600)].groupby('st_to_name').team.count()
b = b.reset_index()
b['s'] = b.st_to_name.map(q)
b['perc'] = b[0] / b['s']
b[['st_to_name', 'depot_name', 'perc']].set_index(['st_to_name', 'depot_name']


# In[654]:

cols = ['train', 'train_type', 'time_start_norm', 'time_end_norm', 'start_st', 'end_st']
train_p = a[a.train_type == '8']
train_plan['start_st'] = train_plan.train.map(train_plan.drop_duplicates('train').set_index('train').st_from_name)
train_plan['end_st'] = train_plan.train.map(train_plan.drop_duplicates('train', keep='last').set_index('train').st_to_name)
train_plan[train_plan.train.isin(train_p.train)][cols].drop_duplicates('train').start_st.value_counts()


# In[ ]:

reg = '2002119299'
a = loco_info[(loco_info.regions.apply(lambda x: reg in x)) 
          & (loco_info.st_from == '-1') 
          & (loco_info.train == '-1')][['loco', 'loc_name', 'train']]
st = 'СУХОВСКАЯ'
paths = pd.read_csv(FOLDER + 'mandatory/paths.csv', encoding='utf-8-sig', sep=';')


# In[902]:

regs = stations.loco_region.unique()
#select = regs[np.random.randint(len(regs))]
#select
for reg in regs:    
    a = loco_info[(loco_info.regions.apply(lambda x: str(reg) in x)) 
          & (loco_info.st_from == '-1') 
          & (loco_info.train == '-1')][['loco', 'loc_name', 'train']]
    if not a.empty:
        b = stations[stations.loco_region == reg].name.unique()
        st = b[np.random.randint(len(b))]        
        paths_s = paths[(paths.st_to == st) & (paths.st_from.isin(b))]
        #lim = np.percentile(paths_s.cost, 25)
        m = paths_s.cost.max()
        costs = paths_s[['st_from', 'cost']].set_index('st_from').to_dict()['cost']
        a['cost'] = a.loc_name.apply(lambda x: costs[x] if x in costs.keys() else 100)    
        act_lim = np.percentile(a.cost, 25)
        good_n = a[a.cost.apply(lambda x: x <= lim)].loco.count()
        total_n = a.loco.count()
        print('Region %s, station %s, 25%%-percentile = %.2f (max reg = %.2f), number of available locos = %d of %d (%.2f%%)'
             % (reg, st, act_lim, m, good_n, total_n, 100 * good_n / total_n))
    else:
        print('Empty reg', reg)


# In[905]:

reg = 2002119299
a = loco_info[(loco_info.regions.apply(lambda x: str(reg) in x)) 
      & (loco_info.st_from == '-1') 
      & (loco_info.train == '-1')][['loco', 'loc_name', 'train']]
if not a.empty:
    b = stations[stations.loco_region == reg].name.unique()
    #st = b[np.random.randint(len(b))]        
    st = 'МАРИИНСК'
    paths_s = paths[(paths.st_to == st) & (paths.st_from.isin(b))]
    lim = np.percentile(paths_s.cost, 25)
    m = paths_s.cost.max()
    costs = paths_s[['st_from', 'cost']].set_index('st_from').to_dict()['cost']
    a['cost'] = a.loc_name.apply(lambda x: costs[x] if x in costs.keys() else 100)    
    print(a.sort_values('cost').to_string(index=False))
    act_lim = np.percentile(a.cost, 25)
    good_n = a[a.cost.apply(lambda x: x <= lim)].loco.count()
    total_n = a.loco.count()
    print('Region %s, station %s, 25%%-percentile = %.2f (actual = %.2f, max = %.2f), number of available locos = %d of %d (%.2f%%)'
         % (reg, st, lim, act_lim, m, good_n, total_n, 100 * good_n / total_n))
else:
    print('Empty reg', reg)


# In[911]:

def func(row):
    df = paths[(paths.st_from == row.st_from_name) & (paths.st_to == row.st_to_name)]
    return df.cost.values[0] if not df.empty else 1000

start = train_plan[train_plan.train_type == '8'].drop_duplicates('train')
end = train_plan[train_plan.train_type == '8'].drop_duplicates('train', keep='last')
start['end'] = start.train.map(end.set_index('train').st_to_name)
q = start[['train', 'st_from_name', 'st_to_name']]

q['cost'] = q.apply(lambda row: func(row), axis=1)
q

