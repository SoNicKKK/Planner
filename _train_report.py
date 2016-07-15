
# coding: utf-8

# <a id='toc'></a>
# # Отчет по запланированным поездам

# 1. [Проверка совпадения четности номеров поездов и направления движения](#oddity)
# 2. [Анализ времен стоянок поездов на станциях смены локомотивов и бригад.](#stop_time)
#    1. [Смена локомотивов](#stop_loco)
#    2. [Смена бригад](#stop_team)
# 4. [Выявление случаев "близкой" отправки поездов с одной станции (с интервалом меньше 5 минут)](#dep_interval)
# 5. [Анализ отклонения запланированного на сутки количества поездов от данных АС ССП (данные средние по суткам).](#ssp)
# 6. [Проверка соответствия первого участка в запланированном маршруте и исходного факта](#info_plan_depart)
# 6. [Детальное сравнение количества поездов по Иркутску с данными ССП](#irk_ssp)
# 6. [Создание отчета](#report)

# ### Функции для экспорта в HTML

# In[325]:

report = ''
FOLDER = 'resources/'
REPORT_FOLDER = 'report/'
PRINT = False


# In[326]:

def add_line(line, p=PRINT):    
    global report        
    if p:                
        if type(line) == pd.core.frame.DataFrame:
            print(line.to_string(index=False))
        elif type(line) == pd.core.series.Series:
            print(line.to_string())
        else:
            print(line)
    if type(line) == pd.core.frame.DataFrame:        
        report += ('%s<br>' % line.to_html(index=False))
    elif type(line) == pd.core.series.Series:
        report += ('%s<br>' % line.to_frame().reset_index().to_html(index=False))
    else:        
        report += ('%s<br>' % line)
    
def add_header(header, h=4, p=PRINT):
    global report
    report += ('<h%d>%s</h%d>' % (h, header, h))
    if p:
        print(header)

def add_image(filename, scale=0.4):
    global report
    report += ('<img src="%s" alt="%s" height="%d%%">' % (filename, filename, int(scale * 100)))

def create_report(filename):
    global report
    report = report.replace('<table border="1" class="dataframe">','<table class="table table-striped">')
    html_string = '''
        <html>
            <head>
                <link rel="stylesheet" href="skeleton.css">
                <style>body{ margin:20 20; background:whitesmoke; }
                table {table-layout : fixed}
                </style>
            </head>
            <body>                
                %s
            </body>
        </html>''' % (report)
    f = open(filename,'w', encoding='utf-8-sig')
    f.write(html_string)
    f.close()
    print('Отчет сформирован: %s' % filename)


# ## Загрузка и подготовка данных

# In[327]:

import numpy as np
import pandas as pd
import time, datetime
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns

#get_ipython().magic('matplotlib inline')
plt.style.use('fivethirtyeight')
plt.rc('font', family='Tahoma')

pd.set_option('max_rows', 50)

time_format = '%b %d, %H:%M'

start_time = time.time()
current_time = pd.read_csv(FOLDER + 'current_time.csv').current_time[0]
twr          = pd.read_csv(FOLDER + 'team_work_region.csv', converters={'twr':str})
links        = pd.read_csv(FOLDER + 'link.csv', converters={'st_from':str, 'st_to':str})
stations     = pd.read_csv(FOLDER + 'station.csv', converters={'station':str})
train_info   = pd.read_csv(FOLDER + 'train_info.csv', converters={'train': str, 'st_from':str, 'st_to':str, 'joint':str,
                                                                 'start_st':str, 'end_st':str})
train_plan   = pd.read_csv(FOLDER + 'slot_train.csv', converters={'train': str, 'st_from':str, 'st_to':str})
loco_info    = pd.read_csv(FOLDER + 'loco_attributes.csv', converters={'train':str, 'loco':str, 'st_from':str, 'st_to':str})
loco_plan    = pd.read_csv(FOLDER + 'slot_loco.csv', converters={'train':str, 'loco':str, 'st_from':str, 'st_to':str})
team_info    = pd.read_csv(FOLDER + 'team_attributes.csv', converters={'team':str,'depot':str, 'oper_location':str,                                                      'st_from':str, 'st_to':str, 'loco':str, 'depot_st':str})
team_plan    = pd.read_csv(FOLDER + 'slot_team.csv', converters={'team':str,'loco':str, 'st_from':str, 'st_to':str})
loco_series  = pd.read_csv(FOLDER + 'loco_series.csv')
task         = pd.read_csv(FOLDER + 'task.csv', converters={'st_from':str, 'st_to':str, 'st_next':str})

st_names     = stations[['station', 'name', 'esr']].drop_duplicates().set_index('station')
team_info.regions = team_info.regions.apply(literal_eval)

print('Время составления отчета:', time.strftime(time_format, time.localtime()))
print('Время запуска планировщика: %s (%d)' % (time.strftime(time_format, time.localtime(current_time)), current_time))


# In[328]:

def nice_time(t):        
    return time.strftime(time_format, time.localtime(t)) if np.isnan(t) == False else None

def add_info(df):    
    if 'st_from' in df.columns:
        df['st_from_name'] = df.st_from.map(st_names.name)
    if 'st_to' in df.columns:
        df['st_to_name'] = df.st_to.map(st_names.name)
    if 'time_start' in df.columns:
        df['time_start_f'] = df.time_start.apply(lambda x: nice_time(x))
    if 'time_end' in df.columns:
        df['time_end_f'] = df.time_end.apply(lambda x: nice_time(x))
    if 'oper_location' in df.columns:
        df['oper_location_name'] = df.oper_location.map(st_names.name)    
        df.oper_location_name.fillna(0, inplace=True)
    if ('oper_location' in df.columns) & ('st_from' in df.columns) & ('st_to' in df.columns):        
        df['loc_name'] = df.oper_location_name
        df.loc[df.loc_name == 0, 'loc_name'] = df.st_from_name + ' - ' + df.st_to_name
    if 'oper_time' in df.columns:
        df['oper_time_f'] = df.oper_time.apply(lambda x: nice_time(x))    
    
# Добавляем во все таблицы названия станций на маршруте и времена отправления/прибытия в читабельном формате
add_info(train_plan), add_info(loco_plan), add_info(team_plan)
add_info(train_info), add_info(loco_info), add_info(team_info)

# Мержим таблицы _plan и _info для поездов, локомотивов и бригад
train_plan = train_plan.merge(train_info, on='train', suffixes=('', '_info'), how='left')
loco_plan = loco_plan.merge(loco_info, on='loco', suffixes=('', '_info'), how='left')
team_plan = team_plan.merge(team_info, on='team', suffixes=('', '_info'), how='left')

# Добавляем признаки поезда и бригады (реальный/локомотиво-резервный/объемный и реальная/фейковая)
train_plan['train_type'] = train_plan.train.apply(lambda x: str(x)[0])
team_plan['team_type'] = team_plan.team.apply(lambda x: 'Реальная' if str(x)[0] == '2' else 'Фейковая')

# Для локомотиво-резервных и объемных поездов заполняем номер
train_plan.loc[train_plan.train_type.isin(['8', '9']), 'number'] = train_plan.train.apply(lambda x: int(str(x)[-4:]))

# Добавляем подвязанные локомотив и бригаду в таблицы loco_plan и train_plan
if 'team' not in loco_plan.columns:
    loco_plan = pd.merge(loco_plan, team_plan.loc[team_plan.state == 1, ['team', 'loco', 'time_start']], 
                         on=['loco', 'time_start'], how='left')
if 'loco' not in train_plan.columns:
    train_plan = pd.merge(train_plan, loco_plan.loc[loco_plan.state.isin([0, 1]), ['loco', 'team', 'train', 'time_start']],
                     on=['train', 'time_start'], how='left')


# <a id='oddity'></a>
# ## Проверка совпадения четности номеров поездов и направления движения [ToC](#toc)

# In[329]:

add_header('Проверки по поездам', h=1, p=False)
add_header('Проверка совпадения четности номеров поездов и направления движения', h=2, p=False)


# In[330]:

train_plan['dir'] = train_plan.link.map(links.set_index('link').dir)
train_plan['odevity'] = (((train_plan.number / 2).astype(int) * 2 == train_plan.number).astype(int) + 1) % 2
train_plan['check_odd'] = train_plan.dir == train_plan.odevity
cols = ['train', 'number', 'st_from_name', 'st_to_name', 'dir', 'odevity', 'check_odd']
fail_dir_number = train_plan.drop_duplicates(subset=['train', 'number']).loc[(train_plan.train_type.isin(['8', '9'])) &
                                                          (train_plan.check_odd == False), cols]
if fail_dir_number.empty == False:
    add_header('Четность номеров поездов и направления не совпадает для %d поездов (показаны первые 10):' %
         len(fail_dir_number.index))
    pd.set_option('display.max_colwidth', 35)
    add_line(fail_dir_number.head(10))
else:
    add_line('Все четности совпадают')


# <a id='stop_time'></a>
# ## Анализ времен стоянок поездов на станциях смены локомотивов и бригад [ToC](#toc)

# #### Параметры для анализа

# In[331]:

# Минимальное время стоянки поезда для смены локомотива
min_loco_stop = 1 * 3600 # 1 hour = 60 min

# Минимальное время стоянки поезда для смены бригады
min_team_stop = 15 * 60 # 15 min

# Горизонт проверки
hor = 24 * 3600


# <a id='stop_loco'></a>
# ### Смена локомотивов [ToC](#toc)

# In[332]:

add_header('Анализ смен локомотивов на маршрутах поездов', h=2, p=False)


# #### Ищем станции смены локомотивов и считаем средние времена

# In[333]:

train_plan.columns
train_plan.loco.fillna('-1', inplace=True)
train_plan.team.fillna('-1', inplace=True)

train_plan['train_end'] = train_plan.train != train_plan.train.shift(-1)
train_plan['loco_end'] = (train_plan.loco != train_plan.loco.shift(-1)) | (train_plan.train_end)
train_plan['team_end'] = (train_plan.team != train_plan.team.shift(-1)) | (train_plan.loco_end)

train_plan['stop_time'] = train_plan.time_start.shift(-1) - train_plan.time_end
train_plan['stop_time_h'] = np.round((train_plan.stop_time / 3600), 2)

train_plan['next_loco'] = train_plan.loco.shift(-1)
train_plan['next_team'] = train_plan.team.shift(-1)

cols = ['train', 'st_from_name', 'st_to_name', 'loco', 'team', 'stop_time_h', 'next_loco']
loco_change = train_plan[(train_plan.train_end == False) & (train_plan.loco_end == True)
                        & (train_plan.time_end < current_time + hor)]


# In[334]:

add_header('Средние времена на смену локомотивов:')
add_line('- по всем сменам: %.2f ч.' % loco_change.stop_time_h.mean())
add_line('- по всем сменам с ненулевым временем: %.2f ч.' % loco_change[loco_change.stop_time_h > 0].stop_time_h.mean())


# #### Ищем поезда, у которых смена локомотивов происходит за нулевое время

# In[335]:

cols = ['train', 'st_from_name', 'st_to_name', 'loco', 'next_loco', 'stop_time_h']
nill_stop_times = loco_change[loco_change.stop_time == 0]
if not nill_stop_times.empty:
    add_header('Всего %d поездов, для которых смена локомотивов происходит за нулевое время. Примеры:' 
               % nill_stop_times.train.count())
    add_line(nill_stop_times[cols].head())
    cols = ['train', 'st_from_name', 'st_to_name', 'time_start_f', 'time_end_f', 'loco', 'team']
    add_line('')
    add_header('Полный план по одному из таких поездов:')
    for train in nill_stop_times.train.values[:1]:
        add_line(train_plan[train_plan.train == train][cols])
else:
    add_header('Нет локомотивов, для которых смена бригад происходит за нулевое время')


# #### Составляем статистику по всем станциям смены, загружаем список приоритетных станций смены

# In[336]:

cols = ['train', 'st_from_name', 'st_to_name', 'loco', 'next_loco', 'stop_time_h']
no_nill_stops = loco_change[loco_change.stop_time > 0]
no_nill_stops[cols]
#no_nill_stops.groupby('st_to_name').stop_time_h.mean()
st_change = no_nill_stops.groupby('st_to_name').train.count().to_frame()            .join(no_nill_stops.groupby('st_to_name').stop_time_h.mean()).reset_index()
st_change['stop_time_h'] = st_change.stop_time_h.apply(lambda x: np.round(x, 2))
st_change.sort_values('train', ascending=False).head(10)
priority_change_stations = pd.read_csv(FOLDER + 'mandatory/priority_loco_change_stations.csv').st_name.values


# #### Ищем поезда, у которых смена локомотивов происходит на неправильных станциях

# In[337]:

# bad change stations
bad_changes = st_change[st_change.st_to_name.isin(priority_change_stations) == False].sort_values('train', ascending=False)
if not bad_changes.empty:
    add_header('Всего %d поездов, у которых смена локомотива, скорее всего, происходит на неправильных станциях' 
               % bad_changes.train.sum())
    add_header('Примеры таких станций:')
    add_line(bad_changes.head(10))
    add_line('')
    cols = ['train', 'st_from_name', 'st_to_name', 'loco', 'next_loco', 'stop_time_h']
    st = bad_changes.iloc[0].st_to_name
    add_line('Поезда, у которых смена локомотива происходит на станции %s:' % st)
    add_line(loco_change[loco_change.st_to_name == st][cols])

    train_plan['loco_info'] = train_plan.train.map(loco_info.drop_duplicates('train').set_index('train').loco)
    cols = ['train', 'st_from_name', 'st_to_name', 'time_start_f', 'time_end_f', 'loco', 'loc_name', 'loco_info']
    add_line('')
    add_header('Полный план по одному из таких поездов:')
    with pd.option_context('display.max_colwidth', 20):
        add_line(train_plan[train_plan.train == loco_change[loco_change.st_to_name == st].iloc[0].train][cols])
else:
    add_header('Нет поездов, у которых смена локомотивов происходит на неправильной станции')


# #### Ищем поезда со слишком долгой стоянкой для смены локомотивов

# In[338]:

cols = ['train', 'st_from_name', 'st_to_name', 'loco', 'next_loco', 'stop_time_h']
long_change = loco_change[(loco_change.st_to_name.isin(priority_change_stations)) 
            & (loco_change.stop_time_h > 24)].sort_values('stop_time_h', ascending=False)
add_header('Всего %d случаев смены локомотива со стоянкой поезда более суток. Примеры:' % long_change.train.count())
add_line(long_change[cols].head(10))


# <a id='stop_team'></a>
# ### Смена бригад [ToC](#toc)

# In[339]:

add_header('Анализ смен бригад на маршрутах поездов', h=2, p=False)


# In[340]:

team_change = train_plan[(train_plan.loco_end == False) & (train_plan.team_end == True)
                        & (train_plan.time_end < current_time + hor)]


# In[341]:

add_header('Средние времена на смену бригады:')
add_line('- по всем сменам: %.2f ч.' % team_change.stop_time_h.mean())
add_line('- по всем сменам с ненулевым временем: %.2f ч.' % team_change[team_change.stop_time_h > 0].stop_time_h.mean())


# #### Ищем поезда, у которых смена бригады происходит за нулевое время

# In[342]:

cols = ['train', 'st_from_name', 'st_to_name', 'team', 'next_team', 'stop_time_h']
nill_stop_times = team_change[team_change.stop_time == 0]
if not nill_stop_times.empty:
    add_header('Всего %d поездов, для которых смена бригад происходит за нулевое время. Примеры:' 
               % nill_stop_times.train.count())
    add_line(nill_stop_times[cols].head())
    cols = ['train', 'st_from_name', 'st_to_name', 'time_start_f', 'time_end_f', 'loco', 'team']
    add_line('')
    add_header('Полный план по одному из таких поездов:')
    for train in nill_stop_times.train.values[:1]:
        add_line(train_plan[train_plan.train == train][cols])
else:
    add_header('Нет поездов, для которых смена бригад происходит за нулевое время')


# #### Ищем поезда со слишком долгой стоянкой для смены бригады

# In[343]:

cols = ['train', 'st_from_name', 'st_to_name', 'team', 'next_team', 'stop_time_h']
long_change = team_change[team_change.stop_time_h > 6].sort_values('stop_time', ascending=False)[cols]
if not long_change.empty:
    add_header('Всего %d случаев смены бригад со стоянкой поезда более 6 часов. Примеры:' % long_change.train.count())
    add_line(long_change[cols].head(10))
    add_line('')
    cols = ['train', 'st_from_name', 'st_to_name', 'time_start_f', 'time_end_f', 'loco', 'team']
    add_header('Полный план по одному из таких поездов:')
    for train in long_change.train.values[:1]:
        add_line(train_plan[train_plan.train == train][cols])
    add_line('')
    add_header('Станции, на которых чаще всего происходили длительные стоянки на смену бригад:')
    add_line(long_change.st_to_name.value_counts().head(10))
else:
    add_header('Нет поездов, у которых смена бригады происходит более 6 часов')


# <a id='dep_interval'></a>
# ## Поиск поездов с интервалом между отправлениями меньше допустимого [ToC](#toc)

# In[344]:

add_header('Поиск поездов с интервалом между отправлениями меньше допустимого', h=2, p=False)


# In[345]:

# Параметры

hor = 24 * 3600
min_time_delta = 5 * 60 # 5 minutes


# In[346]:

# Функция, которая возвращает датафрейм с коллизиями

def check_time_collision(df):
    df['link_end'] = (df.st_from != df.st_from.shift(-1)) | (df.st_to != df.st_to.shift(-1))
    df['link_start'] = (df.st_from != df.st_from.shift(1)) | (df.st_to != df.st_to.shift(1))
    df.loc[df.link_end == False, 'time_to_next'] = df.time_start.shift(-1) - df.time_start
    df.loc[df.link_start == False, 'time_to_prev'] = df.time_start - df.time_start.shift(1)
    collisions = df.loc[(df.time_to_next < min_time_delta) | (df.time_to_prev < min_time_delta)]
    return collisions


# In[347]:

add_line('Время начала планирования: %s' % nice_time(current_time))
cols = ['train', 'loco', 'team', 'st_from_name', 'st_to_name', 'time_start_f', 'time_end_f']
train_plan['train_start'] = train_plan.train != train_plan.train.shift(1)
train_plan.loc[train_plan.stop_time != -1, 'loco_start'] = (train_plan.loco != train_plan.loco.shift(1)) |                                                             (train_plan.train_start)
train_plan.loc[train_plan.stop_time != -1, 'team_start'] = train_plan.team != train_plan.team.shift(1)
mask = ((train_plan.loco_start == True) | (train_plan.team_start == True)) &        (train_plan.time_start < current_time + hor) & (train_plan.time_start >= current_time)
assign_mask = (train_plan.loco != '-1') & (train_plan.team != '-1')
cols_to_sort = ['st_from_name', 'st_to_name', 'time_start']
start_times = train_plan.loc[mask].sort_values(cols_to_sort)
start_times_no_res = train_plan.loc[mask & (train_plan.train_type.isin(['2', '9']))].sort_values(cols_to_sort)
start_times_real = train_plan.loc[mask & (train_plan.train_type == '2')].sort_values(cols_to_sort)
start_times_assign = train_plan.loc[mask & assign_mask].sort_values(cols_to_sort)
start_times_assign_no_res = train_plan.loc[mask & assign_mask &
                                           (train_plan.train_type.isin(['2', '9']))].sort_values(cols_to_sort)
start_times_assign_real = train_plan.loc[mask & assign_mask & (train_plan.train_type == '2')].sort_values(cols_to_sort)

all_coll = check_time_collision(start_times)
no_res_coll = check_time_collision(start_times_no_res)
real_coll = check_time_collision(start_times_real)
all_assign_coll = check_time_collision(start_times_assign)
no_res_assign_coll = check_time_collision(start_times_assign_no_res)
real_assign_coll = check_time_collision(start_times_assign_real)
train_n = len(start_times.train.drop_duplicates().index)
res = pd.DataFrame([['Все поезда', 'Без учета резервных', 'Без учета резервных и фейковых'],                    
                    [len(all_coll.index), len(no_res_coll.index), len(real_coll.index)],                    
                    [len(all_assign_coll.index), len(no_res_assign_coll.index), len(real_assign_coll.index)]]).T
res.columns = ['Тип', 'Из всех поездов', 'Из всех с подвязкой']
add_header('Количество коллизий (интервал между поездами меньше %d минут):' % (min_time_delta / 60))
add_line(res)
add_header('\nРеальные и фейковые поезда с интервалами меньше %d минут (первые 20):' % (min_time_delta / 60))
add_line(no_res_coll[cols].head(20))


# <a id='ssp'></a>
# ## Сравнение количества запланированных поездов с данными АС ССП [ToC](#toc)

# In[348]:

add_header('Сравнение количества запланированных поездов с данными АС ССП', h=2, p=False)


# In[349]:

def count_volumes(full_plan, df_ssp):
    hor = 24 * 3600
    df_ssp.dep_dir.fillna(0, inplace=True)    
    df_ssp.loc[df_ssp.dep_dir == 0, 'depart'] = 0
    df_ssp.dropna(subset=['depart'], inplace=True)
    
    ssp_st_froms = df_ssp.loc[df_ssp.dep_dir == 0].station
    mask_time = (full_plan.time_start >= current_time) & (full_plan.time_start < current_time + hor)
    mask_type = full_plan.train_type.isin(['2', '9'])
    trains = full_plan.loc[mask_time & mask_type & full_plan.st_from_name.isin(ssp_st_froms)].                        groupby(['st_from_name', 'st_to_name']).train.count()
    df_ssp.loc[df_ssp.dep_dir == 0, 'st_from_name'] = df_ssp.station
    df_ssp = df_ssp.fillna(method='ffill')
    df_ssp['st_to_name'] = df_ssp.station
    replace_st_from_names = df_ssp.loc[df_ssp.dep_dir == 0, ['st_from_name', 'st_show_name']].drop_duplicates()
    df_ssp['st_from_show'] = df_ssp.st_from_name.map(replace_st_from_names.set_index('st_from_name').st_show_name)
    df_ssp['st_to_show'] = df_ssp.st_show_name    
    return trains.to_frame().join(df_ssp[['st_from_name', 'st_to_name', 'depart', 'st_from_show', 'st_to_show', 'dep_dir']].                                  set_index(['st_from_name', 'st_to_name'])).reset_index()


# In[350]:

def show_barplot(df, road_name):
    df['delta'] = df.train - df.depart
    df['percent'] = np.round(100 * df.delta / df.depart, 2)
    df['st_from_short'] = df.st_from_show.apply(lambda x: str(x)[:25])
    df['st_to_short'] = df.st_to_show.apply(lambda x: str(x)[:25])
    df['link'] = df.st_from_short + ' - ' + df.st_to_short 
    
    print('%s железная дорога:' % road_name)
    print('Среднее и медиана абсолютного отклонения: %.2f, %.2f' % (df.delta.mean(), df.delta.median()))
    print('Среднее и медиана относительного отклонения (в процентах): %.2f%%, %.2f%%' % (df.percent.mean(), df.percent.median()))

    b = df.sort_values('delta', ascending=False)
    sns.set_style('whitegrid')
    sns.set_context('poster', font_scale=0.7, rc={'axes.labelsize': 18})
    c = sns.barplot(y='link', x='delta', data=df.sort_values('delta', ascending=False), palette='coolwarm')
    xlabel = '%s ж/д: отклонение от данных АС ССП по отправлению поездов на сутки' % road_name
    c.set(xlabel=xlabel, ylabel='')


# In[351]:

VOL_PERCENT = 0.9

def show_two_barplots(df, road_name, save=False, btype='less'):
    df['st_from_short'] = df.st_from_show.apply(lambda x: str(x)[:25])
    df['st_to_short'] = df.st_to_show.apply(lambda x: str(x)[:25])
    df['link'] = df.st_from_short + ' - ' + df.st_to_short     
    sns.set_style('whitegrid')    
    sns.set_context('poster', font_scale=0.7, rc={'axes.titlesize':18, 'axes.labelsize':14})
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(14,20))    
    df.depart.fillna(0, inplace=True)    
    df['bottom'] = df.apply(lambda row: row.depart if row.train <= row.depart else row.train, axis=1)
    df['top'] = df.apply(lambda row: row.train if row.train <= row.depart else row.depart, axis=1)     
    
    sns.set_color_codes('pastel')
    sns.barplot(x='bottom', y='link', data=df[df.train <= VOL_PERCENT * df.depart].sort_values('bottom'), 
                label='Поезда из АС ССП', color="b", orient='h', ax=ax[0])
    sns.barplot(x='bottom', y='link', data=df[df.train * VOL_PERCENT > df.depart].sort_values('bottom'), 
                label='Поезда в результатах', color="r", orient='h', ax=ax[1])
    sns.set_color_codes('muted')
    sns.barplot(x='top', y='link', data=df[df.train <= VOL_PERCENT * df.depart].sort_values('bottom'), 
                label='Поезда в результатах', color="b", orient='h', ax=ax[0])
    sns.barplot(x='top', y='link', data=df[df.train * VOL_PERCENT > df.depart].sort_values('bottom'), 
                label='Поезда из АС ССП', color="r", orient='h', ax=ax[1])    
    ax[0].legend(ncol=1, loc="upper right", frameon=True)    
    ax[1].legend(ncol=1, loc="upper right", frameon=True)
    ax[0].set(xlabel='', title='Нехватка запланированных поездов')
    ax[1].set(xlabel='', title='Избыток запланированных поездов')
    
    sns.despine()
    if save:
        filename = road_name + '.png'
        fig.savefig(REPORT_FOLDER + filename, bbox_inches='tight')
        add_image(filename, scale=1.0)


# In[352]:

def func(x):
    return np.round(np.sqrt(np.mean(x ** 2)), 2)

def print_ssp_stats(ssp, road_name):
    df = count_volumes(train_plan, ssp)
    df.rename(columns={'train':'planned', 'depart':'ssp'}, inplace=True)
    df.dropna(subset=['ssp'], inplace=True)
    df['delta'] = df.planned - df.ssp    
    cols = ['st_from_name', 'st_to_show', 'dep_dir', 'planned', 'ssp', 'delta']    
    add_header('Дорога %s' % road_name, h=3)
    add_header('Сравнение запланированного и "нормативного" количества поездов:')
    add_line(df.sort_values(['dep_dir', 'delta'])[cols])    
    add_header('\nСреднее отклонение по количеству поездов по направлениям:')
    add_line(df.groupby('dep_dir').delta.mean().apply(lambda x: np.round(x, 2)))
    add_header('\nСреднеквадратичное отклонение по направлениям:')
    add_line(df.groupby('dep_dir').delta.agg(func))


# In[353]:

krs = pd.read_csv(FOLDER + 'mandatory/SSP_KRS.csv', sep=';')
vsib = pd.read_csv(FOLDER + 'mandatory/SSP_VSIB.csv', sep=';')
zab = pd.read_csv(FOLDER + 'mandatory/SSP_ZAB.csv', sep=';')
dvs = pd.read_csv(FOLDER + 'mandatory/SSP_DVS.csv', sep=';')


# In[354]:

print_ssp_stats(krs, 'КРАС')


# In[355]:

print_ssp_stats(vsib, 'ВСИБ')


# In[356]:

print_ssp_stats(zab, 'ЗАБ')


# In[357]:

print_ssp_stats(dvs, 'ДВС')


# In[358]:

# Пример построения barplot

#krs = pd.read_csv(FOLDER + 'mandatory/SSP_KRS.csv', sep=';')
#add_header('Красноярская дорога')
#try:
#    show_two_barplots(count_volumes(train_plan, krs), 'Красноярская', save=True, btype='less')
#except:
#    add_line('Красноярская дорога: ошибка в построении графика')


# <a id='info_plan_depart'></a>
# ## Проверка соответствия первого участка в запланированном маршруте и исходного факта [ToC](#toc)

# In[359]:

add_header('Проверка соответствия первого участка в запланированном маршруте и исходного факта', h=2, p=False)


# In[360]:

cols = ['train', 'st_from_name', 'st_to_name', 'time_start_f', 'st_from_name_info', 'st_to_name_info', 'oper_time_f']
td_plan = train_plan[(train_plan.st_from_info.isnull() == False) 
                     & (train_plan.st_from_info != '-1')].drop_duplicates('train')
td_bad_track = td_plan[(td_plan.st_from != td_plan.st_from_info) | (td_plan.st_to != td_plan.st_to_info)]
add_header('Поезда, у которых первый участок в маршруте не совпадает с исходным (всего %d, показаны первые 10):' 
          % td_bad_track.train.count())
add_line(td_bad_track[cols])

td_bad_time = td_plan[(td_plan.time_start != td_plan.oper_time)]
add_header('\nПоезда, у которых время отправления на первый участок в маршруте не совпадает с фактическим (всего %d, показаны первые 10):' 
          % td_bad_time.train.count())
pd.set_option('display.max_colwidth', 19)
add_line(td_bad_time.sort_values('oper_time')[cols].head(10))


# <a id='time_leaps'></a>
# ## Проверка скачков по времени назад [ToC](#toc)

# In[361]:

add_header('Проверка скачков по времени назад', h=2, p=False)


# In[362]:

train_plan['next_time_start'] = train_plan.time_start.shift(-1)
train_plan['next_time_start_f'] = train_plan.time_start_f.shift(-1)
train_plan['train_end'] = train_plan.train != train_plan.train.shift(-1)
cols = ['train', 'st_from_name', 'st_to_name', 'time_start_f', 'time_end_f', 'next_time_start_f']
leaps = train_plan[(train_plan.train_end == False) & (train_plan.next_time_start < train_plan.time_end)][cols]
if leaps.empty:
    add_header('Не найдено поездов со скачками по времени назад в плане')
else:
    add_header('Всего %d поездов со скачками по времени назад в плане. Примеры:' % leaps.train.count())
    add_line(leaps.head(10)[cols])


# <a id='irk_ssp'></a>

# ## Сравнение количества передаваемых в планировщик реальных поездов с ССП [ToC](#toc)

# In[363]:

# Направления для проверки

test = [('ИРКУТСК-СОРТИРОВОЧНЫЙ', 'ГОНЧАРОВО'), ('ИРКУТСК-СОРТИРОВОЧНЫЙ', 'БАТАРЕЙНАЯ')]


# In[490]:

add_header('Детальное сравнение количества поездов с данными ССП по станции %s' % test[0][0], h=2, p=False)


# In[365]:

routes = pd.read_csv(FOLDER + 'routes.csv', dtype={'st_from':str, 'st_to':str, 'train':str})
add_info(routes)
routes['link_name'] = list(zip(routes.st_from_name, routes.st_to_name))
def_tt = pd.read_csv(FOLDER + '/mandatory/travel_times_all_pairs.csv', sep=';').set_index(['st_from_name', 'st_to_name'])


# In[366]:

def get_arrive_time(row, station):
    if (row.oper == 'depart') | (row.oper_time >= current_time):
        start_time = row.oper_time
    else:
        start_time = current_time            
    return start_time + row.tt + row.lag

(test_st, test_st_dir) = test[0]
train_info['st_loc_name'] = train_info.oper_location.map(st_names.name)
train_info.st_loc_name.fillna(train_info.st_from_name,  inplace=True)
train_info['tt'] = train_info.st_loc_name.apply(lambda x: def_tt.ix[x, test_st].tt)
train_info['lag'] = np.round((train_info.tt / 6) * 1.5)
train_info['arr_time'] = train_info[['oper', 'oper_time', 'tt', 'lag']]                                    .apply(lambda row: get_arrive_time(row, test_st), axis=1)
train_info['arr_time_f'] = train_info.arr_time.apply(nice_time)
train_info['plan_time'] = train_info.train.map(train_plan[train_plan.st_from_name == test_st].set_index('train').time_start)
train_info['plan_time_f'] = train_info.plan_time.apply(nice_time)
train_info['delta'] = train_info.plan_time - train_info.arr_time
train_info['delta_h'] = np.round((train_info.delta / 3600), 2)

train_plan['link_name'] = list(zip(train_plan.st_from_name, train_plan.st_to_name))

dir_trains = train_info[train_info.train.isin(routes[routes.link_name == (test_st, test_st_dir)].train)]
cols = ['train', 'oper', 'st_loc_name', 'tt', 'lag', 'oper_time_f', 'arr_time_f', 'plan_time_f', 'delta_h']
dir_trains = dir_trains[dir_trains.arr_time < current_time + 24 * 3600].sort_values('arr_time')
#a[cols]
add_header('Всего %d поездов, по которым ожидается проследование в направлении %s - %s в первые сутки планирования'
           % (dir_trains.train.count(), test_st, test_st_dir))


# In[367]:

no_plan = dir_trains[dir_trains.plan_time.isnull()][cols]
add_header('Всего %d поездов на направлении %s - %s, которых вообще нет в плане' % (no_plan.train.count(), test_st, test_st_dir))
add_line(no_plan)


# In[368]:

plan_day = dir_trains[dir_trains.plan_time < current_time + 24 * 3600][cols]
add_header('Всего %d поездов (%.2f%%), по которым запланировано проследование в направлении %s - %s в первые сутки'
          % (plan_day.train.count(), 100 * plan_day.train.count() / dir_trains.train.count(), test_st, test_st_dir))


# In[369]:

# sns.set(context='notebook', style='whitegrid')
# sns.set_color_codes('dark')
# plt.figure(figsize=(10, 5))
# sns.kdeplot(dir_trains[dir_trains.train.isin(plan_day.train) == False].dropna(subset=['delta']).delta / 3600, shade=True)

delta_lim = 3 * 3600
late_trains = dir_trains[dir_trains.delta > delta_lim]
late_trains.sort_values('delta', ascending=False)[cols]
add_header('Всего %d поездов (%.2f%%), у которых запланированное время проследования на участке %s - %s сильно сдвинуто вперед' 
           % (late_trains.train.count(), 100 * late_trains.train.count() / dir_trains.train.count(), test_st, test_st_dir))
add_line(late_trains.sort_values('delta', ascending=False)[cols])


# In[370]:

plan_cols = ['train', 'oper', 'oper_time_f', 'st_from_name', 'st_to_name', 'time_start_f', 'time_end_f', 'loco', 'team']
add_header('Пример плана по одному из поездов, формированием НЕ на станции %s:' % test_st)
with pd.option_context('display.max_colwidth', 15):
    add_line(train_plan[train_plan.train == late_trains[late_trains.st_loc_name != test_st].sort_values('delta', ascending=False)           .iloc[0].train][plan_cols])


# In[463]:

slot = pd.read_csv(FOLDER + 'slot.csv', dtype={'st_from':str, 'st_to':str})
add_info(slot)
slot['dt_start'] = slot.time_start.apply(datetime.datetime.fromtimestamp)


# In[485]:

test = [('ТАЙШЕТ', 'ТОРЕЯ')]
(test_st, test_st_dir) = test[0]
plan_cols = ['train', 'st_from_name', 'st_to_name', 'dt_start', 'loco', 'team']
train_plan['dt_start'] = train_plan.time_start.apply(datetime.datetime.fromtimestamp)
a = train_plan[(train_plan.st_from_name == test_st) 
           & (train_plan.st_to_name == test_st_dir) & (train_plan.train_type.isin(['2', '9']))
           & (train_plan.time_start >= current_time) & (train_plan.time_end < current_time + 24 * 3600)]\
    .sort_values('dt_start')[plan_cols]
dep_volume = a.set_index('dt_start').resample('1H', how='count').train
dep_volume


# In[486]:

slot_volume = slot[(slot.st_from_name == test_st) & (slot.st_to_name == test_st_dir)]    .set_index('dt_start').resample('1H', how='count').slot
dep_slot = dep_volume.to_frame().join(slot_volume)


# In[487]:

# slot_volume = slot[(slot.st_from_name == test_st) & (slot.st_to_name == test_st_dir)]\
#     .set_index('dt_start').resample('300s')
# slot_volume.dropna(subset=['slot'], inplace=True)
# slot_volume['slot'] = slot_volume.slot.apply(int)
# slot_volume = slot_volume.reset_index().set_index('slot')
# slot_volume


# In[488]:

a = dep_slot[dep_slot.train > dep_slot.slot]
print(a)


# In[489]:

pd.set_option('display.max_colwidth', 40)
cols = ['train', 'st_from_name', 'st_to_name', 'dt_start']
for dt in a.index:
    dt_next = dt + datetime.timedelta(0, 3600, 0)
    print(dt, dt_next)
    print(train_plan[(train_plan.train_type.isin(['2', '9']))
                    & (train_plan.st_from_name == test_st) & (train_plan.st_to_name == test_st_dir)
                    & (train_plan.dt_start >= dt) & (train_plan.dt_start < dt_next)].sort_values('dt_start')[cols])
    print(slot[(slot.st_from_name == test_st) & (slot.st_to_name == test_st_dir)
                    & (slot.dt_start >= dt) & (slot.dt_start < dt_next)].sort_values('dt_start')[['slot', 'dt_start']])
    print('-------')


# <a id='report'></a>
# ### Экспорт в HTML [ToC](#toc)

# In[371]:

filename = REPORT_FOLDER + 'train_report_' + time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + '.html'
create_report(filename)

