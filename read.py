
# coding: utf-8

# <a id='toc'></a>
# # Парсинг логов планировщика и создание csv

# 1. [Подготовка словарей для хранения данных](#dict)
#   1. [Поля сущностей](#cols)
#   2. [Словари для строк из файла, список с данными и датафреймов](#data)
# 2. [Загрузка строк из файла](#read)
# 3. [Разбор строк из файла и заполнение словаря со списками данных](#parse)
#   1. [Данные по поездам](#train)
#     1. [Атрибуты и маршруты поездов](#train_info)
#     2. [Операции с поездами (поездные факты)](#train_oper)
#   2. [Данные по локомотивам](#loco)
#     1. [Атрибуты локомотивов и тяговые плечи](#loco_attributes)
#     2. [Местоположение локомотивов (локомотивные факты)](#fact_loco)
#     3. [Время и пробег до ТО-2](#fact_loco_next_service)
#   3. [Данные по бригадам](#team)
#     1. [Атрибуты бригад](#team_attributes)
#     2. [Местоположение и состояние бригад](#fact_team_location)
#     3. [Последние явки и время начала отдыха бригад](#fact_team_ready)
#   4. [Нормативно-справочная информация](#nsi)
#     1. [Станции и пункты проведения ТО](#station)
#     2. [Участки планирования](#)
#     3. [Участки обращения бригад](#team_region)
#     4. [Участки обкатки бригад](#team_work_region)
#     5. [Весовые нормы локомотивов](#loco_tonnage)
#     6. [Задания на поезда своего формирования из ССП](#task)
#     7. [Пассажирские нитки вариантного графика](#slot_pass)
#     8. [Грузовые нитки вариантного графика](#slot)
#     9. [Вспомогательная информация (индексы поездов, номера локомотивов и бригад, названия и коды станций](#support)
#     10. [Время начала планирования](#current_time)
#   5. [Результаты планирования](#results)
#     1. [Планы по поездам](#slot_train)
#     2. [Планы по локомотивам](#slot_loco)
#     3. [Планы по бригадам](#slot_team)
# 4. [Создание датафреймов](#create_df)
# 5. [Объединение информации между датафреймами](#merge_df)
#   1. [Добавление кодов и названий станций в station](#merge_station)
#   2. [Добавление индекса и операций с поездами в train_info](#merge_train)
#   3. [Добавление номера, местоположения и времени до ТО в loco_attributes](#merge_loco)
#   4. [Добавление номера, местоположения, состояния и информации по явке в team_attributes](#merge_team)
# 6. [Выгрузка результатов в csv-файлы](#save_csv)
#   1. [Создание вспомогательного файла с названиями серий](#series)

# In[159]:

import sys
if len(sys.argv) > 1:
    if 'log_for' in sys.argv:
        file_name = 'input/log_for_analysis.log'        
    else:
        file_name = 'input/jason-FullPlannerPlugin.log'
else:
    file_name = 'input/jason-FullPlannerPlugin.log'
#file_name = 'input/log_for_analysis.log'        
print('Load data from file "%s"' % file_name)


# In[160]:

import pandas as pd
import numpy as np
import re, time


# In[161]:

start_time = time.time()
FOLDER = 'resources/'


# In[162]:



# <a id='dict'></a>
# ## Подготовка словарей для хранения данных [ToC](#toc)

# <a id='cols'></a>
# ### Поля сущностей [ToC](#toc)

# In[163]:

entities_cols = {  'link'                  :['link', 'st_from', 'st_to', 'time', 'dist', 'dir', 'lines', 'road'],
                   'station'               :['station', 'loco_region','norm_time'],
                   'station_names'         :['station', 'name', 'esr'],
                   'train_index'           :['train', 'index', 'ind434'],
                   'team_nums'             :['team', 'number'],
                   'loco_nums'             :['loco', 'number'],
                   'support'               :[],
                   'loco_tonnage'          :['series', 'sections', 'link', 'st_from', 'st_to', 'max_weight'],
                   'task'                  :['id', 'time_start', 'time_end', 'st_from', 'st_next', 'st_to', 'number'],
                   'train_info'            :['train', 'number', 'weight', 'length', 'start_st', 'end_st', 'joint'],
                   'slot_train'            :['train', 'st_from', 'st_to', 'link', 'time_start', 'time_end'],
                   'loco_attributes'       :['loco', 'series', 'regions', 'depot', 'sections', 'ltype'],
                   'slot_loco'             :['loco', 'st_from', 'st_to', 'link', 'time_start', 'time_end', 'state', 'train'],
                   'team_attributes'       :['team', 'regions', 'depot', 'long', 'heavy', 'series', 'fake', 'ttype'],
                   'fact_team_ready'       :['team', 'ready_type', 'depot_st', 'depot_time', 
                                             'return_st', 'return_time', 'rest_time'],
                   'slot_team'             :['team', 'st_from', 'st_to', 'link', 'time_start', 'time_end', 
                                             'slot', 'state', 'loco'], 
                   'slot'                  :['slot', 'st_from', 'st_to', 'link', 'time_start', 'time_end'],
                   'slot_pass'             :['slot', 'st_from', 'st_to', 'link', 'time_start', 'time_end'],
                   'routes'                :['train', 'st_from', 'st_to'],
                   'service'               :['station', 'serv_type', 'duration'],
                   'train_oper'            :['train', 'oper', 'oper_time', 'oper_location', 'st_from', 'st_to'],
                   'train_depart'          :['train', 'st_from', 'st_to', 'oper_time'],
                   'train_arrive'          :['train', 'oper_location', 'oper_time'],
                   'train_ready'           :['train', 'oper_location', 'oper_time'],
                   'fact_loco'             :['loco', 'oper', 'oper_time', 'oper_location', 
                                             'st_from', 'st_to', 'state', 'train'],
                   'fact_loco_next_service':['loco', 'dts', 'tts'],
                   'fact_team_location'    :['team', 'oper_time', 'oper_location', 'st_from', 'st_to', 'state', 'loco'],
                   'loco_info_regs'        :['loco', 'region'],
                   'team_work_region'      :['twr', 'link'],
                   'team_region'           :['team_region', 'asoup', 'depot', 'st_from', 'st_to', 'time_f', 'time_b', 'time_wr'],
                   'service_station'       :['station', 'stype', 'series', 'sections', 'ptype', 'priority', 'duration'],
                   'current_time'          :['current_time']}


# <a id='data'></a>
# ### Словари для строк из файла, списков с данными и датафреймов [ToC](#toc)

# In[164]:

entities_data = {}
for key in entities_cols.keys():
    entities_data[key] = []
    
entities_df_source = {}
for key in entities_cols.keys():
    entities_df_source[key] = []
    
entities_df = {}
for key in entities_cols.keys():
    entities_df[key] = []


# <a id='read'></a>
# ## Загрузка строк из файла [ToC](#toc)

# In[165]:

def simplecount(filename):
    lines = 0
    for line in open(filename, encoding = 'utf_8_sig'):
        lines += 1
    return lines

n = simplecount(file_name)


# In[166]:

with open(file_name, encoding = 'utf_8_sig') as f:    
    for line in f:
        functor = line[1:line.find("(")]
        functor_tell = line[5:line.find("(", 5)]
        if functor in entities_cols.keys():            
            entities_data[functor].append(line)
        elif functor_tell in entities_cols.keys():            
            entities_data[functor_tell].append(line)
        elif '=' in functor:
            entities_data['support'].append(line)

print(sorted((key, len(entities_data[key])) for key in entities_cols.keys()))


# <a id='parse'></a>
# ## Разбор строк из файла и заполнение словаря со списками данных [ToC](#toc)

# <a id='train'></a>
# ### Данные по поездам [ToC](#toc)

# <a id='train_info'></a>
# #### Атрибуты и маршруты поездов [ToC](#toc)

# In[167]:

entities_df_source['train_info'] = []
entities_df_source['routes'] = []

for line in entities_data['train_info']:
    a = line.split(',')
    if 'joint' in line:
        train, number, weight, length, joint = a[0][15:27], a[1][13:-1], a[3][7:-1], a[4][7:-1], a[-1][6:-5]
    else:
        train, number, weight, length, joint = a[0][15:27], a[1][7:-1], a[3][7:-1], a[4][7:-1], -1
    r = line.split('route')
    if len(r) > 2:
        rr = r[2]
        r1 = rr[2:rr.find(']')].split(',')            
        route = [x[8:-1] for x in r1]
        if len(route) > 0:
            start_st, end_st = route[0], route[-1]        
            route_shift = np.roll(route, -1)
            for i in range(0, len(route) - 1):
                entities_df_source['routes'].append([train, route[i], route_shift[i]])
    else:
        start_st, end_st = -1, -1
    entities_df_source['train_info'].append([train, number, weight, length, start_st, end_st, joint])


# <a id='train_oper'></a>
# #### Операции с поездами (поездные факты) [ToC](#toc)

# In[168]:

entities_df_source['train_oper'] = []

for line in entities_data['train_depart']:
    a = line.split(',')
    train, st_from, st_to, t = a[0][17:-1], a[1][14:-1], a[2][8:-2], a[3][5:-3]
    entities_df_source['train_oper'].append([train, 'depart', t, [st_from, st_to], st_from, st_to])

for line in entities_data['train_arrive']:
    a = line.split(',')
    train, st, t = a[0][17:-1], a[2][8:-2], a[3][5:-3]
    entities_df_source['train_oper'].append([train, 'arrive', t, st, -1, -1])

for line in entities_data['train_ready']:
    a = line.split(',')
    train, st, t = a[0][16:-1], a[1][8:-1], a[2][5:-3]            
    entities_df_source['train_oper'].append([train, 'ready', t, st, -1, -1])
    
#entities_df_source['train_oper'][:3]


# <a id='loco'></a>
# ### Данные по локомотивам [ToC](#toc)

# <a id='loco_attributes'></a>
# #### Атрибуты локомотивов и тяговые плечи [ToC](#toc)

# In[169]:

entities_df_source['loco_attributes'] = []
entities_df_source['loco_info_regs'] = []

for line in entities_data['loco_attributes']:
    a = line.split(',')            
    loco, series = a[0][20:32], a[1][19:-1]              
    regions_str = line[79:line.find('depot')-3].split(',')
    regions = []
    for reg in regions_str:
        regions.append(reg[3:-1])
        entities_df_source['loco_info_regs'].append([loco, reg[3:-1]])

    rest_line = line[line.find('depot'):].split(',')
    depot = rest_line[0][14:-2]            
    sections = rest_line[1][9:-1]
    if len(rest_line) > 2:
        ltype = rest_line[2][5:-5]
    else:
        ltype = 'none'
    entities_df_source['loco_attributes'].append([loco, series, regions, depot, sections, ltype])
    
#entities_df_source['loco_attributes'][:3]


# <a id='fact_loco'></a>
# #### Местоположение локомотивов (локомотивные факты) [ToC](#toc)

# In[170]:

entities_df_source['fact_loco'] = []

for line in entities_data['fact_loco']:
    a = line.split(',')
    if 'depart_time' in line:                
        loco, oper, oper_t = a[0][14:-1], 'depart', a[4][12:-1]
        location, state, train = [a[2][23:-1], a[3][8:-1]], a[5][6], a[6][6:-5]
        st_from, st_to = a[2][23:-1], a[3][8:-1]
    elif 'arrive_time' in line:                
        loco, oper, oper_t = a[0][14:-1], 'arrive', a[3][12:-1]
        location, state, train = a[2][17:-1], a[4][6], a[5][6:-4]                
        st_from, st_to = -1, -1
    else:                
        loco, oper, oper_t = a[0][14:-1], 'ready', a[1][10:-1]
        location, state, train = a[2][17:-4], -1, -1                      
        st_from, st_to = -1, -1
    entities_df_source['fact_loco'].append([loco, oper, oper_t, location, st_from, st_to, state, train])      
    
#entities_df_source['fact_loco'][:5]


# <a id='fact_loco_next_service'></a>
# #### Время и пробег до ТО-2 [ToC](#toc)

# In[171]:

entities_df_source['fact_loco_next_service'] = []

for line in entities_data['fact_loco_next_service']:
    a = line.split(',')            
    loco, dts, tts = a[0][27:-1], a[2][21:-1], a[3][8:-1]
    entities_df_source['fact_loco_next_service'].append([loco, dts, tts])
    
#entities_df_source['fact_loco_next_service'][:3]


# <a id='team'></a>
# ### Данные по бригадам [ToC](#toc)

# <a id='team_attributes'></a>
# #### Атрибуты бригад [ToC](#toc)

# In[172]:

entities_df_source['team_attributes'] = []
a = []
for line in entities_data['team_attributes']:                       
    k = re.split('(attributes|team_work_regions|depot|loco_series|long_train|heavy_train|fake|type)', line)    
    team = k[2][4:-2]
    regions_str = k[6][2:-3].split(',')
    regions = []
    for reg in regions_str:
        regions.append(reg[3:-1])                      

    depot, long, heavy, fake = k[8][9:-3], k[12][1], k[14][1], k[16][1]
    series = list(np.unique([x[3:-1] for x in k[10][2:-3].split(',')]))    
    if len(k)>18:
        ttype = k[18][1]
    else:
        ttype = 'none'
    entities_df_source['team_attributes'].append([team, regions, depot, long, heavy, series, fake, ttype])
    
#entities_df_source['team_attributes'][:3]


# <a id='fact_team_location'></a>
# #### Местоположение и состояние бригад [ToC](#toc)

# In[173]:

entities_df_source['fact_team_location'] = []

for line in entities_data['fact_team_location']:
    a = line.split(',')
    if 'location(station' in line:
        team, oper_t = a[0][23:-1], a[3][10:-1]
        location, state, loco = a[2][17:-2], a[5][6], a[4][5:-1]
        st_from, st_to = -1, -1
    else:                            
        team, oper_t = a[0][23:-1], a[4][10:-1]
        location, state, loco = [a[2][23:-1], a[3][8:-3]], a[7][6], a[5][5:-1]
        st_from, st_to = a[2][23:-1], a[3][8:-3]
    entities_df_source['fact_team_location'].append([team, oper_t, location, st_from, st_to, state, loco])
    
#print(entities_df_source['fact_team_location'][:3])


# <a id='fact_team_ready'></a>
# #### Последние явки и время начала отдыха бригад [ToC](#toc)

# In[174]:

entities_df_source['fact_team_ready'] = []

for line in entities_data['fact_team_ready']:    
    a = line.split(',')
    team = a[0][20:-1]
    depot_time, depot_st = a[1][17:-1], a[2][8:-2]            
    return_time, return_st = a[3][18:-1], a[4][8:-2]            
    ready_type = a[5][11:-1]
    rest_time = a[6][16:-3]            
    entities_df_source['fact_team_ready'].append([team, ready_type, depot_st, depot_time, return_st, return_time, rest_time])

#print(entities_df_source['fact_team_ready'][:3])


# <a id='nsi'></a>
# ### Нормативно-справочная информация [ToC](#toc)

# <a id='station'></a>
# #### Станции и пункты проведения ТО [ToC](#toc)

# In[175]:

entities_df_source['station'] = []
entities_df_source['service'] = []

for line in entities_data['station']:
    a = line.split(',')
    station, loco_region, norm_time = a[0][12:22], a[1][12:-1], a[-1][10:-3]
    entities_df_source['station'].append([station, loco_region, norm_time])
    serv = re.split('(loco_region|service|norm_reserve|norm_time)', line)[4].split('type')            
    if len(serv) > 1:
        for item in serv[1:]:
            sp = re.split('(]|,)', item)                    
            serv_type = sp[0][4:-1]
            dur = sp[2][9:-2]
            entities_df_source['service'].append([station, serv_type, dur])
            
#print(entities_df_source['service'][:3])
#print(entities_df_source['station'][:3])


# <a id='link'></a>
# #### Участки планирования [ToC](#toc)

# In[176]:

entities_df_source['link'] = []

for line in entities_data['link']:
    st_from, st_to = line[20:30], line[40:50]            
    attr = line[65:-4].split(',')
    travel_time, dist = attr[0][9:-1], attr[1][9:-1]
    dir = attr[3][10:-1]
    lines, road = attr[4][6:-1], attr[5][5:-1]
    entities_df_source['link'].append([[st_from, st_to], st_from, st_to, travel_time, dist, dir, lines, road])
    
#print(entities_df_source['link'][:3])


# <a id='team_region'></a>
# #### Участки обращения бригад (УОЛБ) [ToC](#toc)

# In[177]:

entities_df_source['team_region'] = []

for line in entities_data['team_region']:
    a = line.split('track')
    if len(a) > 2:
        b = a[0].split(',')
        tr_id = b[0][16:-1]
        asoup_id = b[1][9:-1]
        depot = b[2][6:-1]
        tracks = a[2:]
        wt = a[-1].split(',')
        time_f = wt[2][18:-1]
        time_b = wt[3][9:-1]
        time_wr = wt[4][13:-4]
        for track in tracks:
            st_from, st_to = track[9:19], track[29:39]                    
            entities_df_source['team_region'].append([tr_id, asoup_id, depot, st_from, st_to, time_f, time_b, time_wr])
            
#print(entities_df_source['team_region'][:3])


# <a id='team_work_region'></a>
# #### Участки обкатки бригад [ToC](#toc)

# In[178]:

entities_df_source['team_work_region'] = []

for line in entities_data['team_work_region']:
    a = line.split('track')
    if len(a) > 2:
        twr_id = a[0][21:-2]                
        tracks = a[2:]
        for track in tracks:
            st_from, st_to = track[9:19], track[29:39]                    
            entities_df_source['team_work_region'].append([twr_id, [st_from, st_to]])              
            
#print(entities_df_source['team_work_region'][:3])


# <a id='loco_tonnage'></a>
# #### Весовые нормы локомотивов [ToC](#toc)

# In[179]:

entities_df_source['loco_tonnage'] = []

for line in entities_data['loco_tonnage']:    
    a = line.split(',')                        
    series = a[0][21:-1]
    sections = a[1][9:-1]
    st_from = a[2][14:-1]
    st_to = a[3][8:-2]
    link = [st_from, st_to]
    max_weight = a[4][17:-3]
    entities_df_source['loco_tonnage'].append([series, sections, link, st_from, st_to, max_weight])
    
#print(entities_df_source['loco_tonnage'][:3])


# <a id='task'></a>
# #### Задания на поезда своего формирования из ССП [ToC](#toc)

# In[180]:

entities_df_source['task'] = []

for line in entities_data['task']:
    a = re.split('(id|interval|routes|weight)', line)            
    task_id = a[2][1:-2]
    time_start = a[4][1:11]
    time_end = int(time_start) + int(a[4][12:-2])
    num = int(re.split(',|\)', a[-1])[2])
    task_routes = re.split('(route|,route)', a[6][2:-3])            
    for item in task_routes[2::2]:
        task_route = item[2:-2].split(',')
        st_from = task_route[0][8:-1]
        if len(task_route) > 1:
            st_next = task_route[1][8:-1]
            st_to = task_route[-1][8:-1]
        else:
            st_next, st_to = None, None                
        entities_df_source['task'].append([task_id, time_start, time_end, st_from, st_next, st_to, num])
        
#print(entities_df_source['task'][:3])


# <a id='slot_pass'></a>
# #### Пассажирские нитки вариантного графика [ToC](#toc)

# In[181]:

entities_df_source['slot_pass'] = []

for line in entities_data['slot_pass']:    
    a = line.split('track')            
    slot_id = a[0][14:26]
    for i in range(1, len(a)):                
        st_from, st_to = a[i][9:19], a[i][29:39]
        link = [st_from, st_to]
        time_start, time_end = a[i][52:62], a[i][73:83]                                
        entities_df_source['slot_pass'].append([slot_id, st_from, st_to, link, time_start, time_end]) 
        
#print(entities_df_source['slot_pass'][:3])


# <a id='slot'></a>
# #### Грузовые нитки вариантного графика [ToC](#toc)

# In[182]:

entities_df_source['slot'] = []

for line in entities_data['slot']:
    a = line.split('track')            
    slot_id = a[0][9:21]            
    for i in range(1, len(a)):                
        st_from, st_to = a[i][9:19], a[i][29:39]
        link = [st_from, st_to]
        time_start, time_end = a[i][52:62], a[i][73:83]                                
        entities_df_source['slot'].append([slot_id, st_from, st_to, link, time_start, time_end])
        
#print(entities_df_source['slot'][:3])


# <a id='service_station'></a>
# #### Станции ПТОЛ [ToC](#toc)

# In[183]:

entities_df_source['service_station'] = []

for line in entities_data['service_station']:
    a = line.split(',')
    station = a[0][20:-1]
    stype = a[1][13:-1]
    series = a[2][7:-1]
    sections = a[3][9:-1]
    ptype = a[4][11:-1]
    priority = a[5][9:-1]
    duration = a[6][9:-3]
    entities_df_source['service_station'].append([station, stype, series, sections, ptype, priority, duration])
    
#print(entities_df_source['service_station'][:3])


# <a id='support'></a>
# #### Вспомогательная информация (индексы поездов, номера локомотивов, бригад, названия и коды станций [ToC](#toc)

# In[184]:

entities_df_source['support'] = []

for line in entities_data['support']:
    functor = line[1:line.find("(")]
    if '{' not in functor:                        
        st = line.split('=')                
        st_id = st[0][2:-1]
        st_name = st[1][1:-9]
        st_esr = st[1][-7:-2]                
        entities_df_source['station_names'].append([st_id, st_name, st_esr])
    elif len(functor.split(';')[0]) == 20:
        a = functor.split(';')[0].split('=')
        loco_id = a[0][1:-1]
        loco_num = a[1][1:]             
        entities_df_source['loco_nums'].append([loco_id, loco_num])
    elif len(functor.split(';')[0]) == 26:
        a = functor.split(';')[0].split('=')
        team_id = a[0][1:-1]
        team_num = a[1][1:]
        team_num_formatted = team_num[:4] + '-' + team_num[4:]        
        entities_df_source['team_nums'].append([team_id, team_num])
    elif len(functor.split(';')[0]) > 26:
        a = line.split(';')
        if len(a[0]) < 45:
            num_ind = a[0].split('=')
            train_id = num_ind[0][2:-1]
            ind = num_ind[1][1:]
            ind434=ind[:4] + '-' + ind[6:9] + '-' + ind[9:13]
            entities_df_source['train_index'].append([train_id, ind, ind434])
            
#print(entities_df_source['station_names'][:5])
#print(entities_df_source['loco_nums'][:5])
#print(entities_df_source['team_nums'][:5])
#print(entities_df_source['train_index'][:5])


# <a id='current_time'></a>
# #### Время начала планирования [ToC](#toc)

# In[185]:

entities_df_source['current_time'] = []
for line in entities_data['current_time']:
    if entities_df_source['current_time'] == []:
        entities_df_source['current_time'].append(line[14:-2])
    
#entities_df_source['current_time'][:3]


# <a id='results'></a>
# ### Результаты планирования [ToC](#toc)

# <a id='slot_train'></a>
# #### Планы по поездам [ToC](#toc)

# In[186]:

entities_df_source['slot_train'] = []

for line in entities_data['slot_train']:
    a = line.split('track')        
    train_id = a[0][a[0].index('id')+3:a[0].index('route')-2]                
    for i in range(1, len(a)):                
        st_from, st_to = a[i][9:19], a[i][29:39]
        link = [st_from, st_to]
        time_start, time_end = a[i][52:62], a[i][73:83]                
        entities_df_source['slot_train'].append([train_id, st_from, st_to, link, time_start, time_end])  
        
#print(entities_df_source['slot_train'][:5])


# <a id='slot_loco'></a>
# #### Планы по локомотивам [ToC](#toc)

# In[187]:

entities_df_source['slot_loco'] = []

for line in entities_data['slot_loco']:
    a = line.split('track')                    
    loco = a[0][18:30]                
    for i in range(1, len(a)):
        st_from, st_to = a[i][9:19], a[i][29:39]                
        link = [st_from, st_to]
        time_start, time_end = a[i][52:62], a[i][73:83]                
        state, train = a[i][103], a[i][112:a[i].find("))")]                
        entities_df_source['slot_loco'].append([loco, st_from, st_to, link, time_start, time_end, state, train])  
        
#print(entities_df_source['slot_loco'][:5])


# <a id='slot_team'></a>
# #### Планы по бригадам [ToC](#toc)

# In[188]:

entities_df_source['slot_team'] = []

for line in entities_data['slot_team']:
    a = line.split('track')                    
    team = a[0][18:30]                
    for i in range(1, len(a)):
        sep = a[i].split(',')                
        st_from, st_to = sep[0][9:-1], sep[1][8:-1]                
        link = [st_from, st_to]
        time_start, time_end = sep[2][11:-1], sep[3][9:-1]                
        slot_id, state, loco = sep[4][8:-1], sep[5][6], sep[6][5:sep[6].find('))')]
        entities_df_source['slot_team'].append([team, st_from, st_to, link, time_start, time_end, slot_id, state, loco])      
        
#print(entities_df_source['slot_team'][:5])


# <a id='create_df'></a>
# ## Создание датафреймов [ToC](#toc)

# In[189]:

for key in entities_df.keys():
    try:
        entities_df[key] = pd.DataFrame(entities_df_source[key], columns = entities_cols[key])  
    except:
        print('Fail to create dataframe for key:', key)

#print(sorted(entities_df.keys()))


# <a id='merge_df'></a>
# ## Объединение информации между датафреймами [ToC](#toc)

# <a id='merge_station'></a>
# #### Добавление кодов и названий станций в station [ToC](#toc)

# In[190]:

for col in entities_cols['station_names']:
    if col != 'station':
        entities_df['station'][col] = entities_df['station'].station                    .map(entities_df['station_names'].drop_duplicates('station').set_index('station')[col])


# <a id='merge_train'></a>
# #### Добавление индекса и операций с поездами в train_info [ToC](#toc)

# In[191]:

# Если по поезду в train_oper несколько операций, то оставляем только последнюю по времени
# Если последних операций тоже несколько, то оставляем в таком порядке: сначала depart - потом ready - потом arrive

def get_oper_code(x):
    if x == 'depart':
        return 2
    elif x == 'ready':
        return 1
    else:
        return 0

entities_df['train_oper']['oper_code'] = entities_df['train_oper'].oper.apply(get_oper_code)
entities_df['train_oper'].sort_values(['train', 'oper_time', 'oper_code'])
entities_df['train_oper'] = entities_df['train_oper'].sort_values(['train', 'oper_time', 'oper_code'])                                                     .drop_duplicates(subset=['train'], keep='last')                                                     .drop('oper_code', axis=1)


# In[42]:

if 'ind434' not in entities_df['train_info'].columns:
    entities_df['train_info'] = entities_df['train_info'].set_index('train')                                .join(entities_df['train_index'].set_index('train')).reset_index()
    
if 'oper_time' not in entities_df['train_info'].columns:
    entities_df['train_info'] = pd.merge(entities_df['train_info'],                                          entities_df['train_oper'].drop_duplicates('train'), 
                                         on='train')


# <a id='merge_loco'></a>
# #### Добавление номера, местоположения и времени до ТО в loco_attributes [ToC](#toc)

# In[73]:

entities_df['loco_attributes']['number'] = entities_df['loco_attributes'].loco                                        .map(entities_df['loco_nums'].drop_duplicates('loco').set_index('loco').number)

if 'oper_time' not in entities_df['loco_attributes'].columns:
    entities_df['loco_attributes'] = pd.merge(pd.merge(entities_df['loco_attributes'],
                                                       entities_df['fact_loco'], on='loco'), 
                                              entities_df['fact_loco_next_service'], 
                                              on='loco')


# <a id='merge_team'></a>
# #### Добавление номера, местоположения, состояния и информации по явке в team_attributes [ToC](#toc)

# In[74]:

entities_df['team_attributes']['number'] = entities_df['team_attributes'].team                                            .map(entities_df['team_nums'].drop_duplicates('team').set_index('team').number)

if 'oper_time' not in entities_df['team_attributes'].columns:
    entities_df['team_attributes'] = pd.merge(pd.merge(entities_df['team_attributes'], 
                                                       entities_df['fact_team_ready'], 
                                                       on='team'), 
                                              entities_df['fact_team_location'], 
                                              on='team')


# <a id='save_csv'></a>
# ## Выгрузка результатов в csv-файлы [ToC](#toc)

# In[75]:

TEST_FOLDER = 'test/'

for key in entities_df.keys():
    filename = FOLDER  + key + '.csv'
    entities_df[key].to_csv(filename, index=False, encoding='utf-8')
    print('Файл %s успешно создан' % filename)


# <a id='series'></a>
# #### Создание вспомогательного файла с названиями серий [ToC](#toc)

# In[77]:

ser_id, ser_name, ser_desc = [], [], []
cnt = 1
with open(FOLDER + 'mandatory/series_names.csv', encoding = 'utf_8_sig') as f:
    for line in f:        
        if cnt == 1:
            ser_id.append(line[:-1])
            cnt = 2
        elif cnt == 2:            
            ser_name.append(line[:-1])
            cnt = 3
        elif cnt == 3:            
            if line[:-1] != '':                
                ser_desc.append(line[:-1])
            else:                
                ser_desc.append('none')
            cnt = 1
series = pd.DataFrame(columns=['ser_id', 'ser_name', 'ser_desc', 'ser_type'])
series.ser_id, series.ser_name, series.ser_desc = ser_id, ser_name, ser_desc 
el_series = ['ЭС', 'ВЛ', 'Э5']
series['ser_type'] = series.ser_name.apply(lambda x: 'Электровоз' if any(ser in x for ser in el_series) else 'Тепловоз')
series.to_csv(FOLDER + 'loco_series.csv', index=False, encoding='utf-8')


# In[194]:

t = time.time() - start_time
print('Общее время выполнения: %.2f сек.' % t)
print('Время запуска:', time.ctime())

