
# coding: utf-8

# ## Создание файла с модельными входными данными
# 
# Тестовый полигон представляет из себя фрагмент сети между станциями Тайшет и Таксимо. Войтенко П.Е. была создана таблица с набором поездов, локомотивов и бригад, которые планировались к движению по этому фрагменту на сутки 30.07.2015. Данный скрипт из этой таблицы, а также из не меняющейся нормативно-справочной информации создает файл `test_scenario.log`, на котором можно запускать планировщик и проверять его работу на этом модельном сценарии.

# In[253]:

#get_ipython().magic('run common.py')


# ### Необходимые файлы
# 
# 1. Путь к таблице Войтенко П.Е. указывается в переменной `input_filename`.
# 2. Путь к папке, в которой будут храниться промежуточные файлы сценария и итоговый файл с входными данными, указывается в переменной `TEST_FOLDER`.
# 3. Дополнительные файлы, которые используются при формировании входных данных, должны лежать в папке `TEST_FOLDER`:
#   * файл `paths_id.csv` с путями между всеми станциями планирования.
#   * файл `test_stations.xlsx` со списком станций модельного полигона.
#   * файл `jason-FullPlannerPlugin.log` с реальными данными какого-либо запуска планировщика.
# 3. Сsv-файлы, полученные в результате работы скрипта `read.py`, должны лежать в папке `resources`.
# 
# ### Алгоритм работы
# 
# 1. Все данные по поездам (`train_info`, `train_depart`, `train_arrive`, `train_ready`), локомотивам (`loco_attributes`, `fact_loco_next_service`) и бригадам (`team_attributes`, `fact_team_ready`, `fact_team_location`) формируются из данных таблицы Войтенко.
# 2. Данные по станциям (сообщения `station`) берутся из файла `test_stations.xlsx`. В станциях заменяются id тяговых плеч.
# 3. Данные по участкам планирования (сообщения `link`) берутся из реальных данных и обрезаются в соответствии с тестовым полигоном.
# 4. Весовые нормы (`loco_tonnage`) и пункты ТО (`service_station`) берутся из реальных данных и обрезаются в соответствии с данными тестового полигона (остаются данные только по нужным сериям локомотивов).
# 5. Нитки (грузовые и пассажирские - `slot`, `slot_pass`) берутся из реальных данных и сдвигаются по времени с 00:00 29.07.2015 до 23:59 30.07.2015.
# 6. Участки обкатки бригад (`team_work_region`) формируются скриптом. Из данных Войтенко берутся границы участков обкатки для каждой бригады, ищутся все участки планирования между этими границами. Формируются строки team_work_region с нужными id и набором треков.
# 7. Участки обслуживания бригад (`team_region`) полностью копируются из реальных данных.

# In[254]:

TEST_FOLDER = './test_scenario/'
input_filename = './input/(Новые) 2.11.xlsx'
paths = pd.read_csv(TEST_FOLDER + 'paths_id.csv', sep=';')


# ### Формирование данных по поездам

# In[255]:

df = pd.read_excel(input_filename, header=1)
df.dropna(subset=['Номер поезда'], inplace=True)
df.drop(['поезд'], axis=1, inplace=True)
df.columns = ['train_id', 'number', 'ind', 'weight', 'length', 
              'st_first', 'st_end', 'oper_id', 'oper_time', 'oper_st', 'st_from', 'st_to', 'comm']
df['oper_time'] = df.oper_time.apply(pd.to_datetime)
df.drop(['comm'], axis=1, inplace=True)
df['st_first'] = df.st_first.map(stations.drop_duplicates('esr').set_index('esr').station)
df['st_end'] = df.st_end.map(stations.drop_duplicates('esr').set_index('esr').station)
df['oper_st'] = df.oper_st.map(stations.drop_duplicates('esr').set_index('esr').station)
df['st_from'] = df.st_from.map(stations.drop_duplicates('esr').set_index('esr').station)
df['st_to'] = df.st_to.map(stations.drop_duplicates('esr').set_index('esr').station)
df.head()


# In[256]:

paths = paths.dropna(how='any')
paths


# In[257]:

def get_train_info(x, paths):    
    st_from = int(x['oper_st'])
    st_to = int(x['st_end'])
    try:
        route = paths[(paths.ST_FROM == st_from) & (paths.ST_TO == st_to)].ROUTE.values[0].split(',')    
    except:
        print(x, st_from, st_to, paths[(paths.ST_FROM == st_from) & (paths.ST_TO == st_to)].columns)
        route = ''
    r_str = ','.join(['station(%s)' % r[:-2] for r in route])    
    return '+train_info(id(%s),info([number(%s),category(%s),weight(%s),length(%s),routes([route([%s])]),joint(-1)]))' %     (x.train_id, int(x.number), 2, int(x.weight), int(x.length), r_str)
        
a = df.apply(lambda x: get_train_info(x, paths), axis=1)
a.to_csv(TEST_FOLDER + 'train_info.csv', index=False, sep=';')


# In[258]:

import datetime as dt
def get_oper(x):
    ts = int(x.oper_time.timestamp())
    if x.oper_id == 2:        
        s = '+train_depart(id(%s),track(station(%s),station(%s)),time(%s))' % (x.train_id, x.st_from, x.st_to, ts)
    elif x.oper_id == 5:
        s = '+train_ready(id(%s),station(%s),time(%s))' % (x.train_id, x.oper_st, ts)
    else:
        s = '+train_arrive(id(%s),track(station(%s),station(%s)),time(%s))' % (x.train_id, x.st_from, x.st_to, ts)
    return s        
    
a = df.apply(lambda x: get_oper(x), axis=1)
a.to_csv(TEST_FOLDER + 'train_oper.csv', index=False, sep=';')


# ### Формирование данных по локомотивам

# In[259]:

df_loco = pd.read_excel(input_filename, header=1, sheetname='ЛОК')
df_loco.dropna(subset=['Борт НОМЕР'], inplace=True)
df_loco.drop(['ЛОК'], axis=1, inplace=True)
#print(df_loco.head())
df_loco.columns = ['loco_id', 'number', 'ser_name', 'depot_st', 'sections', 
              'regions', 'tts', 'dts', 'tr_dts', 'oper', 'oper_time', 'oper_st', 'st_from', 'st_to', 'train_id', 'comm', 'sokr']
df_loco['oper_time'] = df_loco.oper_time.apply(pd.to_datetime)
df_loco.drop(['comm', 'sokr'], axis=1, inplace=True)
df_loco['depot_st'] = df_loco.depot_st.map(stations.drop_duplicates('esr').set_index('esr').station)
df_loco['oper_st'] = df_loco.oper_st.map(stations.drop_duplicates('esr').set_index('esr').station)
df_loco['st_from'] = df_loco.st_from.map(stations.drop_duplicates('esr').set_index('esr').station)
df_loco['st_to'] = df_loco.st_to.map(stations.drop_duplicates('esr').set_index('esr').station)
df_loco.head()


# In[260]:

def get_loco_info(x):
    ser_id = loco_series[loco_series.ser_name == x.ser_name].ser_id.values[0]
    reg_id = ''.join([i for i in x.regions if i.isnumeric()])
    return '+loco_attributes(id(%s),attributes([series(%s),loco_regions([id(%s)]),depot(station(%s)),sections(%s),type(1)]))' %    (x.loco_id, ser_id, reg_id, x.depot_st, int(x.sections))
        
a = df_loco.apply(lambda x: get_loco_info(x), axis=1)
a.to_csv(TEST_FOLDER + 'loco_attr.csv', index=False, sep=';')


# In[261]:

def get_loco_service(x):
    ts = int(x.oper_time.timestamp())
    return '+fact_loco_next_service(id(%s),fact_time(%s),next_service(dist_to(%s),time_to(%s),type(2001889869)))' %    (x.loco_id, ts, 1000, int(x.tts) * 3600)
        
a = df_loco.apply(lambda x: get_loco_service(x), axis=1)
a.to_csv(TEST_FOLDER + 'loco_service.csv', index=False, sep=';')


# ### Формирование данных по бригадам

# In[262]:

'''
В таблице Войтенко приведены коды депо приписок бригад. Депо, в общем случае, отличается от станций планирования, используемых
в планировщике. Поэтому надо установить соответствие между ЕСР-кодом депо и ЕСР-кодом соответствующей (ближайшей) станции 
планирования.
'''

depot_st_dict = {318803:89370, 319212:90320, 359276:90440, 319201:92000, 319209:92440, 319210:92710, 359271:92570}


# In[263]:

df_team = pd.read_excel(input_filename, header=1, sheetname='ЛБР')
df_team.dropna(subset=['Таб.номер'], inplace=True)
df_team.drop(['ЛБР'], axis=1, inplace=True)
df_team.columns = ['team_id', 'depot_st', 'ready_st', 'name', 'number', 
                  'ser_name', 'long', 'heavy', 'region', 'ready_time', 'oper_id', 'oper_time', 'oper_st', 'st_from', 'st_to',
                  'train_id', 'loco_id', 'depot_ready_time', 'depot_ready_st', 'return_ready_time', 'return_ready_st', 
                  'rest_start_time']
df_team['oper_time'] = df_team.oper_time.apply(pd.to_datetime)
df_team['depot_ready_time'] = df_team.depot_ready_time.apply(lambda x: pd.to_datetime(x) if x !=0 else -1)
df_team['return_ready_time'] = df_team.return_ready_time.apply(lambda x: pd.to_datetime(x) if x !=0 else -1)
df_team['rest_start_time'] = df_team.rest_start_time.apply(lambda x: pd.to_datetime(x) if x !=0 else -1)

df_team['oper_st'] = df_team.oper_st.map(stations.drop_duplicates('esr').set_index('esr').station)
df_team['st_from'] = df_team.st_from.map(stations.drop_duplicates('esr').set_index('esr').station)
df_team['st_to'] = df_team.st_to.map(stations.drop_duplicates('esr').set_index('esr').station)
df_team['depot_ready_st'] = df_team.depot_ready_st.map(stations.drop_duplicates('esr').set_index('esr').station)
df_team['return_ready_st'] = df_team.return_ready_st.map(stations.drop_duplicates('esr').set_index('esr').station)
df_team['depot_ready_st'].fillna(-1, inplace=True)
df_team['return_ready_st'].fillna(-1, inplace=True)

df_team.depot_st.replace(depot_st_dict, inplace=True)
df_team.ready_st.replace(depot_st_dict, inplace=True)
df_team['depot_st'] = df_team.depot_st.map(stations.drop_duplicates('esr').set_index('esr').station)
df_team['ready_st'] = df_team.ready_st.map(stations.drop_duplicates('esr').set_index('esr').station)

df_team.train_id.replace(0, -1, inplace=True)
df_team.loco_id.replace(0, -1, inplace=True)
df_team.head()


# In[264]:

def get_team_info(x):
    ser_list = x.ser_name.split('; ')
    try:
        ser_id = loco_series[loco_series.ser_name.isin(ser_list)].ser_id.unique()
    except:
        ser_id = -1
    series = ','.join(['id(%s)' % s for s in ser_id])
    reg_id = ''.join([i for i in x.region if i.isnumeric()])
    return '+team_attributes(id(%s),attributes([team_work_regions([id(%s)]),depot(station(%s)),loco_series([%s]),long_train(%s),heavy_train(%s),fake(1),type(1)]))' %    (x.team_id, reg_id, x.ready_st, series, int(x.long), int(x.heavy))
        
a = df_team.apply(lambda x: get_team_info(x), axis=1)
a.to_csv(TEST_FOLDER + 'team_attr.csv', index=False, sep=';')


# In[265]:

def get_team_ready(x):
    try:
        depot_ts = int(x.depot_ready_time.timestamp()) if x.depot_ready_time != -1 else -1
        return_ts = int(x.return_ready_time.timestamp())  if x.return_ready_time != -1 else -1
        rest_start_ts = int(x.rest_start_time.timestamp())  if x.rest_start_time != -1 else -1
    except:
        depot_ts, return_ts, rest_start_ts = -1, -1, -1
        print(x.team_id, x.depot_ready_time, x.return_ready_time, x.rest_start_time)
    last_ready = 'depot' if depot_ts < return_ts else 'return'
    return '+fact_team_ready(id(%s),ready_depot(time(%s),station(%s)),ready_return(time(%s),station(%s)),last_ready(%s),rest_start_time(%s))' %    (x.team_id, x.depot_ready_st, depot_ts, x.return_ready_st, return_ts, last_ready, rest_start_ts)
        
a = df_team.apply(lambda x: get_team_ready(x), axis=1)
a.to_csv(TEST_FOLDER + 'team_ready.csv', index=False, sep=';')


# In[266]:

states = {0:[33], 1:[2, 3], 2:[28, 30], 3:[31, 54], 4:[37], 5:[24, 26, 43], 6:[1, 42], 7:[34], 8:[35, 38], 9:[25, 41]}


# In[267]:

def get_team_location(x):
    oper_time_ts = int(x.oper_time.timestamp())
    try:
        state = [key for key, value in states.items() if x.oper_id in value][0]
    except:
        state = -1
        print(x.oper_id)
        
    if state in [0, 1]:
        s = '+fact_team_location(id(%s),fact_time(%s),location(track(station(%s),station(%s))),oper_time(%s),loco(%s),pass_slot(-1),state(%s))' %            (x.team_id, oper_time_ts, x.st_from, x.st_to, oper_time_ts, int(x.loco_id), state)
    else:
        s = '+fact_team_location(id(%s),fact_time(%s),location(station(%s)),oper_time(%s),loco(%s),state(%s))' %            (x.team_id, oper_time_ts, x.oper_st, oper_time_ts, int(x.loco_id), state)
    return s
        
a = df_team.apply(lambda x: get_team_location(x), axis=1)
a.to_csv(TEST_FOLDER + 'team_location.csv', index=False, sep=';')


# ### Формирование данных по станциям

# In[268]:

big_st = stations[stations.name.isin(['ЮРТЫ', 'ТАЙШЕТ', 'ЛЕНА', 'ТАКСИМО'])].drop_duplicates('name')
big_st


# In[269]:

paths.columns = [col.lower() for col in paths.columns]
p = paths.dropna().set_index(['st_from', 'st_to'])
p['route'] = p.route.apply(lambda x: [int(float(r)) for r in x.split(',')])


# In[270]:

stations[stations.name.isin(['ЮРТЫ', 'ТАЙШЕТ', 'ЛЕНА', 'ТАКСИМО'])]
regs = {'ЮРТЫ':['01'], 'ТАЙШЕТ':['01', '14'], 'ЛЕНА':['14', '47'], 'ТАКСИМО':['47']}


# In[271]:

def get_station(x):
    s = '+station(id(%s),loco_region(%s),service([]),norm_reserve([norm(weight_type(0),0),norm(weight_type(1),0)]),norm_time(%s))' %        (x.station, x.loco_region, x.norm_time)
    return s

# arr = [str(x) for x in p.ix[2000036538, 2000036228].route]
# stations[stations.station.isin(arr)].drop_duplicates('name').sort_values('esr').to_excel('test_stations.xlsx')

st = pd.read_excel(TEST_FOLDER + 'test_stations.xlsx', converters={'loco_region':str, 'station':str})
a = st.apply(get_station, axis=1)
a.to_csv(TEST_FOLDER + 'station.csv', index=False, sep=';')


# ### Формирование данных по участкам планирования

# In[272]:

def get_links(x):
    return '+link(track(station(%s),station(%s)),attributes([duration(%s),distance(%s),push(0),direction(%s),lines(%s),road(%s)]))'         % (x.st_from, x.st_to, x.time, x.dist, x['dir'], x.lines, x.road)

links = pd.read_csv(FOLDER + 'link.csv', dtype={'st_from':str, 'st_to':str})
l = links[(links.st_from.isin(st.station)) & (links.st_to.isin(st.station))]
l.apply(get_links, axis=1).to_csv(TEST_FOLDER + 'link.csv', index=False, sep=';')


# ### Формирование данных по пунктам ТО

# In[273]:

def get_service_station(x):
    return '+service_station(id(%s),service_type(%s),series(%s),sections(%s),power_type(%s),priority(%s),duration(%s))'         % (x.station, x.stype, x.series, x.sections, x.ptype, x.priority, x.duration)

ss = pd.read_csv(FOLDER + 'service_station.csv', dtype={'station':str})
ss['name'] = ss.station.map(st_names.name)
ss['ser_name'] = ss.series.map(loco_series.set_index('ser_id').ser_name)
ss1 = ss[(ss.station.isin(st.station))].copy()
ss1.apply(get_service_station, axis=1).to_csv(TEST_FOLDER + 'service_station.csv', index=False, sep=';')


# ### Формирование данных по участкам обслуживания бригад

# In[274]:

tr = []
with open('./input/jason-FullPlannerPlugin.log', encoding='utf-8') as f:
    prefixes = ['+team_region']
    for line in f:
        if any([x in line for x in prefixes]):
            tr.append(line)
            
with open(TEST_FOLDER + 'from_real_data.csv', 'w') as fw:
    for line in tr:
        fw.write(line)
        
fw.close()


# ### Формирование данных по весовым нормам

# In[275]:

def get_loco_tonnage(x):
    return '+loco_tonnage(series(%s),sections(%s),track(station(%s),station(%s)),max_train_weight(%s))'             % (x.series, x.sections, x.st_from, x.st_to, x.max_weight)

loco_tonnage = pd.read_csv(FOLDER + 'loco_tonnage.csv', dtype={'st_from':str, 'st_to':str})
add_info(loco_tonnage)
loco_tonnage['ser_name'] = loco_tonnage.series.map(loco_series.set_index('ser_id').ser_name)
lt = loco_tonnage[(loco_tonnage.ser_name.apply(lambda x: any([i in str(x) for i in ['ВЛ80Р', 'ЭС5К']])))
            & (loco_tonnage.st_from.isin(st.station)) & (loco_tonnage.st_to.isin(st.station))]
lt.apply(get_loco_tonnage, axis=1).to_csv(TEST_FOLDER + 'loco_tonnage.csv', index=False, sep=';')


# ### Формирование данных по участкам обкатки бригад

# In[276]:

def get_region_borders(x):
    x_num = [i for i in x if i.isnumeric()]
    twr_id = ''.join(x_num)
    st_from = st_dict[st_numbers[int(x_num[0])]]
    st_to = st_dict[st_numbers[int(x_num[1])]]
    return pd.Series([twr_id, st_from, st_to])

def get_tracks(x):
    res = []
    for i in range(len(x) - 1):
        s = 'track(station(%s),station(%s),attributes([]))' % (x[i], x[i+1])
        res.append(s)
        
    return ','.join(res)

def get_team_work_region(x):
    return '+team_work_region(id(%s),tracks([%s]),work_time(with_rest(0), without_rest(0)))'         % (x.twr_id, x.route + ',' + x.route_rev)

import networkx as nx
g = nx.DiGraph()
g.add_nodes_from(st.station.unique())
g.add_weighted_edges_from(list(zip(l.st_from, l.st_to, l.time)))

st_numbers = {0:'ЮРТЫ', 1:'ТАЙШЕТ', 2:'ВИХОРЕВКА', 3:'КОРШУНИХА-АНГАРСКАЯ', 4:'ЛЕНА', 5:'СЕВЕРОБАЙКАЛЬСК',
              6:'НОВЫЙ УОЯН', 7:'ТАКСИМО'}
st_dict = st[['name', 'station']].set_index('name').to_dict()['station']
df_team[['twr_id', 'team_reg_start', 'team_reg_end']] = df_team.region.apply(get_region_borders)
tr = df_team[['twr_id', 'team_reg_start', 'team_reg_end']].drop_duplicates()
tr['route'] = tr.apply(lambda row: nx.dijkstra_path(g, row.team_reg_start, row.team_reg_end), axis=1)
tr['route_rev'] = tr.apply(lambda row: nx.dijkstra_path(g, row.team_reg_end, row.team_reg_start), axis=1)
tr['route'] = tr.route.apply(get_tracks)
tr['route_rev'] = tr.route_rev.apply(get_tracks)
tr.apply(get_team_work_region, axis=1).to_csv(TEST_FOLDER + 'team_work_region.csv', index=False, sep=';')


# ### Формирование данных по ниткам

# In[277]:

def change_slot_times(df):    
    min_dt = dt.datetime.fromtimestamp(slot.time_start.min())
    td = dt.datetime(min_dt.year, min_dt.month, min_dt.day) - dt.datetime(2015, 7, 29)
    df['nt_start'] = df.time_start - td.days * 86400
    df['nt_end'] = df.time_end - td.days * 86400
    df['track'] = df.apply(lambda row: 'track(station(%s),station(%s),time_start(%s),time_end(%s))'
                              % (row.st_from, row.st_to, row.nt_start, row.nt_end), axis=1)
    df['tracks'] = slot.slot.map(slot.groupby('slot').track.unique().apply(lambda x: ','.join(x)))
    return df    
    
slot = pd.read_csv(FOLDER + 'slot.csv')
slot = change_slot_times(slot)
slot.drop_duplicates('slot').apply(lambda row: '+slot(id(%s),category(0),route([%s]))' % (row.slot, row.tracks), axis=1)    .to_csv(TEST_FOLDER + 'slot.csv', sep=';', index=False)

slot_pass = pd.read_csv(FOLDER + 'slot_pass.csv')
slot_pass = change_slot_times(slot_pass)
slot_pass.drop_duplicates('slot').apply(lambda row: '+slot_pass(id(%s),category(0),route([%s]))' 
                                        % (row.slot, row.tracks), axis=1)\
    .to_csv(TEST_FOLDER + 'slot_pass.csv', sep=';', index=False)


# ### Добавление заголовков и создание итогового файла

# In[279]:

import os
files = [files for root, directories, files in os.walk('./test_scenario/') ][0]
files = [f for f in files if ('.csv' in f) & ('paths_id.csv' != f)]
full = []
for filename in sorted(files):
    with open('./test_scenario/' + filename, encoding='utf-8') as f:
        try:
            for line in f:
                full.append(line)
        except:
            print(f)

ct = int(dt.datetime(2015, 7, 29, 18, 0, 0).timestamp())
start_line = '+current_time(%s)\n+config("bulk_planning",0)\n' % ct
end_line = '+current_id(%s,1)' % ct
header = []
head_st = list(st.drop_duplicates('station')               .apply(lambda row: '  %s = %s (%s)' % (row.station, row['name'], row.esr), axis=1).values)
head_train = list(df.apply(lambda row: '  %s = %s; %s; {"АСОУП"=>[]}' 
         % (row.train_id, row.ind[:4]+'01'+row.ind[5:8]+row.ind[9:]+'01', int(row.number)), axis=1).values)
head_loco = list(df_loco.apply(lambda row: '  %s = %s; {"АСОУП"=>""}' % (row.loco_id, int(row.number)), axis=1).values)
head_team = list(df_team.apply(lambda row: '  %s = %s; {}' 
                               % (row.team_id, ''.join([x for x in row.number if x.isnumeric()])), axis=1).values)
header = head_st + head_train + head_loco + head_team  

with open('./test_scenario/test_scenario.log', 'w', encoding='utf-8') as fw_res:
    for x in header:
        fw_res.write(x + '\n')
    fw_res.write(start_line)
    for line in full:
        fw_res.write(line)
    fw_res.write(end_line)    

fw_res.close()
print('Файл test_scenario.log создан')

