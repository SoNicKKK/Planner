
# coding: utf-8

# In[15]:

get_ipython().magic('run common.py')


# In[16]:

TEST_FOLDER = './test_scenario/'
paths = pd.read_csv(FOLDER + '/mandatory/paths_id.csv', sep=';')


# In[17]:

df = pd.read_excel('./input/(Новые) 2.11.xlsx', header=1)
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


# In[18]:

def get_train_info(x):    
    st_from = int(x['oper_st'])
    st_to = int(x['st_end'])    
    route = paths[(paths.ST_FROM == st_from) & (paths.ST_TO == st_to)].ROUTE.values[0].split(',')    
    r_str = ','.join(['station(%s)' % r[:-2] for r in route])    
    return '+train_info(id(%s),info([number(%s),category(%s),weight(%s),length(%s),routes([route([%s])]),joint(-1)]))' %     (x.train_id, int(x.number), 2, int(x.weight), int(x.length), r_str)
        
a = df.apply(lambda x: get_train_info(x), axis=1)
a.to_csv(TEST_FOLDER + 'train_info.csv', index=False, sep=';')


# In[19]:

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


# In[20]:

df_loco = pd.read_excel('./input/(Новые) 2.11.xlsx', header=1, sheetname='ЛОК')
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


# In[21]:

def get_loco_info(x):
    ser_id = loco_series[loco_series.ser_name == x.ser_name].ser_id.values[0]
    reg_id = ''.join([i for i in x.regions if i.isnumeric()])
    return '+loco_attributes(id(%s),attributes([series(%s),loco_regions([id(%s)]),depot(station(%s)),sections(%s),type(1)]))' %    (x.loco_id, ser_id, reg_id, x.depot_st, int(x.sections))
        
a = df_loco.apply(lambda x: get_loco_info(x), axis=1)
a.to_csv(TEST_FOLDER + 'loco_attr.csv', index=False, sep=';')


# In[22]:

def get_loco_service(x):
    ts = int(x.oper_time.timestamp())
    return '+fact_loco_next_service(id(%s),fact_time(%s),next_service(dist_to(%s),time_to(%s),type(2001889869)))' %    (x.loco_id, ts, 1000, int(x.tts) * 3600)
        
a = df_loco.apply(lambda x: get_loco_service(x), axis=1)
a.to_csv(TEST_FOLDER + 'loco_service.csv', index=False, sep=';')


# In[23]:

depot_st_dict = {318803:89370, 319212:90320, 359276:90440, 319201:92000, 319209:92440, 319210:92710, 359271:92570}


# In[24]:

df_team = pd.read_excel('./input/(Новые) 2.11.xlsx', header=1, sheetname='ЛБР')
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


# In[25]:

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


# In[26]:

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


# In[27]:

df_team.oper_id.value_counts()


# In[28]:

states = {0:[33], 1:[2, 3], 2:[28, 30], 3:[31, 54], 4:[37], 5:[24, 26, 43], 6:[1, 42], 7:[34], 8:[35, 38], 9:[25, 41]}


# In[29]:

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


# * stations - поменять тяговые плечи
# * links - можно оставить
# * team_work_region - взять из модельного полигона
# * team_region - взять из реального полигона
# * slot, slot_pass - взять из реального полигона
# * **loco_tonnage - сделать свои**
# * **service_stations - сделать свои**

# In[58]:

big_st = stations[stations.name.isin(['ЮРТЫ', 'ТАЙШЕТ', 'ЛЕНА', 'ТАКСИМО'])].drop_duplicates('name')
big_st


# In[ ]:

paths.columns = [col.lower() for col in paths.columns]
p = paths.dropna().set_index(['st_from', 'st_to'])
p['route'] = p.route.apply(lambda x: [int(float(r)) for r in x.split(',')])


# In[54]:

# Юрты - Тайшет
print(p.ix[2000036538, 2000036518].route)

# Тайшет - Юрты
print(p.ix[2000036518, 2000036538].route)


# Тайшет - Лена
print(p.ix[2000036518, 2000036932].route)

# Лена - Тайшет
print(p.ix[2000036932, 2000036518].route)

# Лена - Таксимо
print(p.ix[2000036932, 2000036228].route)

# Таксимо - Лена
print(p.ix[2000036228, 2000036932].route)


# In[55]:

stations[stations.name.isin(['ЮРТЫ', 'ТАЙШЕТ', 'ЛЕНА', 'ТАКСИМО'])]
regs = {'ЮРТЫ':['01'], 'ТАЙШЕТ':['01', '14'], 'ЛЕНА':['14', '47'], 'ТАКСИМО':['47']}


# In[73]:

def get_station(x):
    s = '+station(id(%s),loco_region(%s),service([]),norm_reserve([norm(weight_type(0),0),norm(weight_type(1),0)]),norm_time(%s))' %        (x.station, x.loco_region, x.norm_time)
    return s

# arr = [str(x) for x in p.ix[2000036538, 2000036228].route]
# stations[stations.station.isin(arr)].drop_duplicates('name').sort_values('esr').to_excel('test_stations.xlsx')

st = pd.read_excel('test_stations.xlsx', converters={'loco_region':str})
a = st.apply(get_station, axis=1)
a.to_csv(TEST_FOLDER + 'station.csv', index=False, sep=';')

