
# coding: utf-8

# In[542]:

get_ipython().magic('run common.py')


# In[543]:

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


# In[544]:

all_lengths = pd.read_csv(FOLDER + '/mandatory/travel_times_all_pairs.csv', sep=';').set_index(['st_from_name', 'st_to_name'])
    
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


# In[545]:

#print(nice_time(current_time + 20*60 + 3*3600))
nct = current_time + 20*60 + 3*3600
st1 = 'ИРКУТСК-СОРТИРОВОЧНЫЙ'
start_st = team_plan[team_plan.state.isin([0, 1])].drop_duplicates('team').set_index('team')
team_plan['team_type'] = team_plan.team.apply(lambda x: int(str(x)[0]))
team_plan['start_st'] = team_plan.team.map(start_st.st_from)
team_plan.depot.fillna(team_plan.start_st, inplace=True)
team_plan['depot_name'] = team_plan.depot.map(st_names.name)
cols = ['team', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'state', 'loco']
a = team_plan[(team_plan.time_start >= nct) & (team_plan.time_start < nct + 24 * 3600)
         & (team_plan.state.isin([0, 1])) & (team_plan.st_from_name == st1) & (team_plan.depot_name == st1)]
print(a.st_to_name.value_counts())
print(a.state.value_counts())
print(a.team_type.value_counts())


# In[546]:

log.head()


# In[547]:

print(nice_time(current_time))
cols = ['team', 'st_from_name', 'st_to_name', 'oper_time_f', 'time_start_norm', 'state', 'wait_ct', 'wait']
team_plan['sinfo'] = team_plan['state_info']
team_plan['oper_time_f'] = team_plan.oper_time.apply(nice_time)
team_plan['wait_ct'] = np.round((current_time - team_plan.oper_time) / 3600, 2)
team_plan['wait'] = np.round((team_plan.time_start - team_plan.oper_time) / 3600, 2)
a = team_plan[(team_plan.state_info == '2') 
          & (team_plan.state.isin([0, 1]))][cols].drop_duplicates('team').sort_values('wait', ascending=False)

(a.wait - a.wait_ct).describe()


# In[548]:

st = 'ИРКУТСК-СОРТИРОВОЧНЫЙ'

# trains
train_plan['train_type'] = train_plan.train.apply(lambda x: int(str(x)[0]))
trains = train_plan[(train_plan.time_start >= current_time) & (train_plan.time_start < current_time + 24 * 3600)
          & (train_plan.train_type.isin([2, 9]))
          & (train_plan.st_from_name == st)]

print('Trains:')
print(trains.st_to_name.value_counts().to_string(), '\n-')
print(trains.train_type.value_counts().to_string())

locos = loco_plan[(loco_plan.time_start >= current_time) & (loco_plan.time_start < current_time + 24 * 3600)          
          & (loco_plan.st_from_name == st)]
print('\nLocos:')
print(locos[locos.state == 1].st_to_name.value_counts().to_string(), '\n-')
print(locos.state.value_counts().to_string())

team_plan['team_type'] = team_plan.team.apply(lambda x: int(str(x)[0]))
team_plan['start_st'] = team_plan.team.map            (team_plan[team_plan.state.isin([0, 1])].drop_duplicates('team').set_index('team').st_from)
team_plan.depot.fillna(team_plan.start_st, inplace=True)
team_plan['depot_name'] = team_plan.depot.map(st_names.name)
teams = team_plan[(team_plan.time_start >= current_time) & (team_plan.time_start < current_time + 24 * 3600)          
          & (team_plan.st_from_name == st)]
print('\nTeams:')
print(teams[teams.state == 1].st_to_name.value_counts().to_string(), '\n-')
print(teams[teams.state == 1].team_type.value_counts().to_string(), '\n-')
print(teams[teams.state.isin([0, 1])].state.value_counts().to_string(), '\n-')
print(teams[teams.state.isin([0, 1])].depot_name.value_counts().to_string())


# In[549]:

all_lengths['dist'] = np.round(all_lengths.tt * 45 / 3600)


# In[550]:

routes = pd.read_csv(FOLDER + 'routes.csv', dtype={'st_from':str, 'st_to':str, 'train':str})
add_info(routes)
routes['first_st'] = routes.st_from_name
routes['end_st'] = routes.train.map(routes.drop_duplicates('train', keep='last').set_index('train').st_to_name)
rs = routes.drop_duplicates('train').copy(deep=True)
rs['route'] = rs.apply(lambda row: (row.first_st, row.end_st), axis=1)
rs['tt'] = rs.route.map(lambda x: all_lengths.ix[x[0], x[1]].tt)
rs['dist'] = rs.route.map(lambda x: all_lengths.ix[x[0], x[1]].dist)


# In[551]:

print(nice_time(current_time))
train_plan['train_time'] = list(zip(train_plan.train, train_plan.time_start))
loco_plan['train_time'] = list(zip(loco_plan.train, loco_plan.time_start))
train_plan['loco'] = train_plan.train_time.map(loco_plan.drop_duplicates('train_time').set_index('train_time').loco)
train_plan['route'] = train_plan.train.map(rs.set_index('train').route)
train_plan['tt'] = train_plan.train.map(rs.set_index('train').tt)
train_plan['dist'] = train_plan.train.map(rs.set_index('train').dist)
cols = ['train','weight', 'st_from_name', 'st_to_name', 'time_start_norm', 'loco', 'route', 'tt', 'dist']
no_loco = train_plan[(train_plan.time_start < current_time + 24 * 3600) & (train_plan.loco.isnull())]
no_loco.drop_duplicates('train').st_from_name.value_counts().head(10)


# In[552]:

st_name = 'КАРЫМСКАЯ'
tr = no_loco.drop_duplicates('train')
cols = ['train','weight', 'st_from_name', 'st_to_name', 'time_start_norm', 'loco', 'route']
print(tr[tr.st_from_name == st_name].sort_values('time_start')[cols].to_string(index=False))


# In[553]:

loco_cols = ['loco', 'regions', 'ser_name', 'sections', 'loc_name', 'oper_time_f', 'first_train', 'tts', 'dts']
loco_info['ser_name'] = loco_info.series.map(loco_series.set_index('ser_id').ser_name)
loco_info['oper_time_f'] = loco_info.oper_time.map(nice_time)
loco_starts = loco_plan.drop_duplicates('loco').set_index('loco')
loco_info['first_train'] = loco_info.loco.map(loco_starts.train)
with pd.option_context('display.max_colwidth', 30):
    print(loco_info[loco_info.loc_name == st_name][loco_cols].to_string(index=False))


# In[554]:

loco_plan['ser_name'] = loco_plan.series.map(loco_series.set_index('ser_id').ser_name)
loco_plan['loco_end'] = loco_plan.loco != loco_plan.loco.shift(-1)
loco_plan['next_train'] = loco_plan.train.shift(-1)
loco_plan.loc[loco_plan.loco_end == True, 'next_train'] = -1
loco_ends = loco_plan.drop_duplicates(subset=['loco', 'train'], keep='last')
loco_cols = ['loco', 'regions', 'ltype', 'ser_name', 'sections', 'train', 'time_end_norm', 'next_train', 'tts', 'dts']
loco_ends[(loco_ends.st_to_name == st_name)
         & (loco_ends.time_end < current_time + 24 * 3600)
         & (loco_ends.ltype == 1)
         & (loco_ends.next_train == -1)].sort_values('time_end')[loco_cols]


# In[555]:

loco_cols = ['loco', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'train']
loco_plan[loco_plan.loco == '200200094024'][loco_cols]


# In[556]:

loco_cols = ['loco', 'st_from_name', 'st_to_name', 'time_start_norm', 'train', 'all_trains']
loco_plan['all_trains'] = loco_plan.loco.map(loco_plan.groupby('loco').train.unique())
a = loco_plan.drop_duplicates(['loco', 'train'])
b = a[(a.time_start < current_time + 24 * 3600) 
    & (a.state == 0) & (a.st_from_name == st_name)][loco_cols]


# In[557]:

def is_useful(train, all_trains):
    if train == all_trains[-1]:
        return False
    else:
        i = list(all_trains).index(train)
        next_tr = all_trains[i + 1]
        if next_tr[0] in ['2', '9']:
            return True
        else:
            return False

#b['is_last'] = b[['train', 'all_trains']].apply(lambda row: row.all_trains[-1] == row.train, axis=1)
b['is_useful'] = b[['train', 'all_trains']].apply(lambda row: is_useful(row.train, row.all_trains), axis=1)
b[b.is_useful == False].st_to_name.value_counts()


# In[558]:

routes['link_name'] = list(zip(routes.st_from_name, routes.st_to_name))
a = routes[routes.train == '200020313247'].link_name.unique()


# In[559]:

loco_tonnage = pd.read_csv(FOLDER + 'loco_tonnage.csv', dtype={'st_from':str, 'st_to':str})
add_info(loco_tonnage)
loco_tonnage['link_name'] = list(zip(loco_tonnage.st_from_name, loco_tonnage.st_to_name))
b = loco_tonnage[loco_tonnage.link_name.isin(a)]


# In[560]:

loco_tonnage['ser_name'] = loco_tonnage.series.map(loco_series.set_index('ser_id').ser_name)
loco_tonnage.dropna(subset=['st_from_name', 'st_to_name'], how='any', inplace=True)
b = loco_tonnage.groupby(['link_name', 'ser_name', 'sections']).max_weight.unique().reset_index()
b['ln_cat'] = pd.Categorical(b.link_name, categories=a, ordered=True)
c = b[(b.ln_cat.notnull()) & b.ser_name.apply(lambda x: ('ВЛ8' in x) | ('ЭС5К' in x))].sort_values('ln_cat')


# In[561]:

c['very_max'] = c.max_weight.apply(lambda x: x.max())
c[c.very_max > 6000].sort_values(['ln_cat', 'very_max'])


# 1. Тайшет-Зима-Китой: 6300 только для ВЛ80Р
# 2. Большой Луг-Андриановская, Андриановская-Ангасолка: заданы нормы только для тепловозов и ВЛ60.

# In[562]:

cols = ['link_name', 'ser_name', 'sections', 'max_weight']
loco_tonnage[(loco_tonnage.st_from_name == 'Андриановская'.upper()) & (loco_tonnage.st_to_name == 'Ангасолка'.upper())]    .sort_values('ser_name')[cols]


# In[563]:

prst = pd.read_csv(FOLDER + 'mandatory/priority_team_change_stations.csv', sep=';')
prst['nearest'] = prst.name.apply(lambda x: all_lengths[all_lengths > 0].ix[x].tt.min())
prst.nearest.mean()


# In[564]:

team_region = pd.read_csv(FOLDER + 'team_region.csv')
team_region.time_f.mean() / 3600 - 2


# ### Поезда, которые по номеру являются локомотивами резервом, но индекс начинается не с 0001

# In[565]:

train_info['oper_time_f'] = train_info.oper_time.apply(nice_time)
train_info['ind_start'] = train_info.ind434.apply(lambda x: x[:4])
train_info['loco'] = train_info.train.map(loco_info.groupby('train').loco.unique())
train_info['lstate'] = train_info.train.map(loco_info.groupby('train').state.unique())
train_info['lstate'] = train_info.lstate.apply(lambda x: x if np.isnan(x) else x[0])
train_info['loco'] = train_info.loco.apply(lambda x: x if type(x) == float else (x[0] if len(x) == 1 else x))
train_info['in_plan'] = train_info.train.isin(train_plan.train)
train_info['loco_p'] = train_info.train.map(loco_plan.sort_values('time_start').drop_duplicates('train').set_index('train').loco)


# In[566]:

cols = ['train', 'number', 'ind434', 'loc_name', 'loco', 'ltype', 'lstate', 'loco_p']
train_res = train_info[(train_info.number >= 4000) 
                       & (train_info.number < 5000)]
train_res['ltype'] = train_res.loco.map(loco_info.set_index('loco').ltype)
#nice_print(train_res[train_res.ind_start != '0001'].sort_values('number')[cols], num=True)
#nice_print(train_res[train_res.in_plan == False][cols], num=True)
nice_print(train_res[train_res.loco_p.isnull()].sort_values('number')[cols], num=True)
#nice_print(train_res[cols], num=True)


# In[567]:

#pd.set_option('display.width', 500)
print(train_res[cols].ltype.value_counts())
train_res[(train_res.ltype == 1) & (train_res.loco == train_res.loco_p)][cols].head()


# In[568]:

cols = ['loco', 'ltype', 'oper_time_f', 'loc_name', 'train', 'train_number', 'state']
loco_info['train_number'] = loco_info.train.map(train_info.set_index('train').number)
nice_print(loco_info[(loco_info.ltype == 0) 
                      & (loco_info.train == '-1')                     
                      & (loco_info.st_from == '-1')].sort_values('train_number')[cols])


# In[569]:

slot = pd.read_csv(FOLDER + 'slot.csv', dtype={'st_from':str, 'st_to':str})
add_info(slot)
slot_routes = slot.drop_duplicates('slot')[['slot', 'st_from_name', 'time_start']].set_index('slot')        .join(slot.drop_duplicates('slot', keep='last')[['slot', 'st_to_name', 'time_end']].set_index('slot'))
slot_routes = slot_routes[['st_from_name', 'st_to_name', 'time_start', 'time_end']]


# In[570]:

ns = [9205003038, 9205004359]
team_info['oper_time_f'] = team_info.oper_time.apply(nice_time)
print(team_info[team_info.number.isin(ns)][['team', 'number', 'ttype', 'oper_time_f', 'loc_name', 'state']])


# In[571]:

loco_info[loco_info.loco == '200200082272'][['loco', 'number', 'ser_name', 'regions', 'loc_name', 'oper_time_f', 'state', 'train', 'ltype']]


# In[572]:

loco_plan[loco_plan.loco == '200200082272'][['loco', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'train']]


# In[579]:

d = datetime.datetime.fromtimestamp(current_time)
#team_plan[team_plan.time_start < ]
day_start = datetime.datetime(d.year, d.month, d.day, 18, 0, 0)
day_start


# In[582]:

day_next = day_start + datetime.timedelta(1)


# In[583]:

team_plan['dt_start'] = team_plan.time_start.apply(datetime.datetime.fromtimestamp)


# In[599]:

st_name = 'ИРКУТСК-СОРТИРОВОЧНЫЙ'
team_plan[(team_plan.dt_start >= day_start) 
          & (team_plan.dt_start < day_next)
          & (team_plan.st_from_name == st_name) & (team_plan.state.isin([0, 1]))
         & (team_plan.depot_name == st_name)
         & (team_plan.st_to_name == 'БАТАРЕЙНАЯ')][['team', 'st_from_name', 'st_to_name', 'dt_start', 'loco', 'state']].sort_values('dt_start')

