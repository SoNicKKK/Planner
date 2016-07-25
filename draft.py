
# coding: utf-8

# In[6]:

#get_ipython().magic('run common.py')


# In[7]:

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


# In[8]:

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


# In[9]:

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


# In[10]:

print(nice_time(current_time))
cols = ['team', 'st_from_name', 'st_to_name', 'oper_time_f', 'time_start_norm', 'state', 'wait_ct', 'wait']
team_plan['sinfo'] = team_plan['state_info']
team_plan['oper_time_f'] = team_plan.oper_time.apply(nice_time)
team_plan['wait_ct'] = np.round((current_time - team_plan.oper_time) / 3600, 2)
team_plan['wait'] = np.round((team_plan.time_start - team_plan.oper_time) / 3600, 2)
a = team_plan[(team_plan.state_info == '2') 
          & (team_plan.state.isin([0, 1]))][cols].drop_duplicates('team').sort_values('wait', ascending=False)

(a.wait - a.wait_ct).describe()


# In[11]:

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


# In[95]:

df = pd.read_csv('./input/otsev_uth_detail.csv', encoding='utf-8-sig', sep=';', 
                 dtype={'train_id':str, 'train_index':str, 'loco_id':str, 'team_id':str}, 
                 parse_dates=['time', 'train_time', 'loco_time', 'team_time', 'team_ready_time1', 'team_ready_time2'])                 
print(df.columns)


# In[97]:

print(nice_time(current_time))
cols = ['uns', 'team_id', 'out',
        'team_type_name', 'team_time',
                                'team_ready_time1', 'team_ready_time2']
df['delta1'] = df.team_ready_time1 - df.team_time
nice_print(df[df.team_type_asoup_id == 54].sort_values('delta1')[cols].head())


# In[103]:

team_info['depot_time_f'] = team_info.depot_time.apply(nice_time)
team_info['return_time_f'] = team_info.return_time.apply(nice_time)
nice_print(team_info[team_info.team == '200200256743'][['team', 'oper_time_f', 'loc_name', 'depot_time_f', 'return_time_f']])


# In[102]:

m1 = team_info[(team_info.depot_time == -1) 
          & (team_info.return_time == -1)][['team', 'oper_time_f', 'loc_name', 'depot_time_f', 'return_time_f']]

with pd.option_context('display.max_colwidth', 40):
    nice_print(df[df.team_id.isin(m1.team)][cols])

