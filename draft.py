
# coding: utf-8

# In[18]:

get_ipython().magic('run common.py')


# In[19]:

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


# In[20]:

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


# In[21]:

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


# In[22]:

print(nice_time(current_time))
cols = ['team', 'st_from_name', 'st_to_name', 'oper_time_f', 'time_start_norm', 'state', 'wait_ct', 'wait']
team_plan['sinfo'] = team_plan['state_info']
team_plan['oper_time_f'] = team_plan.oper_time.apply(nice_time)
team_plan['wait_ct'] = np.round((current_time - team_plan.oper_time) / 3600, 2)
team_plan['wait'] = np.round((team_plan.time_start - team_plan.oper_time) / 3600, 2)
a = team_plan[(team_plan.state_info == '2') 
          & (team_plan.state.isin([0, 1]))][cols].drop_duplicates('team').sort_values('wait', ascending=False)

(a.wait - a.wait_ct).describe()


# In[23]:

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


# In[32]:

ts = ['200252616155', '200253041216']


# In[37]:

print(nice_time(current_time))
import datetime as dt
dt.datetime.fromtimestamp(1469296440)

