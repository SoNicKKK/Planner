
# coding: utf-8

# In[9]:

get_ipython().magic('run common.py')


# In[10]:

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


# In[11]:

# prev_team = pd.read_csv(FOLDER + 'prev_team.csv', dtype={'team':str})
# prev_team.to_csv('prev_team2330.csv')


# In[15]:

prev_team = pd.read_csv('prev_team2330.csv', dtype={'team':str})
prev_team.columns = ['ind', 'team', 'prev_time']
prev_team.drop('ind', axis=1, inplace=True)


# In[30]:

team_plan['prev_time'] = team_plan.team.map(prev_team.set_index('team').prev_time)
a = team_plan[team_plan.state == 2].drop_duplicates('team')[['team', 'time_start', 'time_end', 'prev_time']]
a[a.time_end != a.prev_time]
#team_plan[team_plan.team == '200200242608'][['team', 'time_start', 'time_end', 'state']]

