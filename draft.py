
# coding: utf-8

# In[462]:

get_ipython().magic('run common.py')


# In[377]:

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


# In[378]:

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


# In[386]:

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

