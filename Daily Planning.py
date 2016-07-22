
# coding: utf-8

# In[1]:

#get_ipython().magic('run common.py')


# In[2]:

st_name = 'ИРКУТСК-СОРТИРОВОЧНЫЙ'


# In[21]:

d = datetime.datetime.fromtimestamp(current_time - 18 * 3600)
day_start = int(datetime.datetime.timestamp(datetime.datetime(d.year, d.month, d.day, 18, 0, 0))) + 24 * 3600
print(time.ctime(current_time))
print(time.ctime(day_start))


# In[22]:

team_plan['dt_start'] = team_plan.time_start.apply(datetime.datetime.fromtimestamp)
team_plan['depot_name'] = team_plan.depot.map(st_names.name)
team_start_stations = team_plan[team_plan.state.isin([0, 1])].drop_duplicates('team').set_index('team')
team_plan['start_st'] = team_plan.team.map(team_start_stations.st_from_name)
team_plan.depot.fillna(team_plan.start_st, inplace=True)


# In[28]:

ct = datetime.datetime.fromtimestamp(current_time)
dt = datetime.datetime.fromtimestamp(day_start)


# In[29]:

team_plan['hour_start'] = team_plan.time_start.apply(lambda x: 3600 * int(x / 3600))
team_plan['hour_start_norm'] = team_plan.hour_start.apply(nice_time)


# In[39]:

# Total team depart
team_cols = ['team', 'st_from_name', 'st_to_name', 'time_start_norm', 'hour_start_norm', 'state', 'loco', 'depot_name']
a = team_plan[(team_plan.st_from_name == st_name) 
              & (team_plan.state.isin([0, 1]))
              & (team_plan.time_start >= day_start)
              & (team_plan.time_start < day_start + 24 * 3600)].sort_values('time_start')
print(a.groupby(['st_to_name', 'state']).team.count())
b = a.groupby(['st_to_name', 'hour_start_norm']).team.count().to_frame().reset_index()
c = b.pivot('st_to_name', 'hour_start_norm', 'team').fillna(0)
c


# In[37]:

b_dep = a.groupby(['depot_name', 'hour_start_norm']).team.count().to_frame().reset_index()
c_dep = b_dep.pivot('depot_name', 'hour_start_norm', 'team').fillna(0)
print(c_dep.sum(axis = 1).sort_values(ascending=False))
c_dep


# In[33]:

st1 = 'БАТАРЕЙНАЯ'
d = team_plan[(team_plan.st_from_name == st_name) & (team_plan.st_to_name == st1)
           & (team_plan.state.isin([0, 1]))
           & (team_plan.time_start >= day_start)
           & (team_plan.time_start < day_start + 24 * 3600)].sort_values('time_start')
d[d.state == 1].hour_start_norm.value_counts().sort_index()
#d[d.state == 1][team_cols].head(10)


# In[8]:

team_plan['type'] = team_plan.team.apply(lambda x: int(str(x)[:1]))
print(time.ctime(current_time))
team_plan[(team_plan['type'] == 7) & (team_plan.state.isin([0, 1]) & (team_plan.time_start <= current_time + 30 * 3600))]        .drop_duplicates('team').hour_start_norm.value_counts().sort_index()


# In[9]:

st_name = 'ИРКУТСК-СОРТИРОВОЧНЫЙ'
irk_t = team_plan[(team_plan.st_from_name == st_name) & (team_plan.state.isin([0, 1]))
         & (team_plan.time_start >= current_time) & (team_plan.time_start < current_time + 24 * 3600)]
irk = team_plan[(team_plan.team.isin(irk_t.team)) & (team_plan.state.isin([0, 1]))].copy(deep=True)
mins = irk.groupby('team').time_start.min()
irk['end_trip'] = (irk.team != irk.team.shift(-1)) | (irk.state != irk.state.shift(-1)) | (irk.loco != irk.loco.shift(-1))
irk['start_trip'] = (irk.team != irk.team.shift(1)) | (irk.state != irk.state.shift(1)) | (irk.loco != irk.loco.shift(1))
irk[['team', 'start_trip', 'end_trip']]
irk[(irk.start_trip == True) | (irk.end_trip == True)][['team', 'st_from_name', 'st_to_name', 'state', 'loco']]
starts = irk[(irk.start_trip == True)][['team', 'st_from_name', 'time_start', 'time_start_norm', 'state', 'loco']].reset_index()
ends = irk[(irk.end_trip == True)][['st_to_name', 'time_end', 'time_end_norm']].reset_index()
cols = ['team', 'st_from_name', 'st_to_name', 'time_start', 'time_end', 'time_start_norm', 'time_end_norm', 'state', 'loco']
trips = pd.concat([starts, ends], axis=1)[cols]
trips[(trips.st_from_name == st_name)].st_to_name.value_counts()

