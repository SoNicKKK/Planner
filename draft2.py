
# coding: utf-8

# In[142]:

from random import randrange
import datetime as dt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
sns.set(style='whitegrid', context='notebook')
sns.set_color_codes('dark')

k = 1

sd = int(dt.datetime(2016, 8, 4).timestamp())
team_num = 10
times = []
for i in range(team_num):
    times.append([randrange(2), dt.datetime.fromtimestamp(sd + 60 * int(randrange(14400 + 7200) / 60))])
    
df = pd.DataFrame(times, columns = ['tt', 'time'])
names = ['Alice', 'Bob', 'Clark', 'David', 'Ed', 'Frank', 'George', 'Helen', 'Ivan', 'John', 'Kelly', 'Lance', 'Mario',
        'Nolan', 'Oleg', 'Paul', 'Quentin', 'Ricky', 'Steve', 'Thomas', 'Ulaf', 'Valera', 'Wayne', 'Xi', 'Yang', 'Zed']
df['ts'] = df.time.apply(lambda x: int(x.timestamp()))
df['ts_norm'] = (df.ts - df.ts.mean()) / k
df = df.sort_values('ts').reset_index()
df['name'] = names[:team_num]
df.drop('index', axis=1, inplace=True)
df


# In[143]:

slot_num = 4
times = []
for i in range(slot_num):        
    times.append([randrange(2), dt.datetime.fromtimestamp(sd + 7200 + 60 * int(randrange(7200) / 60))])
    
df_slot = pd.DataFrame(times, columns = ['tt', 'time'])
df_slot = df_slot.sort_values('time')
train_names = ['train' + str(x) for x in np.arange(slot_num)]
df_slot['name'] = train_names
df_slot['ts'] = df_slot.time.apply(lambda x: int(x.timestamp()))
df_slot['ts_norm'] = (df_slot.ts - df_slot.ts.mean()) / k
norm_sh = df.ts_norm.min() - df_slot.ts_norm.min()
df_slot['ts_norm_shift'] = df_slot.ts_norm + norm_sh
df_slot = df_slot.reset_index().drop('index', axis=1)
df_slot


# In[148]:

def new_util(x):
    u = []
    t_t = x.ts_norm
    for t_i in train_times:
        add = 0 if x.tt == 1 else 1 - (t_i - t_t) / 15000        
        delta = (t_i - t_t) / 3600       
        util = np.exp(-(delta ** 2)) + 0.2 * add
        u.append(util)
    return pd.Series(u)

def prev_util(x):
    #t_t = x.ts
    t_t = x.ts_norm
    u = []
    #t_0 = df_slot[df_slot.ts >= t_t].ts.min()
    t_0 = df_slot[df_slot.ts_norm >= t_t].ts.min()
    for t_i in train_times:
        if t_t > t_i:
            util = 0
        else:
            add = 0 if x.tt == 1 else 1 - (t_i - t_t) / 150
            #util = np.round(((t_0 - t_t) / np.exp((t_i - t_0))) / 1000 + 1.0 * add, 3)            
            #util = (t_0 - t_t) / 
        u.append(util)
    return pd.Series(u) 

def get_util(x):
    t_t = x.ts
    u = []
    for t_i in train_times:        
        #print(t_i - t_t, add)
        if (team_num < slot_num) & (t_t > t_i):
            add = 1 - np.exp(-0.5*((t_i - t_t))**2)
            util = 1.0 + 0.1 * add
        else:
            add = 0 if x.tt == 1 else 1 - (t_i - t_t) / 150
            if (x.tt == 1) & (t_t > t_i):
                k = 20
            else:
                k = 5        
            delta_h = (t_t - t_i)
            util = np.exp(-(delta_h ** 2) / k) + 0.1 * add
        u.append(util)
    return pd.Series(u)    
    

df.drop([x for x in df.columns if 'train' in x], axis=1, inplace=True)
train_list = sorted(df_slot.name.unique())
#train_times = list(df_slot.ts.values)
train_times = list(df_slot.ts_norm_shift.values)
df[train_list] = df.apply(lambda x: new_util(x), axis=1)


# In[145]:

plt.figure(figsize=(15, 5))
cmap = matplotlib.cm.rainbow
for i in range(len(df.index)):
    plt.plot(df_slot.ts - sd, df.ix[i][train_list], label='%d - %s' % (i, df.ix[i]['name']), color=cmap(i / float(team_num)))
    plt.scatter(df.ix[i].ts - sd, 3.0, color=cmap(i / float(team_num)), s=50)

plt.xlabel('Время готовности поезда')
plt.ylabel('Значение ФП')
for t in df_slot.ts.values:
    plt.plot([t - sd] * 100, np.linspace(0, 3.5, 100), 'r--', lw=0.5)
    
#plt.xticks(df_slot.ts, df_slot.time, rotation=45)
plt.legend(loc='best', frameon=True)
plt.tight_layout()


# In[146]:

from scipy import optimize
row, col = optimize.linear_sum_assignment(-df[train_list].as_matrix())
print(sorted(list(zip(['train %d' % t for t in col], df.ix[row].name))))
df['bt'] = -1
df.ix[row, 'bt'] = col

def get_max(row):
    l = row[train_list]
    is_max = row.index == ('train' + str(int(row.bt)))    
    return ['background-color: yellow' if v else '' for v in is_max]    

print(df_slot[['name', 'time', 'ts_norm']])
df.sort_values('ts').style.apply(get_max, axis=1)


# In[125]:

a = pd.melt(df.sort_values('ts').reset_index()[train_list])
a.columns = ['train', 'util']
a['train'] = a.train.apply(lambda x: int(x[-1]))


# In[ ]:

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure().gca(projection='3d')
#ax = fig.add_subplot(111, projection='3d')

fig.scatter(a.index, a.train, a.util, c='r', marker='o')

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

plt.show()


# In[ ]:

x = np.linspace(0, 4, 100)
plt.figure(figsize=(12, 5))
t_0 = 2
x1 = np.array([i for i in x if i >= t_0])
x2 = np.array([i for i in x if i < t_0])
for k in [5, 20]:
    y1 = np.exp(-((x1 - t_0)**2) / k)
    y2 = np.exp(-((x2 - t_0)**2) / 50)
    plt.scatter(x1, y1, color=cmap(k / float(100)), s=5, label='k=%d' % k)
    plt.scatter(x2, y2, color=cmap(k / float(100)), s=5, label='k=%d' % k)
#k = [ for i in x ]
plt.ylim(-0.2, 1.2)
plt.legend()


# In[5]:

#get_ipython().magic('run read.py')
#get_ipython().magic('run common.py')


# In[15]:

# pt = pd.read_csv(FOLDER + 'prev_team.csv')
# pt.to_csv('prev.csv', index=False)
pt = pd.read_csv('prev.csv', dtype={'team':str})
pt['plan_ready_time'] = pt.team.map(team_plan[team_plan.state == 2].drop_duplicates('team').set_index('team').time_start)
pt[pt.prev_ready_time != pt.plan_ready_time]


# In[13]:

a = pt[pt.prev_ready_time != pt.plan_ready_time].team
cols = ['team', 'st_from_name', 'st_to_name', 'time_start', 'time_end', 'state', 'loco']
for team in a:
    print(team_plan[team_plan.team == team][cols].to_string(index=False))
    print('\n')


# In[23]:

st1, st2 = 'ИРКУТСК-СОРТИРОВОЧНЫЙ', 'ГОНЧАРОВО'
print(nice_time(current_time))
train_plan['dt_start'] = train_plan.time_start.apply(datetime.datetime.fromtimestamp)
train_plan['train_type'] = train_plan.train.apply(lambda x: int(x[0]))
train_plan[(train_plan.st_from_name == st1) 
           & (train_plan.st_to_name == st2)
          & (train_plan.time_start < current_time * 24*3600)
          & (train_plan.train_type != 8)].set_index('dt_start').train.resample('1H', 'count')


# ## Метод аукционов

# In[434]:

r, c = 4, 4
util = [float(i) for i in np.random.randint(10, size=r*c)]
util = np.array(util).reshape(r, c)
util0 = util.copy()
util


# In[435]:

k0 = [i for i in range(util.shape[1]) if i%2==0]
k1 = [i for i in range(util.shape[1]) if i%2==1]
k0, k1
best_rate = 1.0 # k0 / k_all


# In[436]:

def get_error(curr_rates, best_rates):
    return round(np.sqrt(sum((np.array(curr_rates) - np.array(best_rates)) ** 2)) / len(best_rates), 4)


# In[437]:

from scipy import optimize
row, col = optimize.linear_sum_assignment(-util)
print(sum(util[row, col]))
print(row, col)
best_sum = sum(util0[row, col])
eps, eps_err = 0.01, 0.00001

curr_k0 = len([i for i in col if i in k0])
curr_k1 = len([i for i in col if i in k1])
total = len(col)
std = get_error([curr_k0 / total, curr_k1 / total], [best_rate, 1 - best_rate])
print(std)
curr_rate = curr_k0 / (curr_k0 + curr_k1)
curr_rate_err = curr_rate - best_rate
print(curr_k0, curr_k1, curr_rate, best_rate, std)

l = 0 if curr_rate_err > 0 else 1
err_ind = [i for i in range(len(col)) if col[i]%2 == l]

res, err_rate = [best_sum], [1000, std]
assign = [(row, col)]
print(res, err_rate)
while (err_rate[-2] >= err_rate[-1]) & (err_rate[-1] != 0):
    delta = []
    for i in err_ind:
        r, c = row[i], col[i]
        m = max([j for j in util[r, k0] if util[r, c] > j])
        delta.append(util[r, c] - m)
        #print(r, c, delta)

    d = min(delta) + eps
    util[row[l], col[l]] -= d
    row, col = optimize.linear_sum_assignment(-util)
    print(row, col)    
    l = 0 if curr_rate_err > 0 else 1
    err_ind = [i for i in range(len(col)) if col[i]%2 == l]
    res.append(sum(util0[row, col]))
    curr_k0 = len([i for i in col if i in k0])
    curr_k1 = len([i for i in col if i in k1])
    curr_rate = curr_k0 / (curr_k0 + curr_k1)
    curr_rate_err = curr_rate - best_rate
    #err_rate.append(np.abs(curr_rate_err))
    #print(res, curr_k0, curr_k1, curr_rate, best_rate, curr_rate_err)
    std = get_error([curr_k0 / total, curr_k1 / total], [best_rate, 1 - best_rate])
    err_rate.append(std)
    print(res, curr_k0, curr_k1, curr_rate, best_rate, std)
    assign.append((row, col))
    #print(res, current_rate_err)    
    
print('End')
# print(row, col)
# print(sum(util[row, col]))
# print(res, err_rate[1:])


# In[433]:

res_norm = 1 - res / best_sum
err = np.sqrt(res_norm ** 2 + np.array(err_rate[1:]) ** 2) / 2
print(res_norm)
print(err_rate[1:])
print(err)
assign_ind = np.argmin(err)
print(assign[assign_ind])
print(sum(util0[assign[assign_ind]]))


# In[438]:

#get_ipython().magic('run common.py')


# In[565]:

lines = []
with open('out.txt') as f:
    for line in f:
        lines.append(line[:-1])
        
a = [int(line.split('?')[-1].strip()) for line in lines[:18]]
pd.Series(a).apply(datetime.datetime.fromtimestamp).sort_values()
df_loco = pd.DataFrame([[line.split('?')[1].strip(), int(line.split('?')[-1].strip())] for line in lines[:18]], 
                       columns=['loco', 'time'])


# In[562]:

b = [[line.split('?')[1].strip(), int(line.split('?')[-2].strip()), line.split('?')[-1].strip()] for line in lines[19:]]
#teams = [line.split('?')[1].strip() for line in lines[19:]]
#pd.Series(b).apply(datetime.datetime.fromtimestamp).sort_values()
df_team = pd.DataFrame(b, columns=['team', 'time', 'loco'])
df_team['dt'] = df_team.time.apply(datetime.datetime.fromtimestamp)
df_team = df_team.set_index('team')
df_team


# In[527]:

sns.set(style='whitegrid', context='notebook')
plt.figure(figsize=(14, 4))
plt.scatter(a, [0.0] * len(a), c='r', s=50, label='loco')
plt.scatter(b, [1.0] * len(b), c='b', s=50, label='team')
for i in range(len(a)):
    plt.plot([a[i], b[i]], [0.0, 1.0], 'g--', lw=1.0)    
    print('shift %d = %d' % (i, b[i]-a[i]))
    plt.annotate(xy=(a[i]-250, -0.1), s=i)
    
for j in range(len(b)):
    plt.annotate(xy=(b[j]-250, 1.1), s=j)   


# In[528]:

print(datetime.datetime.fromtimestamp(1470739560))
print(datetime.datetime.fromtimestamp(1470756168))
print(datetime.datetime.fromtimestamp(1470786408))
print(datetime.datetime.fromtimestamp(1470773400))
print(nice_time(current_time))


# In[579]:

df_team['line'] = ''
e = {}
with open('out2.txt', encoding='utf-8-sig') as f:
    for line in f:
        q = line[:-1].split(' - ')[1]
        #team = q.split()[0][5:]
        #loco = q.split()[1][5:]
        l = [t for t in teams if t in q]
        if l !=[]:
            #print(l, line[line.find(' - ')+3:-1])
            e[l[0]] = line[line.find(' - ')+3:-1]

df_team = df_team.reset_index()
df_team['line'] = df_team.team.apply(lambda x: e[x])
df_team


# In[581]:

sns.set(style='whitegrid', context='notebook')
b = df_team.time
plt.figure(figsize=(14, 4))
plt.scatter(a, [0.0] * len(a), c='r', s=50, label='loco')
plt.scatter(b, [1.0] * len(b), c='b', s=50, label='team')
good = df_team[df_team.line.apply(lambda x: 'Полезность' in x)].time
plt.scatter(good, [1.0] * len(good), s=100, c='g')
for i in range(len(a)):
    plt.plot([a[i], b[i]], [0.0, 1.0], 'g--', lw=1.0)    
    print('shift %d = %d' % (i, b[i]-a[i]))
    plt.annotate(xy=(a[i]-250, -0.1), s=i)
    
for j in range(len(b)):
    plt.annotate(xy=(b[j]-250, 1.1), s=j)
    
df_team['loco_time'] = df_team.loco.map(df_loco.set_index('loco').time)
lt = df_team[df_team.loco != '0'].dropna().loco_time
tt = df_team[df_team.loco != '0'].dropna().time
for i in range(len(tt)):
    plt.plot([lt, tt], [0.0, 1.0], c='black', lw=0.7)
plt.legend(loc='best')


# In[631]:

def get_util(x):
    u = []
    for t_i in train_times:        
        delta = COEF * (t_i - x.ts_norm)
        util = np.exp(-(delta ** 2))
        u.append(util)
    return pd.Series(u)   

df_loco['ts_norm'] = (df_loco.time - df_loco.time.min()) / (df_loco.time.max() - df_loco.time.min())
df_team['ts_norm'] = (df_team.time - df_team.time.min()) / (df_team.time.max() - df_team.time.min())
COEF = max(len(df_loco.ts_norm), len(df_team.ts_norm)) / 10
#sh = df_team.ts_norm.min() - df_loco.ts_norm.min()
#df_loco['ts_norm'] = df_loco.ts_norm + sh
train_times = df_loco.ts_norm
a = df_team.apply(lambda row: get_util(row), axis=1).applymap(lambda x: round(x, 3))
#a.style.applymap(lambda x: 'background-color: yellow' if x > 0.5 else '')
plt.figure(figsize=(12, 10))
sns.heatmap(a, annot=True, lw=0.01)


# In[625]:

plt.scatter(df_loco.ts_norm, [0.0] * len(df_loco.ts_norm), s=50, c='r')
plt.scatter(df_team.ts_norm, [1.0] * len(df_team.ts_norm), s=50, c='b')


# In[ ]:



