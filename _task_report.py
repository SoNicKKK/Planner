
# coding: utf-8

# In[1]:

FOLDER = 'resources/'

import numpy as np
import pandas as pd
import time, datetime
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

#%matplotlib inline
plt.style.use('fivethirtyeight')
plt.rc('font', family='Times New Roman')

pd.set_option('max_rows', 50)

time_format = '%b %d, %H:%M'

start_time = time.time()
current_time = pd.read_csv(FOLDER + 'current_time.csv').current_time[0]
twr          = pd.read_csv(FOLDER + 'team_work_region.csv', converters={'twr':str})
links        = pd.read_csv(FOLDER + 'link.csv')
stations     = pd.read_csv(FOLDER + 'station.csv', converters={'station':str})
train_info   = pd.read_csv(FOLDER + 'train_info.csv', converters={'train': str, 'st_from':str, 'st_to':str, 'oper_location':str,
                                                                 'st_from':str, 'st_to':str})
train_plan   = pd.read_csv(FOLDER + 'slot_train.csv', converters={'train': str, 'st_from':str, 'st_to':str})
loco_info    = pd.read_csv(FOLDER + 'loco_attributes.csv', converters={'train':str, 'loco':str, 'depot':str,
                                                                      'st_from':str, 'st_to':str})
loco_plan    = pd.read_csv(FOLDER + 'slot_loco.csv', converters={'train':str, 'loco':str, 'st_from':str, 'st_to':str})
team_info    = pd.read_csv(FOLDER + 'team_attributes.csv', converters={'team':str,'depot':str, 'oper_location':str,                                                                  'st_from':str, 'st_to':str, 'loco':str, 'depot_st':str})
team_plan    = pd.read_csv(FOLDER + 'slot_team.csv', converters={'team':str,'loco':str, 'st_from':str, 'st_to':str})
loco_series  = pd.read_csv(FOLDER + 'loco_series.csv')

team_info.regions = team_info.regions.apply(literal_eval)
st_names = stations[['station', 'name', 'esr']].drop_duplicates().set_index('station')
print('Время составления отчета:', time.strftime(time_format, time.localtime()))
print('Время запуска планировщика: %s (%d)' % (time.strftime(time_format, time.localtime(current_time)), current_time))


# In[2]:

# Мержим таблицы _plan и _info для поездов, локомотивов и бригад
# Добавляем во все таблицы названия станций на маршруте и времена отправления/прибытия в читабельном формате

def add_info(df):    
    if 'st_from' in df.columns:
        df['st_from_name'] = df.st_from.map(st_names.name)
    if 'st_to' in df.columns:
        df['st_to_name'] = df.st_to.map(st_names.name)
    if 'time_start' in df.columns:
        df['time_start_norm'] = df.time_start.apply(lambda x: time.strftime(time_format, time.localtime(x)))
    if 'time_end' in df.columns:
        df['time_end_norm'] = df.time_end.apply(lambda x: time.strftime(time_format, time.localtime(x)))
    if 'oper_location' in df.columns:
        df['oper_location_name'] = df.oper_location.map(st_names.name)    
        df.oper_location_name.fillna(0, inplace=True)
    if ('oper_location' in df.columns) & ('st_from' in df.columns) & ('st_to' in df.columns):        
        df['loc_name'] = df.oper_location_name
        df.loc[df.loc_name == 0, 'loc_name'] = df.st_from_name + ' - ' + df.st_to_name
    
add_info(train_plan)
add_info(loco_plan)
add_info(team_plan)
add_info(loco_info)
add_info(team_info)
add_info(train_info)
train_plan = train_plan.merge(train_info, on='train', suffixes=('', '_info'), how='left')
loco_plan = loco_plan.merge(loco_info, on='loco', suffixes=('', '_info'), how='left')
team_plan = team_plan.merge(team_info, on='team', suffixes=('', '_info'), how='left')
team_plan['team_type'] = team_plan.team.apply(lambda x: 'Реальная' if str(x)[0] == '2' else 'Фейковая')


# In[3]:

def nice_time(t):
    return time.strftime(time_format, time.localtime(t)) if t > 0 else ''


# In[4]:

REPORT_FOLDER = 'report/'
PRINT = False
report = ''

def add_line(line, p=PRINT):    
    global report        
    if p:                
        if type(line) == pd.core.frame.DataFrame:
            print(line.to_string(index=False))
        elif type(line) == pd.core.series.Series:
            print(line.to_string())
        else:
            print(line)
    if type(line) == pd.core.frame.DataFrame:        
        report += ('%s<br>' % line.to_html(index=False))
    elif type(line) == pd.core.series.Series:
        report += ('%s<br>' % line.to_frame().reset_index().to_html(index=False))
    else:
        report += ('%s<br>' % line)
    
def add_header(header, h=4, p=PRINT):
    global report
    report += ('<h%d>%s</h%d>' % (h, header, h))
    if p:
        print(header)

def add_image(filename):
    global report
    report += ('<img src="%s" alt="%s" height="40%%">' % (filename, filename))

def create_report(filename):
    global report
    report = report.replace('<table border="1" class="dataframe">','<table class="table table-striped">')
    html_string = '''
        <html>
            <head>
                <link rel="stylesheet" href="skeleton.css">
                <style>body{ margin:20 20; background:whitesmoke; }
                table {table-layout : fixed}
                </style>
            </head>
            <body>                
                %s
            </body>
        </html>''' % (report)
    f = open(filename,'w', encoding='utf-8-sig')
    f.write(html_string)
    f.close()
    print('Report created: %s' % filename)
    
def create_zip(filename):
    zip_filename = filename[:-5] + '.zip'
    zf = zipfile.ZipFile(zip_filename, mode='w')
    try:
        #print 'Отчет заархивирован в файл'
        zf.write(filename)
        zf.write('report\skeleton.css')
    finally:
        print('Zip-file created: %s' % zip_filename)
        zf.close()    


# In[5]:

import sys
if len(sys.argv) > 1:
    if sys.argv[1].upper() in stations.name.values:
        st_name = sys.argv[1]
    else: st_name = 'ИРКУТСК-СОРТИРОВОЧНЫЙ'
else: st_name = 'ИРКУТСК-СОРТИРОВОЧНЫЙ'
add_line('Отчет строится для поездов формированием на станции %s' % st_name)
add_line('Время начала планирования: %s' % nice_time(current_time))


# In[6]:

train_plan['train_type'] = train_plan.train.apply(lambda x: int(x[0]))
train_plan.loc[train_plan.train_type == 9, 'task'] = train_plan.loc[train_plan.train_type == 9, 'train'].apply(lambda x: x[4:9])
sf = train_plan[train_plan.train_type == 9]
sf_start = sf.drop_duplicates('train')
sf_irk = sf_start[sf_start.st_from_name == st_name]
add_header('Все запланированные поезда своего формирования по направлениям:')
add_line(sf_irk.st_to_name.value_counts())


# In[7]:

def get_planned_trains(row):
    return sf_irk[(sf_irk.st_to_name == row.st_next_name) & (sf_irk.time_start >= row.time_start)
                 & (sf_irk.time_start < row.time_end)].train.count()

#f = {'number':'sum', 'id':'unique'}
task = pd.read_csv(FOLDER + 'task.csv', dtype={'st_from':str, 'st_to':str, 'st_next':str})
add_info(task)
task['duration'] = task.time_end - task.time_start
task['st_next_name'] = task.st_next.map(st_names.name)
cols = ['id', 'time_start', 'time_end', 'time_start_norm', 'time_end_norm', 'duration', 'st_from_name', 'st_next_name', 'number']
a = task[task.st_from_name == st_name].sort_values(['time_start', 'st_to_name'])[cols].drop_duplicates()
b = a.groupby(['time_start', 'time_end', 'time_start_norm', 'time_end_norm', 'st_next_name'])        .agg({'number':'sum', 'id':'unique'}).reset_index()
b['plan_at_time'] = b.apply(lambda row: get_planned_trains(row), axis=1)
b = b[['id', 'time_start', 'time_end', 'time_start_norm', 'time_end_norm', 'st_next_name', 'number', 'plan_at_time']]
add_header('Задания (сгруппированные по направлениям) и кол-во поездов, запланированных в нужное время:')
add_line(b)


# In[8]:

b['part_id'] = b['id'].apply(lambda x: [str(t)[7:] for t in x])
problem_tasks = b[b.number > b.plan_at_time].iloc[0]['id']
problem_tasks_part_id = b[b.number > b.plan_at_time].iloc[0]['part_id']
add_header('Примеры заданий, по которым запланировано недостаточно поездов: %s' % problem_tasks)


# In[9]:

cols = ['train', 'task', 'st_from_name', 'st_to_name', 'time_start_norm']
add_header('Поезда по этим заданиям (отсортированы по id):')
add_line(sf_irk[sf_irk.task.isin(problem_tasks_part_id)].sort_values('train')[cols])


# In[10]:

filename = REPORT_FOLDER + 'task_' + time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + '.html'
create_report(filename)
create_zip(filename)


# In[16]:

st_name = 'ИРКУТСК-СОРТИРОВОЧНЫЙ'
loco_info['ser_name'] = loco_info.series.map(loco_series.set_index('ser_id').ser_name)
loco_info['oper_time_f'] = loco_info.oper_time.map(nice_time)
irk_loco = loco_info[(loco_info.loc_name == st_name) & (loco_info.ltype == 1)][['loco', 'ser_name', 'sections', 'regions', 'oper', 'oper_time_f', 'loc_name', 'tts', 'dts']]
irk_loco.loco.count()


# In[24]:

cols = ['loco', 'st_from_name', 'st_to_name', 'time_start_norm', 'time_end_norm', 'train', 'end']
loco_plan['end'] = ((loco_plan.loco != loco_plan.loco.shift(-1)) | (loco_plan.train != loco_plan.train.shift(-1))) & (loco_plan.train != '-1')
loco_ends = loco_plan[loco_plan.end == True]
loco_ends[loco_ends.st_to_name == st_name].sort_values('time_end')[cols]

