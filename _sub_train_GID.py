
# coding: utf-8

# In[282]:

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
print('Planning start time: %s (%d)' % (time.strftime(time_format, time.localtime(current_time)), current_time))


# In[283]:

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


# In[284]:

def nice_time(t):
    return time.strftime(time_format, time.localtime(t)) if t > 0 else ''


# In[285]:

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


# In[286]:

import sys
if len(sys.argv) > 1:
    filename = sys.argv[1]    
else:
    filename = 'resources/others/Иркутск-Сорт_(нечет_отправление)_ Период28июн 10_30-29июн 10_30(VSD).xlsx'
    
if ('xls' not in filename) & ('csv' not in filename):
    filename = 'resources/others/Иркутск-Сорт_(нечет_отправление)_ Период28июн 10_30-29июн 10_30(VSD).xlsx'
    
print('Filename:', filename)


# In[287]:

add_header('Анализ файла %s' % filename, h=2, p=False)


# In[288]:

df = pd.read_excel(filename, header=2)
df = df[['П/п', 'Номер', 'Индекс', 'Время', 'Вес']]
df.columns = ['n', 'number', 'ind', 'time', 'weight']
max_train_idx = df[df.n.apply(lambda x: '-' in str(x))].index.min()
df = df.ix[:max_train_idx - 1]
df['ind'] = df.ind.apply(lambda x: x.replace(' ', '-'))
df['ind_part'] = df.ind.apply(lambda x: x[:-5])
stations['esr4'] = stations.esr.apply(lambda x: (str(x))[:4])
df['start_esr'] = df.ind.apply(lambda x: x[:4])
df['end_esr'] = df.ind.apply(lambda x: x[-4:])
df['start_name'] = df.start_esr.map(stations.drop_duplicates('esr4').set_index('esr4').name)
df['end_name'] = df.end_esr.map(stations.drop_duplicates('esr4').set_index('esr4').name)
add_header('Все поезда из ГИДа:')
add_line(df)


# ### Пробуем просто найти поезда по индексу

# In[289]:

df['train_id'] = df.ind.map(train_info.set_index('ind434').train)
good = df[df.train_id.isnull() == False].ind.count()
bad = df[df.train_id.isnull()].ind.count()
total = df.ind.count()
add_header('Поиск поездов по индексу')
add_line('Найдено поездов по индексу: %d из %d' % (good, total))
add_line('Не найдено поездов: %d' % bad)


# ### Пробуем найти поезда по части индекса (без хвоста)

# In[290]:

train_info['ind_part'] = train_info.ind434.apply(lambda x: x[:-5])
train_info.groupby('ind_part').train.unique()
df['susp_trains'] = df.ind_part.map(train_info.groupby('ind_part').train.unique())
df.train_id.fillna(df.susp_trains, inplace=True)
good = df[df.train_id.isnull() == False].ind.count()
bad = df[df.train_id.isnull()].ind.count()
add_line('Найдено поездов по индексу и части индекса: %d из %d' % (good, total))
add_line('Не найдено поездов: %d' % bad)
add_line('Примеры найденных поездов:')
add_line(df.head())


# ### Из оставшихся выбираем поезда своего формирования (они будут созданы из ССП)

# In[291]:

df[df.train_id.isnull()]
irk = df[(df.train_id.isnull()) & (df.start_esr == '9300')].ind.count()
bad = df[(df.train_id.isnull()) & (df.start_esr != '9300')].ind.count()
add_header('Поиск поездов своего формирования (станция Иркутск)')
add_line('Найдено поездов своего формирования: %d' % irk)
add_line('Всего найдено поездов: %d из %d' % (total - bad, total))
add_line('Не найдено поездов: %d' % bad)
add_line('Примеры найденных поездов:')
add_line(df[(df.train_id.isnull()) & (df.start_esr == '9300')].head())


# ### Смотрим, какие поезда остались не найденными

# In[292]:

add_header('Оставшиеся поезда')
add_line(df[(df.train_id.isnull()) & (df.start_esr != '9300')])


# ### Убираем поезда с близких станций формирования (начинающиеся на 93)

# In[293]:

add_header('Без поездов с близких станций формирования:')
not_found = df[(df.train_id.isnull()) & (df.start_esr.apply(lambda x: x[:2] != '93'))]
add_line(not_found)


# ### Загружаем csv с отсевами, ищем непереданные в планировщик поезда там

# In[294]:

otsev = pd.read_csv('./resources/others/otsev_detail.csv', sep=';',
                    dtype={'train_index':str, 'train_id':str, 'loco_id':str, 'team_id':str})
otsev['ind434'] = otsev['train_index'].apply(lambda x: str(x)[:4] + '-' + str(x)[6:9] + '-' + str(x)[9:-2])
otsev['time'] = otsev.time.apply(lambda x: x[:-3])
pd.set_option('display.max_colwidth', 35)
add_header('Проблемные поезда в логах отсевов')
add_line(otsev[otsev.ind434.isin(not_found.ind)].sort_values('ind434').sort_values(['train_index', 'time'])    [['uns', 'train_index', 'loco_id', 'team_id', 'location_name', 'type_name', 'time']])


# In[295]:

add_header('Данные по проблемным поездам в логах отсевов (поиск по части индекса)')
otsev['ind434_part'] = otsev.ind434.apply(lambda x: x[:-5])
a = otsev[(otsev.ind434.isin(not_found.ind) == False) 
             & (otsev.ind434_part.isin(not_found.ind_part))].sort_values('ind434')\
                [['train_id', 'train_index', 'ind434', 'out', 'otsev_list']].dropna(subset=['out'])
add_line(a)


# ### Непереданные поезда, которых нет и в логах отсевов

# In[296]:

cols = ['number', 'ind', 'time', 'weight', 'start_name', 'end_name']
add_header('Оставшиеся поезда, которых нет и в логах отсевов')
add_line(not_found[not_found.ind_part.isin(otsev.ind434_part) == False][cols])


# In[302]:

filename = REPORT_FOLDER + 'train_GID_' + time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + '.html'
create_report(filename)
create_zip(filename)

