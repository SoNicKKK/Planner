
# coding: utf-8

# In[80]:

import numpy as np
import pandas as pd
import time, datetime
from ast import literal_eval
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile


# In[81]:

report = ''
FOLDER = 'resources/'
REPORT_FOLDER = 'report/'


# In[82]:

import sys
JOIN_OPS, ZIP, PRINT = False, False, True
if len(sys.argv) > 1:
    if 'ops' in sys.argv:
        JOIN_OPS = True
    if 'zip' in sys.argv:
        ZIP = True 
    if 'noprint' in sys.argv:
        PRINT = False


# In[83]:

time_format = '%b %d, %H:%M'
def nice_time(x):
    return time.strftime(time_format, time.localtime(x))


# In[84]:

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
    print('Отчет сформирован: %s' % filename)
    
def create_zip(filename):
    zip_filename = filename[:-5] + '.zip'
    zf = zipfile.ZipFile(zip_filename, mode='w')
    try:
        #print 'Отчет заархивирован в файл'
        zf.write(filename)
        zf.write('report\skeleton.css')
    finally:
        print('Отчет заархивирован в файл %s' % zip_filename)
        zf.close()    


# In[85]:

pd.set_option('max_rows', 50)

start_time = time.time()
current_time = pd.read_csv(FOLDER + 'current_time.csv').current_time[0]
twr          = pd.read_csv(FOLDER + 'team_work_region.csv', converters={'twr':str})
links        = pd.read_csv(FOLDER + 'link.csv')
stations     = pd.read_csv(FOLDER + 'station.csv', converters={'station':str})
train_info   = pd.read_csv(FOLDER + 'train_info.csv', converters={'train': str, 'st_from':str, 'st_to':str})
train_plan   = pd.read_csv(FOLDER + 'slot_train.csv', converters={'train': str, 'st_from':str, 'st_to':str})
loco_info    = pd.read_csv(FOLDER + 'loco_attributes.csv', converters={'train':str, 'loco':str})
loco_plan    = pd.read_csv(FOLDER + 'slot_loco.csv', converters={'train':str, 'loco':str, 'st_from':str, 'st_to':str})
team_info    = pd.read_csv(FOLDER + 'team_attributes.csv', converters={'team':str,'depot':str, 'oper_location':str,                                                                  'st_from':str, 'st_to':str, 'loco':str, 'depot_st':str})
team_plan    = pd.read_csv(FOLDER + 'slot_team.csv', converters={'team':str,'loco':str, 'st_from':str, 'st_to':str})
loco_series  = pd.read_csv(FOLDER + 'loco_series.csv')

team_info.regions = team_info.regions.apply(literal_eval)
st_names = stations[['station', 'name', 'esr']].drop_duplicates().set_index('station')


# In[86]:

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
train_plan = train_plan.merge(train_info, on='train', suffixes=('', '_info'), how='left')
loco_plan = loco_plan.merge(loco_info, on='loco', suffixes=('', '_info'), how='left')
team_plan = team_plan.merge(team_info, on='team', suffixes=('', '_info'), how='left')
team_plan['team_type'] = team_plan.team.apply(lambda x: 'Реальная' if str(x)[0] == '2' else 'Фейковая')
loco_plan['train_time'] = list(zip(loco_plan.train, loco_plan.time_start))
train_plan['train_time'] = list(zip(train_plan.train, train_plan.time_start))
train_plan['loco'] = train_plan.train_time.map(loco_plan.drop_duplicates('train_time').set_index('train_time').loco)
loco_plan['loco_time'] = list(zip(loco_plan.loco, loco_plan.time_start))
team_plan['loco_time'] = list(zip(team_plan.loco, team_plan.time_start))
loco_plan['team'] = loco_plan.loco_time.map(team_plan.drop_duplicates('loco_time').set_index('loco_time').team)


# In[87]:

print('''--------
Возможные ключи: 
ops - добавляет в отчет последние операции с бригадами из последнего файла "Операции*.txt"
noprint - отключает вывод отладочных принтов
zip - архивирует отчет
--------''')


# In[88]:

import os
import time
files = [files for root, directories, files in os.walk('./resources/others')][0]
times = {}
os.chdir('./resources/others')
try:
    for f in files:
        if 'Бригады_УТХ' in f:
            times[f] = int(os.path.getmtime(f))    

    if times != {}:
        uth_filename = max(times, key=lambda k: times[k])
        date_modified = times[uth_filename]
    else:
        uth_filename = 'Бригады_УТХ' + '.xls'
        date_modified = 0
    print('Данные об УТХ-бригадах взяты из файла %s (дата изменения %s)' % (uth_filename, nice_time(date_modified)))
    os.chdir('..')
    os.chdir('..')
except:
    os.chdir('..')
    os.chdir('..')


# In[89]:

# Загрузка УТХ-бригад из экселевской выгрузки
import xlrd
uth = pd.read_excel('./resources/others/' + uth_filename)
uth.columns = ['Номер', 'Машинист', 'Депо', 'Вид движения', 'Факт.явка', 'План.явка']
uth['Вид движения'] = uth['Вид движения'].apply(lambda x: str(x).replace('\n\t\t\t', ';'))
uth['irk'] = uth['Депо'].apply(lambda x: 'ТЧЭ-5 В' in x)
uth = uth[uth.irk]
#time_f = '%Y-%m-%d %H:%M:%S'
##uth['uth_presence'] = uth['План.явка'].apply(lambda x: time.mktime(datetime.datetime.strptime(x, '%H:%M %d.%m.%y').timetuple()))
if (uth['План.явка'].dtype == float):
    uth['План.явка'] = uth['План.явка'].apply(lambda x: datetime.datetime(*xlrd.xldate.xldate_as_tuple(x, 0)))
    print('Формат времени в столбце "Плановая явка" заменен c формата Excel на python datetime')

try:
    uth['uth_presence'] = uth['План.явка'].apply(lambda x: time.mktime(x.timetuple()))
except:
    try:
        uth['uth_presence'] = uth['План.явка'].apply(lambda x: time.mktime(time.strptime(x, "%Y-%m-%d %H:%M:%S %z")))
    except:
        uth['uth_presence'] = uth['План.явка'].apply(lambda x: time.mktime(time.strptime(x, "%H:%M %d.%m.%y")))
uth.head()


# In[90]:

info_cols = ['number', 'name', 'loc_name', 'state', 'depot_time_norm', 'is_planned']
team_info['name'] = team_info.number.map(uth.set_index('Номер')['Машинист'])
team_info['uth_presence'] = team_info.number.map(uth.set_index('Номер').uth_presence)
team_info['depot_time_norm'] = team_info.depot_time.apply(lambda x: time.strftime(time_format, time.localtime(x)) if x !=-1 else x)
planned = team_plan[team_plan.state.isin([0, 1])].drop_duplicates('team')
team_info['is_planned'] = team_info.team.isin(planned.team)


# In[91]:

df_input_show = team_info[team_info.number.isin(uth['Номер'])][info_cols]
df_input_show.is_planned.replace(False, 'Нет', inplace=True)
df_input_show.is_planned.replace(True, 'Да', inplace=True)
df_input_show.columns=['Номер', 'Машинист', 'Исх.местоположение', 'Исх.состояние', 'Время явки', 'В плане?']
cols = ['Номер', 'Машинист', 'Депо', 'Вид движения', 'План.явка', 'uth_presence']
df_show = uth[cols].set_index(['Номер', 'Машинист']).join(df_input_show.set_index(['Номер', 'Машинист'])).fillna('-').reset_index()


# In[92]:

team_cols = ['number', 'name', 'st_from_name', 'st_to_name', 'time_start', 'time_start_norm', 
             'state', 'loco_number', 'train_number', 'all_states']
team_plan['name'] = team_plan.number.map(uth.set_index('Номер')['Машинист'])
team_plan['loco_number'] = team_plan.loco.map(loco_info.set_index('loco').number)
team_plan['loco_time'] = list(zip(team_plan.loco, team_plan.time_start))
loco_plan['loco_time'] = list(zip(loco_plan.loco, loco_plan.time_start))
loco_plan['train_number'] = loco_plan.train.map(train_info.drop_duplicates('train').set_index('train').number)
loco_plan.loc[loco_plan.train_number.isnull(), 'train_number'] = loco_plan.train.apply(lambda x: str(x)[-4:])
team_plan['train_number'] = team_plan.loco_time.map(loco_plan.drop_duplicates('loco_time').set_index('loco_time').train_number)
team_plan['all_states'] = team_plan.team.map(team_plan.groupby('team').state.unique())
uth_plan = team_plan[team_plan.number.isin(uth['Номер'])]
df_output_show = uth_plan[uth_plan.state.isin([0, 1])].drop_duplicates('team').sort_values('time_start')[team_cols]
df_output_show.loco_number.fillna(-1, inplace=True)
df_output_show.columns = ['Номер', 'Машинист', 'Ст.отпр.', 'Ст.направл.', 'plan_time_start', 'Время отпр.', 
                          'Состояние', 'Номер ЛОК', 'Номер П', 'Все состояния']


# In[93]:

add_line('Время сбора данных и запуска планировщика: %s' % time.strftime(time_format, time.localtime(current_time)))
add_header('Всего %d иркутских бригад загружено в ОУЭР из УТХ' % uth['Номер'].count())
add_line('Из них:')
add_line('- передано в планировщик: %d' % team_info[team_info.number.isin(uth['Номер'])].team.count())
add_line('- не передано в планировщик: %d' % uth[uth['Номер'].isin(team_info.number) == False]['Номер'].count())
add_line('- запланировано: %d' % df_output_show['Номер'].count())
df_show_uth_plan = df_show.set_index(['Номер', 'Машинист']).join(df_output_show.set_index(['Номер', 'Машинист'])).fillna('-')


# In[94]:

def add_state_legend():
    add_line('Состояния бригад:')
    add_line('0 - следует пассажиром')
    add_line('1 - ведет локомотив')
    add_line('2 - явка в депо приписки')
    add_line('3 - находится на домашнем отдыхе')
    add_line('4 - отдыхает в пункте оборота')
    add_line('5 - прикреплена к локомотиву на станции')
    add_line('6 - прибыла на станцию с локомотивом')
    add_line('7 - прибыла на станцию пассажиром')
    add_line('8 - явка в пункте оборота')
    add_line('9 - сдача локомотива')


# In[95]:

files = [files for root, directories, files in os.walk('./resources/others')][0]
times = {}
os.chdir('./resources/others')
try:
    for f in files:
        if ('Операции' in f) & ('.txt' in f):
            times[f] = int(os.path.getmtime(f))

    if times != {}:
        ops_filename = max(times, key=lambda k: times[k])
        date_modified = times[ops_filename]
    else:
        ops_filename = 'Операции с УТХшными ЛБ.txt'
        date_modified = 0
    os.chdir('..')
    os.chdir('..')
except:
    os.chdir('..')
    os.chdir('..')
print('Данные об операциях с УТХ-бригадами взяты из файла "%s" (дата изменения %s)' % (ops_filename, nice_time(date_modified)))


# In[96]:

lines = []
cur_team_id = 0
cur_team_name = ''
with open ('./resources/others/' + ops_filename, encoding='utf-8-sig') as fop:
    for line in fop:        
        if line[:7] == 'Бригада':            
            sp = line[:-1].split()            
            cur_team_id = sp[2][:-1]
            cur_team_name = sp[1][:-1]
        if line[:4] == '2016':
            sp = line[:-1].split('\t')
            l = [cur_team_id, cur_team_name] + sp
            lines.append(l)
        
lines[:10]
cols = ['team', 'name', 'team_type', 'op_id', 'op_name', 'op_time', 'op_location']
df_ops = pd.DataFrame(lines, columns = ['team', 'name', 'op_time', 'op_id', 'team_type', 'op_name', 'op_location'])
df_ops = df_ops[cols]
df_ops.sample(3)


# In[97]:

print('Всего бригад в файле %s: %d' % (ops_filename, df_ops.team.drop_duplicates().count()))
print('Время сбора данных: %s' % time.strftime(time_format, time.localtime(current_time)))


# In[98]:

df_ops['timestamp'] = df_ops['op_time'].apply(lambda x:                                               time.mktime(datetime.datetime.strptime(x[:-6], "%Y-%m-%d %H:%M:%S").timetuple()))
print('Время последней операции в файле %s: %s' 
      % (ops_filename, time.strftime(time_format, time.localtime(df_ops.timestamp.max()))))


# In[99]:

mask = df_ops.timestamp <= current_time
cols = ['team', 'name', 'team_type', 'op_id', 'op_name', 'op_time', 'op_location']
last = df_ops[mask].groupby('team').timestamp.max().to_frame().reset_index().set_index(['team', 'timestamp'])            .join(df_ops.set_index(['team', 'timestamp'])).reset_index()
last[cols].sample(3)


# In[100]:

good = df_show[df_show['В плане?'] == 'Да']['Машинист'].unique()
last_good = last[last.name.isin(good) == False].sort_values(['op_name', 'timestamp']).reset_index()
last_good[cols].head()


# In[101]:

last.columns = ['Id', 'Timestamp', 'Машинист', 'Тип бр.', 
                       'Id посл.оп.', 'Посл.операция', 'Время посл.оп.', 'Место посл.оп.']
op_cols = ['Id', 'Машинист', 'Тип бр.', 'Id посл.оп.', 'Посл.операция', 'Время посл.оп.', 'Место посл.оп.']
if JOIN_OPS:    
    show_cols = ['Номер', 'Машинист', 'Депо', 'Вид движения', 'Тип бр.', 
             'Id посл.оп.', 'Посл.операция', 'Время посл.оп.', 'Место посл.оп.',
             'План.явка', 'Исх.местоположение', 'Исх.состояние', 'Время явки', 'В плане?', 'Ст.отпр.', 'Ст.направл.', 
             'Время отпр.', 'Состояние', 'Номер ЛОК', 'Номер П', 'Все состояния']
    res = df_show_uth_plan.reset_index().set_index('Машинист').join(last[op_cols].set_index('Машинист'))
else:
    show_cols = ['Номер', 'Машинист', 'Депо', 'Вид движения',              
             'План.явка', 'Исх.местоположение', 'Исх.состояние', 'Время явки', 'В плане?', 'Ст.отпр.', 'Ст.направл.', 
             'Время отпр.', 'Состояние', 'Номер ЛОК', 'Номер П', 'Все состояния']
    res = df_show_uth_plan
res_to_index_start_with_0 = res.reset_index().sort_values(['uth_presence', 'Машинист'])[show_cols].reset_index()
res_to_index_start_with_0['index'] = res_to_index_start_with_0['index'] + 1
add_line(res_to_index_start_with_0, p=False)


# In[106]:

not_input = res_to_index_start_with_0[res_to_index_start_with_0['В плане?'] == '-']
not_planned = res_to_index_start_with_0[res_to_index_start_with_0['В плане?'] == 'Нет']
add_header('Не переданные бригады:')
add_line(list(not_input['Номер'].unique()))
add_header('Не запланированные бригады:')
add_line(list(not_planned['Номер'].unique()))


# In[74]:

add_state_legend()
filename = REPORT_FOLDER + 'uth_report_' + time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + '.html'
create_report(filename)
if ZIP:
    create_zip(filename)


# In[75]:

res_to_index_start_with_0[res_to_index_start_with_0['В плане?'] == 'Нет']['Номер'].unique()


# In[76]:

arr = [9205004041, 9205002684, 9205007941, 9205003679, 9205003528,
       9205004034, 9205004569, 9205003485, 9205000277, 9205000533,
       9205000335, 9205003824, 9205005071]
cols = ['number', 'depot', 'ready_type', 'state', 'loc_name', 'oper_time_f', 'loco', 'ttype']
team_info['oper_time_f'] = team_info.oper_time.apply(lambda x: time.ctime(x))
team_info[team_info.number.isin(arr)][cols].sort_values('state')

