
# coding: utf-8

# # Построение отчета по использованию в планировании УТХ-бригад
# 
# ## Входные данные для отчета
# 
# Отчет строится на следующих данных:
# 
# 1. Лог планировщика (`jason-FullPlannerPlugin.log`). Перед построением отчета из лога планировщика скриптом `read.py` должны быть созданы соответствующие csv-файлы.
# 2. Список УТХ-бригад. Он должен располагаться в подпапке `./resources/others/`, файл должен иметь название типа `Бригады_УТХ*.xls`. Если файлов с таким названием несколько, то скрипт выберет последний по дате изменения. Этот файл - это обычная выгрузка списка УТХ-бригад из АРМа Технолога (путь для выгрузки из АРМа Технолога: _(левая панель) Перевозочный процесс - Бригадная модель - Бригады УТХ - (выбрать нужные сутки) - кнопка "Запуск" - кнопка "Выгрузить в Excel"_).
# 3. (Опционально) Отчет по отсевам по УТХ-бригадам. Это файл `otsev_uth_detail.csv`, он тоже должен располагаться в подпапке `./resources/others/`. Этот файл создается модулем отсевов (*(с) Варанкин*), на тестовых комплексах он располагается по пути `\server\bin\log\planner_filters\%папка с нужным запуском%\.`.
# 
# ## Варианты запуска скрипта
# 
# Скрипт можно запустить командой `python uth_report.py` из командной строки. По умолчанию будет построен отчет по станции Иркутск-Сортировочный, в отчет не будут добавлены последние операции с бригадами (на основании которых формировались входные данные для планировщика), сам отчет будет лежать в папке `report` в виде html-файла с названием `uth_report_%Date%_%Time%.html`, вместо %Date% и %Time% будут подставлены, соответственно, дата и время создания отчета.
# 
# Запуск можно модифицировать следующим образом:
# 
# 1. Запустить с ключом **`ops`: `python uth_report.py ops`**. В этом случае в отчет будут добавлены последние операции с бригадами из файла `./resources/others/otsev_uth_detail.csv` (см. п.3 в предыдущем разделе).
# 2. Запустить с ключом **`noprint`: `python uth_report.py noprint`**. Это косметическая штука: в этом случае в консоль не будут выводиться некоторые отладочные сообщения. Но поскольку таких сообщений не очень много, то использование этого ключа не критично.
# 3. Запустить с ключом **`zip`: `python uth_report.py zip`**. В этом случае после завершения построения отчета будет создан zip-файл.  Имя этого файла будет совпадать с названием файла с отчетом, а помимо собственно html-файла с отчетом в архив будет добавлен файл со стилями `skeleton.css`, который желателен для красивого отображения отчета в браузере.
#   1. Обычный запуск этого отчета, который я чаще всего делал, выглядел как `python uth_report.py ops noprint zip`. В этом случае применятся все три опции.
# 4. (Экспериментальная функция) Запустить с ключом вида **`"depot(%TCH%,%ST_NAME%)"`** (кавычки обязательны!), вместо %TCH% надо указать код ТЧЭ бригады (например, "ТЧЭ-1 В-СИБ" или "ТЧЭ-13 ДВОСТ" - коды можно посмотреть в файле `Бригады_УТХ*.xls` в столбце "Депо приписки"), вместо %ST_NAME% надо указать точное название станции (например, ТАЙШЕТ или ИРКУТСК-СОРТИРОВОЧНЫЙ, можно не капсом). В этом случае отчет будет строиться не для бригад депо ИРКУТСК-СОРТИРОВОЧНЫЙ, а для бригад депо %TCH%, отправляющихся со станции %ST_NAME%. 
#   1. Пример запуска с этим ключом: `python uth_report.py "depot(ТЧЭ-1 В-СИБ,ТАЙШЕТ)"`.
#   2. Можно запускать с несколькими ключами:  `python uth_report.py ops noprint zip "depot(ТЧЭ-1 В-СИБ,ТАЙШЕТ)"`
#   3. Запуск по умолчанию аналогичен запуску командой `python uth_report.py "depot(ТЧЭ-5 В-СИБ,ИРКУТСК-СОРТИРОВОЧНЫЙ)"`
#   
# Для удобства создан батник (лежит рядом) `uth_report.bat`, он сначала создает csv-файлы из лога планировщика, а затем формирует самый востребованный отчет (по Иркутску, без операций). Соответственно, для получения отчета надо разархивировать `jason-FullPlannerPlugin.log` в папку `input` и запустить батник. Разумеется, его можно модифицировать по своему усмотрению.
#   
# ## Известные подводные камни
# 
# 1. Кодировка файла `otsev_uth_detail.csv`. Сейчас это кодировка ANSI. При чтении файла нужная кодировка задается параметром `encoding`, сейчас нужная строчка скрипта выглядит так: `df_ops = pd.read_csv('input/' + ops_filename, sep=';', encoding='mbcs', dtype={'team_id':str})`. Надо на всякий случай следить за возможными падениями из-за смены кодировки. Именования кодировок в питоне можно посмотреть [здесь](https://docs.python.org/3/library/codecs.html#standard-encodings).
# 2. Для корректной работы скрипта нужный файл `Бригады_УТХ*.xls` **НЕ ДОЛЖЕН** быть открыт (в экселе). Иначе определение нужного (последнего) файла по маске имени сработает неправильно.

# In[186]:

import numpy as np
import pandas as pd
import time, datetime
import zipfile
from ast import literal_eval


# In[168]:

report = ''
FOLDER = 'resources/'
REPORT_FOLDER = 'report/'


# In[169]:

# Парсинг ключей запуска

import sys
JOIN_OPS, ZIP, PRINT, TCH, ST_NAME = False, False, True, 'ТЧЭ-5 В-СИБ', 'ИРКУТСК-СОРТИРОВОЧНЫЙ'
argv = sys.argv
if len(sys.argv) > 1:
    if 'ops' in sys.argv:
        JOIN_OPS = True
    if 'zip' in sys.argv:
        ZIP = True 
    if 'noprint' in sys.argv:
        PRINT = False
    if any(['depot' in arg for arg in sys.argv]):
        st = [arg for arg in argv if 'depot' in arg][0]        
        dep = st[6:-1]
        TCH, ST_NAME = [term.strip().upper() for term in dep.split(',')]


# In[170]:

time_format = '%b %d, %H:%M'
def nice_time(x):
    return time.strftime(time_format, time.localtime(x))


# ## Функции для создания html-файла
# 
# Весь отчет записывается в глобальную переменную `report` в html-разметке, для добавления строк используются методы `add_header` и `add_line`.

# In[171]:

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


# ## Загрузка результатов планирования из csv-файлов

# In[172]:

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


# In[173]:

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


# In[174]:

print('''--------
Возможные ключи: 
ops - добавляет в отчет последние операции с бригадами из последнего файла "Операции*.txt"
noprint - отключает вывод отладочных принтов
zip - архивирует отчет
--------''')


# ## Чтение списка УТХ-бригад

# ### Поиск нужного файла

# In[175]:

import os
files = [files for root, directories, files in os.walk('./resources/others')][0]
times = {}

# Тут сделано немного через жопу: сначала происходит os.chdir на нужную папку, а затем два раза возвращаемся обратно через
# os.chdir('..'). Наверняка это можно сделать более правильно.

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


# ### Загрузка бригад из файла в датафрейм

# In[176]:

import xlrd
uth = pd.read_excel('./resources/others/' + uth_filename)
uth.columns = ['Номер', 'Машинист', 'Депо', 'Вид движения', 'Факт.явка', 'План.явка']
uth['Вид движения'] = uth['Вид движения'].apply(lambda x: str(x).replace('\n\t\t\t', ';'))
uth['irk'] = uth['Депо'].apply(lambda x: TCH in x)
uth = uth[uth.irk]
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


# ## Чтение и обработка результатов планирования, выделение УТХ-бригад

# In[177]:

info_cols = ['number', 'name', 'loc_name', 'state', 'depot_time_norm', 'is_planned']
team_info['name'] = team_info.number.map(uth.set_index('Номер')['Машинист'])
team_info['uth_presence'] = team_info.number.map(uth.set_index('Номер').uth_presence)
team_info['depot_time_norm'] = team_info.depot_time.apply(lambda x: nice_time(x) if x !=-1 else x)
planned = team_plan[team_plan.state.isin([0, 1])].drop_duplicates('team')
team_info['is_planned'] = team_info.team.isin(planned.team)


# In[178]:

df_input_show = team_info[team_info.number.isin(uth['Номер'])][info_cols]
df_input_show.is_planned.replace(False, 'Нет', inplace=True)
df_input_show.is_planned.replace(True, 'Да', inplace=True)
df_input_show.columns=['Номер', 'Машинист', 'Исх.местоположение', 'Исх.состояние', 'Время явки', 'В плане?']
cols = ['Номер', 'Машинист', 'Депо', 'Вид движения', 'План.явка', 'uth_presence']
df_show = uth[cols].set_index(['Номер', 'Машинист']).join(df_input_show.set_index(['Номер', 'Машинист'])).fillna('-').reset_index()


# In[179]:

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


# In[180]:

add_line('Время сбора данных и запуска планировщика: %s' % time.strftime(time_format, time.localtime(current_time)))
add_header('Всего %d бригад депо %s загружено в ОУЭР из УТХ' % (uth['Номер'].count(), ST_NAME))
add_line('Из них:')
add_line('- передано в планировщик: %d' % team_info[team_info.number.isin(uth['Номер'])].team.count())
add_line('- не передано в планировщик: %d' % uth[uth['Номер'].isin(team_info.number) == False]['Номер'].count())
add_line('- запланировано: %d' % df_output_show['Номер'].count())
df_show_uth_plan = df_show.set_index(['Номер', 'Машинист']).join(df_output_show.set_index(['Номер', 'Машинист'])).fillna('-')


# ## Чтение данных о последних операциях

# In[191]:

files = [files for root, directories, files in os.walk('./resources/others')][0]
ops_filename = 'otsev_uth_detail.csv'
if ops_filename in files:
    df_ops = pd.read_csv('input/' + ops_filename, sep=';', encoding='mbcs', dtype={'team_id':str})    
    if 'Номер' not in df_show_uth_plan.columns:
        df_show_uth_plan = df_show_uth_plan.reset_index()    
    df_show_uth_plan['team'] = df_show_uth_plan['Номер'].map(team_info.drop_duplicates('number').set_index('number').team)
    df_show_uth_plan['oper_id'] = df_show_uth_plan.team.map(df_ops.drop_duplicates('team_id')
                                                            .set_index('team_id').team_type_asoup_id)
    df_show_uth_plan['oper_name'] = df_show_uth_plan.team.map(df_ops.drop_duplicates('team_id')
                                                              .set_index('team_id').team_type_name)
    df_show_uth_plan['Посл.операция'] = df_show_uth_plan.apply(lambda row: '(%s) %s' 
                                                               % (row.oper_id, row.oper_name), axis=1)
    df_show_uth_plan['Время посл.оп.'] = df_show_uth_plan.team.map(df_ops.drop_duplicates('team_id')
                                                                   .set_index('team_id').team_time)    
    df_show_uth_plan['Место посл.оп.'] = df_show_uth_plan.team.map(df_ops.drop_duplicates('team_id')
                                                                   .set_index('team_id').team_location_name)    


# ## Формирование результирующей таблицы

# In[182]:

if JOIN_OPS:    
    show_cols = ['Номер', 'Машинист', 'Депо', 'Вид движения',  
             'Посл.операция', 'Время посл.оп.', 'Место посл.оп.',
             'План.явка', 'Исх.местоположение', 'Исх.состояние', 'Время явки', 'В плане?', 'Ст.отпр.', 'Ст.направл.', 
             'Время отпр.', 'Состояние', 'Номер ЛОК', 'Номер П', 'Все состояния']
else:
    show_cols = ['Номер', 'Машинист', 'Депо', 'Вид движения',              
             'План.явка', 'Исх.местоположение', 'Исх.состояние', 'Время явки', 'В плане?', 'Ст.отпр.', 'Ст.направл.', 
             'Время отпр.', 'Состояние', 'Номер ЛОК', 'Номер П', 'Все состояния']

res_to_index_start_with_0 = df_show_uth_plan.sort_values(['uth_presence']).reset_index()[show_cols].reset_index()

# Два reset_index() в предыдущей строке и следующая строчка нужны только для того, чтобы в таблице появилась нумерация строк с 1
res_to_index_start_with_0['index'] = res_to_index_start_with_0['index'] + 1

add_line(res_to_index_start_with_0, p=False)


# In[183]:

not_input = res_to_index_start_with_0[res_to_index_start_with_0['В плане?'] == '-']
not_planned = res_to_index_start_with_0[res_to_index_start_with_0['В плане?'] == 'Нет']
add_header('Не переданные бригады:')
add_line(list(not_input['Номер'].unique()))
add_header('Не запланированные бригады:')
add_line(list(not_planned['Номер'].unique()))


# In[184]:

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


# In[185]:

add_state_legend()
filename = REPORT_FOLDER + 'uth_report_' + time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())) + '.html'
create_report(filename)
if ZIP:
    create_zip(filename)

