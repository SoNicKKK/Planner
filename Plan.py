﻿
# coding: utf-8

# # Заметки о работе и проверке модуля планирования

# ## Как проверять работу планировщика на тестовых комплексах
# 
# ### Какие комплексы есть
# 
# Всего есть четыре сервера (точнее, комплекса: каждый из них представляет из себя несколько серверов), на которых запускается наш планировщик:
# 
# 1. **Боевой комплекс**. Это продакшен-сервер, находится в Иркутске (или в Екатеринбурге - тут я путаюсь, но это и неважно). На него ставятся релизные сборки, которые (в идеале) выпускаются раз в две недели. Между выпуском версии и установкой сборки проходит какое-то время, пока ее проверяют сотрудники в Иркутске, так что вполне могут быть случаи, что на бою оказывается какая-то очень древняя версия планировщика. Большого внимания бою уделять не стоит, разве что кто-то очень попросит (Немцов). Если на бою выявляется какой-то критический баг, требующий новой версии планировщика, то его надо исправлять патчем (= отправкой новой версии админам). Про патчи - см. ниже раздел "Как готовить планировщик для релизной сборки".
# 
# 2. **ТК1** ("первый тестовый/тренажерный комплекс"). У него статус очень похож на статус боевого: туда тоже ставятся релизные сборки, где они проверяются сотрудниками в Иркутске перед выкладкой на бой. Механизм нашей работы с ним примерно аналогичен бою:  тоже обновления только патчами, очень тщательно следить не стоит (если только по требованию руководства).
# 
# 3. **ТК2** ("второй тестовый/тренажерный комплекс"). Это наш тестовый комплекс, где мы проверяем сборки перед релизом. Именно на него, в основном, смотрят технологи и проверяют цифры, подвязку, показатели и пр. Его надо мониторить (я обычно это делаю по утрам, сразу после прихода) просто на предмет того, идут ли расчеты планировщика, все ли в порядке с данными (см. раздел "На что смотреть для быстрой проверки").
# 
# 4. **Нижегородка**. Это отладочный комплекс, который расположен в офисе НИИАС на Нижегородке. В отличие от остальных комплексов, сюда могут свободно ходить разработчики, подкладывать свои файлы и скрипты, что-то подкручивать и тестировать (в рамках разумного). С выпуском новых официальных релизов версия на Нижегородке (иногда можно встретить сокращение НК = "нижегородский комплекс") обновляется. 
# 
# ### Где брать логи
# 
# #### Лог обмена данными
# 
# Главное наше оружие при проверке каких-то ошибок в планировании - это лог обмена данными между планировщиком и Вектором, файл `jason-FullPlannerPlugin.log`. С ТК1, ТК2 и НК его можно достать самостоятельно. Для этого нужен компьютер из сети СТТ - и такой компьютер у нас есть, это НИИАСовский компьютер (с одним монитором), который стоит под моим столом справа (далее в этом тексте он будет фигурировать как "СТТ-компьютер"). На нем надо открыть программу `WinSCP` (значок есть на рабочем столе), в появившемся окне выбрать один из серверов (надо смотреть на названия, они подписаны: TK1, TK2, NIZH), залогиниться. Откроется обычное двухпанельное окно файлового менеджера, справа будет файловая система на соответствующем сервере планировщиков, там можно найти нужный файл и скопировать на этот локальный компьютер. Дальше можно на флешке переносить файл куда угодно.
# 
# Логи лежат по пути `.../rvec/server/bin` (перед `rvec` может быть какой-то еще фрагмент пути). В этой папке всегда находится один незаархивированный файл `jason-FullPlannerPlugin.log` и несколько заархивированных `jason-FullPlannerPlugin*.zip`. Вместо звездочек в название архива подставляется **реальное время запуска самого расчета планировщика**. Подробнее про то, что это за время - см. раздел ["Из чего состоит расчет на серверах"](#Из-чего-состоит-расчет-на-серверах).
# 
# #### Логи отсевов.
# 
# Логи модуля отсевов ((с) Варанкин) лежат на тех же серверах по пути `.../rvec/server/bin/log/planner_filters`. Там нас интересуют два файла:
# 
# * `otsev_detail.csv` - это последние операции со всеми поездами, локомотивами и бригадами, которые были выбраны исходным SQL-запросом (потом к этим бригадам были применены правила отсевов, некоторые бригады отсеялись). Про формат этого файла, думаю, подробнее напишет Сергей Варанкин.
# * `otsev_uth_detail.csv`. Это тот же файл, но в нем оставлены только операции по бригадам, переданным нам из УТХ на текущие сутки.
# 
# #### Логи самого планировщика.
# 
# Логи, которые создает сам планировщик при расчете, лежат в `.../rvec/server/bin/log/uni_planner_log`. 
# 
# #### Логи авторана.
# 
# В эти файлы пишется (крупноблочно) процесс хода расчета (см. ["Из чего состоит расчет на серверах"](#Из-чего-состоит-расчет-на-серверах)) с указанием времени, сколько заняла та или иная часть. Собственно работа планировщика - это строка вида *"... ЕПР: сформирован план ... поездов, ... локомотивов, ... бригад за ..."*. Из этой строки можно узнать, сколько объектов вернул планировщик (slot_train, slot_loco, slot_team) и сколько времени занял сам расчет. Все строки выше этой относятся к подготовке данных. 
# 
# Важный нюанс. Если в процессе подготовки данных случился какой-то эксепшен, то планировщик просто не запустится. Но в авторане появится строчка "Запуск планировщиков выполнен за ...". Это известный баг логирования, про который я писал очень давно. На самом деле планировщик в этом не запускается. Выявить эти случаи можно по соответствующему файлу `jason-FullPlannerPlugin.log`- в нем будет только шапка без каких-либо данных.
# 
# #### Логи подготовки данных.
# 
# Это сугубо векторовские логи, которые нам смотреть необязательно. Я их смотрел только для того, чтобы побыстрее выявить эксепшены, которые возникли при подготовке данных (они не запишутся ни в авторан, ни в лог отсевов, их можно выявить только через векторовский лог). Лог можно найти там же по пути `.../rvec/conf_fast_start/.netbeans-config/var/log`. Здесь нас интересуют файлы с именами вида `*Server_DEV_*` (вместо * может стоять что угодно). Этих файлов может быть несколько (они создаются в начале суток и каждый раз при перезапуске сервера планировщиков). Надо (по дате изменения) выбрать нужный, затем по таймстемпам найти нужный временной фрагмент и начать его листать (лог большой, там много мусора), выискывая глазами всякие стектрейсы. Иногда можно что-то найти, иногда ошибка спрятана глубоко и быстро визуально не обнаруживается.
# 
# ### Доступ к веб-клиенту
# 
# На результаты планирования можно посмотреть через веб-клиент. Там можно по каждой станции посмотреть все поезда, которые мы запланировали к прибытию или отправлению на эту станцию, посмотреть время прибытия/отправления, локомотив и бригады, с которыми поезд прибывал и отправлялся. Технологи только так и проверяют результаты - так что и нам придется это использовать, чтобы как-то проверить сказанное технологами. 
# 
# Для каждого из комплексов (см. выше) на СТТ-компьютере в хроме установлена закладка (просто см. по названию). Заходить можно под логином/паролем tsyganova/tsyganova. Далее в верхнем меню надо нажать кнопку "Планирование", выбрать нужную станцию, нужный план (выпадающий список "Версия") и нажать кнопку "Обновить". Назначение остальных фильтров интуитивно понятно (после выбора надо снова нажимать "Обновить"). 
# 
# Про возможности формы отображения планов лучше расскажет Оля Цыганова. Кратко скажу, что в таблице три секции: справа отображается информация по поезду на начало планирования (номер, индекс, вес, длина, последняя операция), далее - информация по планируемому прибытию поезда на выбранную станцию (время, локомотив, бригада), далее - информация по отправлению поезда с выбранной станции. 
# 
# Если вы не видите какого-то поезда на форме, то надо:
# * проверить фильтр "Чет/нечет" - возможно, поезд едет в нечетную сторону, а выбран фильтр "ЧЕТ".
# * проверить фильтр "Интервал". В нем выбирается горизонт (от начала планирования) для "обрезания" плана. Если выбран интервал "6 часов", то будут показаны операции прибытия и отправления, запланированные не позднее чем через 6 часов от времени начала планирования.
#   * если операция с поездом планируется более чем через 24 часа от начала планирования, то увидеть его в форме невозможно.
# 
# ### На что смотреть для быстрой проверки
# 
# Быстро оценить, что происходит, можно просто по размеру лога `jason-FullPlannerPlugin.log` или соответствующего архива. Возможны четыре очевидные ситуации:
# 
# 1. Лог имеет размер 35...45 Мб (архив - 3.5...4 Мб). Тут все в порядке - данные в планировщик передались, расчет был завершен. С технической стороны тут все в порядке, далее имеет смысл изучать конкретные планы и логи.
# 2. Лог имеет размер 15...20 Мб (архив - 1...2 Мб). Это означает, что в планировщик были переданы не все данные. Скорее всего, была передана только нормативно-справочная информация, а вот актуальных данных по поездам, локомотивам и бригадам вообще нет. Такое чаще всего происходит, если отвалился поток данных из АСОУП в онтологию. Планировщик, скорее всего, в этом случае вернет сколько-то поездов (это будут фейковые поезда по заданиям `task` на поезда своего формирования), 0 локомотивов и 0 бригад. Что делать: администраторам чинить поток данных.
# 3. Лог имеет размер несколько килобайт (архив - несколько сот байт). Это означает, что данные в планировщик вообще не были переданы. При этом в логе будет только шапка - и больше ничего. Это свидетельствует о том, что случился необработанный эксепшен в процессе подготовки данных для планировщика, сам процесс не был завершен, данные не были сформированы. **Важное замечание**: при этом в логе авторана все равно появится строчка "Запуск планировщиков выполнен за ...". Это известный баг логирования, про который я писал очень давно. На самом деле планировщик в этом не запускается. Что делать: кому-то из группы онтологии искать и исправлять ошибку.
# 4. Лога (архива) нет вообще. Это означает, что сломался процесс запуска планировщика по расписанию. Такое иногда происходит из-за какой-то сложной ошибки вроде переполнения памяти и т.п. Что делать: кому-то из группы онтологистов или администраторов искать и исправлять ошибку, перезапускать комплекс.
# 
# ### Разные нюансы при проверке результатов планирования
# #### Индексы поезда
# 
# На форме в веб-клиенте указываются индексы поездов в формате 4-3-4. По ним эти поезда надо искать в логах. Соответствие id поезда и индекса пишется в начале лога (блок `Поезда (obj_id = индекс поезда; номер поезда; источник):`, обычно начинается со строчки 363 лога или около того). Проблема в том, что в логе пишутся индексы в 15-значном формате. Правило для поиска такое: если индекс поезда идет в формате `ABCD-EFG-HIJK` (4-3-4), то соответствующий ему 15-значный индекс будет таким: `ABCD0*EFGHIJK0*`, где вместо * может стоять какая-то любая цифра (после D и K идут нули).
# 
# Если кто-то из технологов говорит, что поезда с каким-то индексом нет в форме (хотя он должен быть), то надо поискать в логах поезд с точно таким же индексом, а также поезд, у которого совпадают только первые две части (4-3...), а "хвост" (последние 4 цифры - `HIJK` - другой). Такое вполне может быть, у поездов часто меняются концовки индексов (концовка индекса представляет собой 4 цифры ЕСР-кода станции назначения, станция назначения поезда может меняться по ходу следования).
# 
# Если для проверки использовать не файл с логом, а мои тестовые скрипты, то там все проще: индексы преобразуются при парсинге логов, в файле `train_info.csv` (который создается после завершения работы скрипта `read.py`) будет колонка `ind434`, где будет индекс в формате 4-3-4.
# 
# #### Фамилии бригад
# 
# На формах в веб-клиенте **не надо** искать бригады по фамилиям. Дело в том, что в этих фамилиях перемешаны кириллица и латиница, поэтому можно легко не найти бригаду, хотя она есть. Для поиска надо использовать табельный номер.
# 
# #### Сменно-суточное планирование
# 
# На форме планирования можно переключиться на горизонт "сменно-суточное". По сути, это тот же план, возвращенный планировщиком, но агрегированный, без конкретики по поездам/локомотивам/бригадам. В нем просто посчитано количество тех или иных поездов, локомотивов или бригад (подробнее см. форму) - отправленных или принятых, ушедших на ТО, ушедших или вышедших с отдыха и т.п. Количества посчитаны по каждому часу ж/д суток, посчитаны суммы по полусуткам и суткам. 
# 
# В основном, сменно-суточное планирование нужно для того, чтобы правильно оценить количество бригад, которое должно выйти на работу в каждом депо в следующие сутки. Эти количества передаются планировщику УТХ, планировщик УТХ подбирает конкретные бригады под это количество и возвращает пофамильный список. При следующем расчете этот список будет передан в наш планировщик (эти бригады будут переданы с признаком `uth(1)`) - и их надо будет планировать в первую очередь (этот механизм уже реализован).
# 
# Форма сменно-суточного планирования нуждается в тестировании - я не могу сейчас ручаться, что цифры на ней считаются абсолютно правильно и что они полностью соответствуют текущему плану.
# 
# #### Обновления иногда не переносятся
# 
# Это пункт "немного в сторону", но про него надо помнить. Из-за того, что у нас, по сути, четыре разных комплекса, то иногда бывает сложно уследить, какие доработки и обновления на каких комплексах установлены. Например, какой-то баг во входных данных был исправлен на НК, но не был добавлен в релизные метки (см. ниже) или не был добавлен патчем на ТК1-2 (тоже см. ниже). Потом случилось обновление комплексов - и эта доработка отовсюду исчезла. Надо это держать в голове и воспринимать спокойно. Если что, версию планировщика всегда можно посмотреть по метке в логах (поиск в логе по слову `version`) или посмотреть время изменения jar-файла планировщика (он лежит по пути `server/bin/jason/`).

# ## Из чего состоит расчет на серверах

# ## Как готовить планировщик для релизной сборки
# 
# ### На каких данных
# 
# Собрать джарник на сервере
# 
# Прогнать на имитации, посмотреть проценты подвязки, кол-во локомотивов резервом, времена стоянок на смену Л и ЛБ, процент фейковых.
# 
# Прогнать на реальных данных, проверить то же самое.
# 
# Опционально: проверить на "проблемных" данных (долго - 2 часа)
# 
# #### Батники и скрипты на сервере
# 
# ### Что проверять
# 
# ### Как и куда выкладывать
# 
# ТФС, labels.txt, патчи

# ## Нужные люди
# 
# ### Технологи
# 
# Войтенко, Капустин, Фрольцов, Семенчук, Мищенко
# 
# Алтунин, Жемионис
# 
# ### Аналитики
# 
# Цыганова, Есаков
# 
# ### Программисты
# 
# Костенко, Варанкин-Тугушев, Ежов, Типцова
# 
# ### Тестировщики
# 
# Бураева

# ## Регламент запуска планировщика

# ## ТСТ

# ### Где могут быть проблемы в подвязке
# 
# Весовые нормы
# 
# Участки обкатки
# 
# Локомотивам прописывается несколько тяговых плеч

# ## Показатели
# 
# ### Какие важны
# 
# Производительность локомотивов
# 
# Среднее время стоянки для смены Л (ЛБ)
# 
# Проценты подвязки
# 
# Полезный пробег и процент полезного использования Л
# 
# ### Какие есть подгоны
# 
# Производительность: отбрасывание коротких маршрутов, локомотивов резервом, локомотивов, которые вторую половину суток стоят без работы
# 
# Аналогичные подгоны для полезного пробега и процента полезного использования.
# 
# ### Что может влиять
# 
# 
# 
# ### Разница между скриптами и анализатором
# 
# Есть расхождения почти во всех показателях. Надо выделить время и проверить подсчет процентов подвязки и производительности в анализаторе и в скриптах.
