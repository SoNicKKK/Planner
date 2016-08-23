
# coding: utf-8

# <a id='head'></a>
# # Планы работ на ближайшее время (3-4 недели)
# 
# 1. [Задачи по планированию локомотивов](#Задачи-по-планированию-локомотивов)
#   1. [Пост-планирование и создание фейковых локомотивов](#post)
#   2. [Проверка длительных стоянок поездов под смену локомотивов на реальных данных](#Проверка-длительных-стоянок-поездов-под смену-локомотивов-на-реальных-данных)  
#   3. [Подвязка нескольких локомотивов под один поезд](#Подвязка-нескольких локомотивов-под-один-поезд)
#   4. [Планирование ТО локомотива вместе с расчетом функции полезности и вместе с подвязкой](#Планирование-ТО-локомотива-вместе-с-расчетом-функции-полезности-и-вместе-с-подвязкой)
#   5. [Объединение локомотивов резервом в сплотки](#Объединение-локомотивов-резервом-в-сплотки)  
# 2. [Задачи по планированию бригад](#Задачи-по-планированию-бригад)
#   1. [Фиксация времени явки бригад от предыдущего расчета. Доработка временных интервалов фиксации](#prev_team)
#   2. [Подвязка поездов точно на нитки (по временам отправления/прибытия)](#slot_time)
#   3. [Исправления в обработке входных данных](#Исправления-в-обработке-входных-данных)
#   4. [Определение депо приписки у фейковых бригад](#Определение-депо-приписки-у-фейковых-бригад)
#   5. [Планирование бригад с исходным state = 2](#state2)
#   6. [При расчете времени отдыха учитывать, что явка могла быть до начала планирования](#early)  
#   7. [Переработки бригад](#Переработки-бригад)
# 3. [Задачи по исправлению входных данных](#Задачи-по-исправлению-входных-данных)
#   1. [Не передаются три станции](#Не-передаются-три-станции)
#   2. [Сдвоенные поезда](#Сдвоенные-поезда)
#   3. [Неправильные местоположения поездов, локомотивов или бригад](#Неправильные-местоположения-поездов,-локомотивов-или-бригад)
#   4. [Станция Зыково не принадлежит ни одному участку обкатки бригад](#Станция-Зыково-не-принадлежит-ни-одному-участку-обкатки-бригад)
#   5. [Вывозные локомотивы](#Вывозные-локомотивы)  

# Так исторически сложилось, что задачи в части планирования локомотивов делает **Боря**, а в части планирования бригад - **Костя**. Примерно такое разделение и планируется по задачам на ближайшее будущее. Правда, на мой взгляд, загрузка Бори в ближайшее время получается больше. Посмотрите задачу "Подвязка нескольких локомотивов под один поезд" - возможно, ее удастся как-то распараллелить.
# 
# ## Задачи по планированию локомотивов
# 
# <a id='post'></a>
# ### Пост-планирование и создание фейковых локомотивов
# 
# У нас пока что не стопроцентная подвязка локомотивов под поезда: бывают случаи, когда планировщик не смог найти локомотив под поезд. Это приводит к тому, что и бригада под такой поезд найдена не будет. К сожалению, это не совсем устраивает УТХ. После расчета в 14:40 мы должны выдать в УТХ объемный план по бригадам на следующие сутки: сказать, сколько бригад каждого депо приписки понадобится. Соответственно, если мы к каким-то поездам даже не пытаемся подвязать бригаду (а это как раз поезда без локомотивов), то объемный план для УТХ будет занижен на несколько бригад. Предлагаю сделать фазу пост-планирования: после завершения всех туров подвязки локомотивов и бригад анализировать, к каким поездам мы не нашли локомотивы/бригады, и создавать фейковые локомотивы и бригады. Подробное описание приведено в [задаче в JIRA VP-6980](http://jira.programpark.ru/browse/VP-6980). Предложение, разумеется, обсуждаемо - в первую очередь, нужно мнение Бори.
# 
# Это текущая задача Бори.
# 
# ### Проверка длительных стоянок поездов под смену локомотивов на реальных данных
# 
# Надо проверить и исправить (или для начала хотя бы - найти причину) подвязки локомотивов к поездам, из-за которых возникают стоянки поездов на смену локомотивов по 10+ часов. Примеры таких длительных стоянок можно смотреть в поездном отчете: надо найти подзаголовок "Всего ... случаев смены локомотива с длительной стоянкой" - там будут приведены примеры таких поездов и полный план по первому поезду с длительной стоянкой.
# 
# 
# ### Подвязка нескольких локомотивов под один поезд
# 
# Сейчас все наши структуры данных и алгоритмы планирования заточены под то предположение, что к одному поезду может быть прикреплен только один локомотив. Вообще говоря, это не так. К длинному и тяжелому поезду можно прикрепить один очень мощный локомотив, а можно - два обычных. Кроме того, есть сдвоенные поезда (поезда весом до 12 000), которые в принципе можно везти только двумя локомотивами. 
# 
# Поэтому надо сделать ряд доработок. Во-первых, доработать структуры данных, чтобы у нас была возможность хранить и возвращать такие подвязки. Во-вторых, немного изменить обработку входных данных: сейчас, если у нас на входе два локомотива ссылаются на один и тот же поезд, то "останется только один", а второй будет безвозвратно потерян. Второй локомотив терять не надо, надо довозить поезд до границы тягового плеча в два локомотива. В-третьих, надо исправить собственно алгоритм подвязки: разрешить поиск двух локомотивов под тяжелый поезд, если нет тяжелого локомотива (это менее оптимальная операция, поэтому так должно происходить только в крайних случаях). 
# 
# Пункт "в-третьих" можно на время отложить и реализовать сначала первые два. Так мы хотя бы будем гарантировать, что мы не теряем какие-то локомотивы в планировании и правильно отображаем исходную подвязку, а назначить два локомотива под поезд диспетчер, в крайнем случае, сможет и вручную. **Андрей**, тут от тебя потребуется проработка постановки задачи. **Боря, Костя**, попробуйте, по возможности, разделить реализацию друг с другом (поскольку, как мне кажется, загрузка Бори в ближайшее время получается уж очень большой).
# 
# ### Планирование ТО локомотива вместе с расчетом функции полезности и вместе с подвязкой
# 
# Сейчас в том случае, если локомотиву не хватает времени до ТО, чтобы ехать с поездом до границы тягового плеча, то этой паре <поезд, локомотив> ставится полезность $-\infty$. Только после окончания планирования на текущем фрейме происходит проверка неподвязанных локомотивов на предмет того, а  не стоит ли локомотивы отправить на ТО. Получается не совсем оптимальное планирование: вполне можно было провести локомотиву ТО и подвязать его под какой-то поезд на текущем фрейме, а вместо этого мы подвязываем под поезд, возможно, очень плохой локомотив, хороший локомотив оставляем без подвязки, ТО хорошему локомотиву заказываем пост-фактум.
# 
# Поэтому предлагается планировать локомотиву ТО одновременно с подвязкой и вычислением функции полезности. Подробно свое видение я изложил в [задаче в JIRA VP-6961](http://jira.programpark.ru/browse/VP-6961). Подводные камни в реализации присутствуют: видимо, чтобы протащить ТО через пред-планирование, надо будет задействовать систему хинтов. Надо все сделать аккуратно. Ожидаемый профит: увеличатся проценты подвязки локомотивов, будут уменьшены стоянки поездов для смены локомотивов, возможно, увеличится количество ТО (это нормально). **Андрей**, постановочная часть тут твоя. Разумеется, обсуждение постановки с Борей/Костей только приветствуется.
# 
# ### Объединение локомотивов резервом в сплотки
# 
# Сейчас на каждый локомотив резервом создается отдельный поезд и каждый такой локомотив требует отдельную бригаду. Это не всегда верно. Если несколько локомотивов резервом планируются отправлением примерно в одно и то же время, то их можно объединить в один поезд. При этом бригаду достаточно сажать на каждый нечетный локомотив (одни бригада при объединении двух локомотивов, две - при объединении трех и т.д.). Свое видение я изложил в документе, который лежит в ТФС по пути `Восточный полигон/Планировщики/ПланировщикиJason/docs/! Алгоритм планирования/Мини-постановки/Локо на чужих ТП.docx`. Но он написан довольно давно, его надо критически переосмыслить. **Андрей**, это тоже на тебе.

# [В начало](#head)
# ## Задачи по планированию бригад
# 
# <a id='prev_team'></a>
# ### Фиксация времени явки бригад от предыдущего расчета. Доработка временных интервалов фиксации
# 
# Сейчас реализовано следующим образом. Берем время начала планирования $t_0$, прибавляем три часа: все предыдущие явки, которые попали в интервал $(t_0, t_0 + 3)$ фиксируем - их сдвигать нельзя. Все предыдущие явки из интервала $(t_0+3, t_0+6)$ можно сдвигать не более, чем на три часа. Остальное можно сдвигать практически как угодно. (На самом деле это реализовано так: при расчете функции полезности для бригады ставим полезность $-\infty$ поезду, если время его готовности к отправлению превышает время предыдущей явки бригады плюс допустимую задержку).
# 
# Технологи подумали - и решено было интервалы немного поменять. Каждые сутки мы разбиваем на трехчасовые интервалы: с 0 до 3, с 3 до 6, ..., с 21 до 0. Далее находим такой трехчасовой интервал, время начала которого находится ближе всего в будущем к времени начала планирования $t_0$. Например, если время начала планирования - 19:30, то ближайшим трехчасовым интервалом будет интервал 21...0 текущих суток, а если время начала планирования - 22:30, то ближайший трехчасовой интервал - с 0 до 3 следующих суток. Далее для предыдущих явок, которые лежат в интервале от $t_0$ до правой границы этого ближайшего трехчасового интервала, допустимый сдвиг явок $D = 0$, для явок, которые лежат в следующем трехчасовом интервале, $D = 3$ (в часах), для более поздних явок $D = 24$.  Почему так надо сделать: диспетчеры в своем планировании ориентируются именно на эти интервалы, а не на время запуска планировщика. Поэтому фиксацию надо проводить именно по таким интервалам.
# 
# Это желательно сделать в понедельник (и выложить джарник на ТК2 - можно без добавления в labels.txt, просто коммит и патч). Понимаю, что Костя прямо сейчас занят другой задачей, но вроде мы договоривались, что она будет идти в отдельной ветке, а описанные выше доработки не должны занять много времени.
# 
# ### Подвязка поездов точно на нитки (по временам отправления/прибытия)
# 
# Это текущая задача Кости, она описана в [задаче в JIRA VP-6992](http://jira.programpark.ru/browse/VP-6992). Ее желательно закончить к четвергу (25.08.2016), чтобы, если в четверг вдруг будет собираться релиз, мы смогли это в него включить. После реализации надо проверить, что ничего не поломалось (тестовыми отчетами). Возможно, надо будет написать новый тест, который будет выявлять случаи, когда время отправления/прибытия поезда не соответствует времени из какой-либо нитки. Такие случаи допустимы (если на участке планирования вообще нет ниток или они все лежат в прошлом относительно времени отправления поезда), но их надо исследоват.ь
# 
# ### Исправления в обработке входных данных
# 
# У нас иногда происходит слишком жесткая отсечка входных данных. В некоторых случаях надо ее смягчить:
# 
# 1. Не отсекать бригаду, если время операции по бригаде не совпадает с временем операции у локомотива. Если у них при этом совпадают местоположения, то в качестве верного брать максимальное время из двух. Аналогично - для связки поезд-локомотив.
# 2. Если у бригады время явки в депо приписки больше времени последней операции, то значит, что эта бригада в данный момент завершает один цикл работы, затем уйдет на домашний отдых, но планировщик УТХ ее запланировал на явку в конце следующих суток (через 16 часов после ухода на отдых). Такие бригады не надо исключать из планирования после прибытия в депо приписки. Их надо делать доступными, начиная от указанного времени явки.
# 
# ### Определение депо приписки у фейковых бригад
# 
# Если создается фейковая бригада на станции, для которой `norm_time == 0`, то в качестве депо приписки этой бригады указывать станцию с `norm_time != 0`, находящуюся на маршруте следования локомотива. Если такой нет, то таки да, брать в качестве депо начальную станцию.
# 
# Сейчас в качестве депо приписки всегда ставится станция, с которой бригада начинает свой маршрут. Это не всегда верно: например, маршрут бригады может начинаться на маленькой станции, где вообще нет своих бригад, а мы назначим эту станцию как депо приписки, будем стараться вернуть фейковую бригаду на эту станцию и значит, у какого-то локомотива, возможно, запланируем смену бригады на этой маленькой станции. Это может быть верно с реальными бригадами, но с фейковыми лучше планировать попроще, без смен на мелких станциях. На эту тему есть [задача в JIRA VP-5535](http://jira.programpark.ru/browse/VP-5535).
# 
# <a id='state2'></a>
# ### Планирование бригад с исходным state = 2
# 
# Это бригады, которые на начало планирования уже явились в депо на явку, но еще не прикреплены ни к одному поезду. Такие бригады надо подвязывать с большим приоритетом (они ведь уже явились на работу). Полагаю, тут надо а) добавлять какое-то слагаемое в функцию полезности для такой бригады и локомотивов, у которых время готовности меньше времени явки бригады (или времени начала планирования?) - тогда бригаде вообще не придется ждать; б) выяснить у технологов, какое максимальное время может находиться в депо такая бригада без подвязки, отсекать локомотивы, у которых время готовности больше, чем время явки бригады + это допустимое время пересидки. Постановочная часть по этому вопросу - **за Андреем**.
# 
# Кроме того, сейчас бывают случаи, что в результатах планирования в маршруте какой-то бригады есть только один трек - и это трек со state = 2. По сути, это и означает то, что бригада пришла на явку, покрутилась - и ушла. Такие бригады вообще не надо возвращать в результатах планирования (а если это фейковая бригада, то вообще ее не создавать), а добавлять бригаду в `workless_team`.
# 
# <a id='early'></a>
# ### При расчете времени отдыха учитывать, что явка могла быть до начала планирования
# 
# Я замечал случаи, когда отдых бригад (трек с состоянием 4) длился меньше времени, чем необходимо. А необходимое минимальное время отдыха - это половина последнего рабочего времени от времени последней явки до времени начала отдыха. У меня есть подозрение, что когда мы рассчитываем это рабочее время (чтобы потом посчитать половину) для бригад, у которых явка была до начала планирования (и мы ее не запланировали сами, а взяли из входных сообщений `fact_team_ready`), то в качестве времени явки мы берем не переданное нам время явки, а время начала планирования. Это неверно - надо еще раз проверить, найти ошибку и исправить. Посмотреть примеры можно в бригадном отчете (team_report*.html), раздел "Проверка времен отдыха бригад", подзаголовок "Примеры бригад без переработки с недостаточным отдыхом". В столбце `min_rest_time` там пишется минимальное рассчитанное рабочее время, в `rest_time` - запланированное рабочее время. Считать `min_rest_time`, исходя из результатов планирования и входных данных, не так просто, так что я мог где-то ошибиться.
#   
# ### Переработки бригад
# 
# Попадаются случаи, когда время работы бригад (от явки до ухода на отдых) превышает нормативы. Такое время не должно превышать 12 часов в самом худшем случае. В бригадном отчете есть раздел "Бригады с переработкой", там приведены примеры. Их надо проанализировать, найти ошибки и исправить. Возможно, ошибка не программная, а какая-то "постановочная" - **Андрей**, тут понадобится твоя помощь.

# [В начало](#head)
# ## Задачи по исправлению входных данных
# 
# Это задачи на онтологистов - Сергея Варанкина и того, кто будет вместо него (Даниил Тугушев).
# 
# В целом, по проверке входных данных есть отчет `_input_report`. В идеале, все тесты в нем должны проходить без ошибок. Описание тестов на входные данные можно найти в документе в ТФС по пути `Восточный полигон\Планировщики\ПланировщикиJason\docs\! Алгоритм планирования\Автоматические тесты.docx`.
# 
# ### Не передаются три станции
# 
# [JIRA VP-6730](http://jira.programpark.ru/browse/VP-6730)
# 
# В планировщик не поступают сообщения `+station` о станциях Биробиджан I, Биробиджан II и Камышовая. Поскольку в этом сообщении указывается id тягового плеча, которому принадлежит станция, то это приводит к тому, что в соответствующих тяговых плечах получаются дырки. Поэтому локомотивы и бригады не планируются на участках возле этих станций - все маршруты обрываются на подступах к этим станциям. Следствия могут быть разнообразными: например, увеличится количество пересылок локомотивов резервом, вырастет количество случаев проезда бригадами мимо станций смены и т.п.
# 
# **Это было исправлено Сергеем, надо проверить, что это попало на ТК2**.
# 
# ### Сдвоенные поезда 
# 
# [JIRA VP-6015](http://jira.programpark.ru/browse/VP-6015)
# 
# Плохое качество данных:
# * По многим поездам, которые указаны в атрибутах `joint` у составляющих поездов, не передаются сообщения `train_info` и сообщение об операции.
# * Много случаев, когда определенный `id` сдвоенного поезда указан только для одного составляющего поезда (второй либо не передан, либо идет с `joint(-1)`).
# 
# Тесты на входные данные по сдвоенным поездам есть в отчете `_input_report`.
# 
# ### Неправильные местоположения поездов, локомотивов или бригад
# 
# Есть случаи (около 2% поездов и почти все бригады в `state(0)`), когда станция отправления бригады совпадает со станцией направления. Примеры - в отчете `input_report`.
# 
# Около 4% случаев, когда участка местоположения нет в маршруте поезда. Видимо, это связано с тем, что кратчайший маршрут строится от станции отправления без учета станции направления. Предлагаемое решение: строить маршрут от станции направления, а потом вручную прибавлять к нему в начало станцию отправления. Возможные подводные камни: оптимальный маршрут от станции направления может пролегать в обратном направлении - к станции отправления.
# 
# Около 20% от поездов, находящихся в пути, имеют очень старое время операции (есть случаи, когда время операции на 20 часов меньше времени начала планирования). Надо разобрать эти случаи, выяснить причины. По результатам анализа принять решения о доработках. Примеры - в отчете `input_report`.
# 
# Около 8% локомотивов имеют несовпадающий по местоположению или времени факт о связанной бригаде. Примеры - в отчете `input_report`.
# 
# ### Станция Зыково не принадлежит ни одному участку обкатки бригад
# 
# [JIRA VP-5633](http://jira.programpark.ru/browse/VP-5633)
# 
# Это приводит к тому, что реальные бригады не могут быть назначены на участок Красноярск-Восточный - Иланская.
# 
# **Сергей это исправил локально, надо проверить, попало ли это обновление на ТК2.**
# 
# ### Вывозные локомотивы
# 
# [JIRA VP-6955](http://jira.programpark.ru/browse/VP-6955)
# 
# Определять в онтологии, что локомотив является вывозным. Передавать соответствующий признак в планировщик. Планировать вывозные локомотивы на небольшие маршруты (см. требования к планировщику и список работ [здесь](#Вывозные-локомотивы)).
