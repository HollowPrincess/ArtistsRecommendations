# ArtistsRecommendations

***Задача***: научиться искать похожих музыкальных исполнителей.

*Примечание: для воспроизводимости кода в корень проекта необходимо положить архив с данными.*

**1. Подготовка данных.**

Первичная обработка данных написана в скрипте **dataset_preparation.ipynb**

В рамках этого скрипта происходит чтение файлов sessions, tracks, persons, love и playlists. Из этих файлов извлекаются необходимые для работы данные и названия песен заменяются именами исполнителей для дальнейшей работы.

В истории прослушивания присутствуют не все исполнители, пропущено порядка 84 тысяч идентификаторов, поэтому идентификаторы для исполнителей были изменены на последовательные, чтобы избежать трудностей при работе с матрицами смежности.

Далее данные причесываются в скрипте **prettify_data.ipynb**.
В скрипте **prepare_tags.ipynb** происходит обработка тегов.

**2. Протокол оценки качества.**

Можно переформулировать задачу: необходимо рекомендовать пользователям исполнителей, основываясь только на истории прослушиваний (так как задание заключается в поиске похожих исполнителей, то в данном случае не используются данные о самих пользователях, которые в общем случае важны при построении рекомендаций).

На вход мы получаем исполнителя **p1**, на выходе у нас должен быть список исполнителей **P = {p2, p3, …}** такой что **p1** не принадлежит **P**.

Если переформулировать задачу, то можно сказать, что пользователь **u1** прослушал исполнителя **p1**, надо рекомендовать **P = {p2, p3, …}: p1 &notin; P**.

Тогда для оценки можно использовать историю прослушивания пользователей:

Разбиваем исполнителей на тренировочное и тестовое множество, пусть 90% - тренировочные, 10% - тестовые.

Чтобы оценить качество алгоритма на тестовом множестве считаем метрику качества mean average precision at k, где в качестве k рассмотрим 1, 5, 10 и 20. Эта метрика хорошо отражает точность предсказаний и она была выбрана, потому что в рамках задачи необходимо знать, сколько предсказаний валидны и важно, чтобы валидные рекомендации оказались в списке как можно выше.
В качестве основной метрики будет рассматриваться метрика при k=20, так как хочется сделать как можно больше валидных рекомендаций, остальные метрики рассматриваются из общего интереса.
Для каждого исполнителя рассматриваются все пользователи, которые слушали композиции этого исполнителя, на основании этого формируем список исполнителей, которых слушают эти пользователи. 
Строятся k рекомендаций на основании предсказания системы. 
Затем определяется, какую долю исполнителей в рекомендациях угадали на основании списка.

Baseline для оценки качества: в качестве рекомендации берем просто k самых прослушиваемых исполнителей из всех и рекомендуем их. Сравнение с этим методом поможет понять, имеет ли смысл построенная система рекомендаций.

Реализация в скрипте **baseline.ipynb**. 

Функция выбора top-k исполнителей из множества реализована в скрипте **recommendations.py**.

**3. Векторные представления.**

Итак, следующие идеи для построения векторных представлений исполнителей:
* Матрица смежности исполнителей, где в ячейках стоит количество плейлистов, общих для двух исполнителей.
* Матрица смежности исполнителей, где в ячейках стоит количество пользователей, лайкнувших двух исполнителей.
* Матрица смежности исполнителей, где в ячейках стоит количество пользователей, слушавших двух исполнителей.
По факту эти представления описывают графы связей, где исполнители являются вершинами, а числа в ячейках - вес ребер.
При этом в плейлистах находится слишком мало исполнителей (63639 из истории прослушивания), относительно общего числа исполнителей в истории прослушиваний (511 768), поэтому рассматривать такое представление без дополнительных признаков не стоит, так как слишком много исполнителей будут иметь нулевой вектор.
То же самое и с лайками: всего 107 563 исполнителей из истории прослушиваний имеют лайки.

Также было интересно рассмотреть следующие признаки:
* Комбинации трех описанных выше.
* Показатели популярности исполнителя: количество лайков, время прослушивания, количество прослушиваний.
* Наличие/отсутствие каждого тега у исполнителя (one hot encoding по тегам).

**4. Модели.**

В качестве моделей были рассмотрены:

Первые три предлагаемых векторных представления по факту являются описанием графов связей: вершины — это исполнители, а значения в матрицах — вес ребер. Поэтому есть смысл попробовать **affinity propagation** (скрипт **affinity.ipynb**), так как этот алгоритм ориентирован на работу с графами. Но так как в плейлистах находится слишком маленький процент исполнителей, и исполнителей с лайками тоже гораздо меньше, чем исполнителей в истории прослушиваний, то кластеризация affinity propagation проводится только для случая матрицы смежности между исполнителями, где в ячейках содержится количество пользователей, слушающих двух исполнителей (скрипт **prepare_matrices_for_clustering.ipynb**).

Для оптимизации гиперпараметров алгоритмов была использована реализация алгоритма Tree of Parzen Estimators (TPE) в фреймворке Hyperopt. В качестве функции для оптимизации алгоритмов кластеризации был выбран индекс Дэвиса-Болдина. 
Индекс Дэвиса-Болдина был выбран на основании статьи про [метрики качества кластеризации](https://neerc.ifmo.ru/wiki/index.php?title=%D0%9E%D1%86%D0%B5%D0%BD%D0%BA%D0%B0_%D0%BA%D0%B0%D1%87%D0%B5%D1%81%D1%82%D0%B2%D0%B0_%D0%B2_%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B5_%D0%BA%D0%BB%D0%B0%D1%81%D1%82%D0%B5%D1%80%D0%B8%D0%B7%D0%B0%D1%86%D0%B8%D0%B8#.D0.92.D0.BD.D1.83.D1.82.D1.80.D0.B5.D0.BD.D0.BD.D0.B8.D0.B5_.D0.BC.D0.B5.D1.80.D1.8B_.D0.BE.D1.86.D0.B5.D0.BD.D0.BA.D0.B8_.D0.BA.D0.B0.D1.87.D0.B5.D1.81.D1.82.D0.B2.D0.B0).

Также из-за огромного объема истории прослушиваний, было принято решение взять срез по пользователям: были выбраны данные только 15 тысяч самых активных пользователей.

Из-за ограничений по памяти (12,72 Гб в Google Colaboratory) матрица полностью не влезет в память, поэтому в качестве столбцов рассматривается только тысяча самых популярных исполнителей.

Так как этот метод больше ориентирован на работу с графами, то было принято решение не использовать методы понижения размерности.
Итоговые оценки качества оставляют желать лушчего.

Так как объем экземпляров очень велик, было решено использовать на этих данных **SVD** и **MiniBatch Kmeans** (**kmeans_with_svd.ipynb**).
Программа не успела отработать.

Также была рассмотрена **нейронная сеть**, которую описала компания Google для рекомендаций видео на Youtube в 2018 года в статье [«Latent Cross: Making Use of Context in Recurrent Recommender Systems»](https://static.googleusercontent.com/media/research.google.com/ru//pubs/archive/46488.pdf). Так как задача заключается в поиске похожих исполнителей, то при построении системы не учитывались признаки контекста. (Скрипт **Network_matrix_with_tags_likes_etc.ipynb**)

По факту это случай многомерной классификации: были выбраны топ-1000 исполнителей для которых сеть предсказывает распределение вероятностей рекомендации исполнителя.

В качестве признаков были использованы признаки полученные с помощью SVD из матриц смежности для лайков и плейлистов, а также one-hot encoding тегов (скрипт **form_matrices_with_tags_likes_etc.ipynb**). В качестве распределений вероятностей для обучения была рассмотрена нормированная матрица смежности исполнителей, усеченная до 1000 столбцов по топ-1000 исполнителям, которая была ранее рассмотрена в кластеризации.
Программа не успела отработать.
Также было бы неплохо опробовать KMeans на этих данных, но времени не хватило.

**5. Результаты.**
В качестве **baseline** были выбраны топ-k популярных исполнителей из тренировочного множества, где k соответствует параметру метрики и были получены следующие оценки качества на тестовом множестве:

* **МАР@1**: 0.447
* **МАР@5**: 0.363
* **МАР@10**: 0.320
* **МАР@20**: 0.287

Значения метрик качества для **affinity** слишком плохи, чтобы сравнивать.

Для **Kmeans** и **нейронной сети** результаты не были получены (для Kmeans программа не успела завершить работу до дедлайна, в нейросети есть неочевидная ошибка).

