#!/usr/bin/env python
# coding: utf-8

# # Задание 1. A/B–тестирование
# 1.1 Условие
# 
# Одной из основных задач аналитика в нашей команде является корректное проведение экспериментов. Для этого мы применяем метод A/B–тестирования. В ходе тестирования одной гипотезы целевой группе была предложена новая механика оплаты услуг на сайте, у контрольной группы оставалась базовая механика. В качестве задания Вам необходимо проанализировать итоги эксперимента и сделать вывод, стоит ли запускать новую механику оплаты на всех пользователей.
# 
# 1.2 Входные данные
# 
# В качестве входных данных Вы имеете 4 csv-файла:
# 
# groups.csv - файл с информацией о принадлежности пользователя к контрольной или экспериментальной группе (А – контроль, B – целевая группа) 
# 
# groups_add.csv - дополнительный файл с пользователями, который вам прислали спустя 2 дня после передачи данных
# 
# active_studs.csv - файл с информацией о пользователях, которые зашли на платформу в дни проведения эксперимента. 
# 
# checks.csv - файл с информацией об оплатах пользователей в дни проведения эксперимента. 

# In[82]:


import pandas as pd
import numpy as np
import requests
from urllib.parse import urlencode
import json
import pingouin as pg
from scipy.stats import norm
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from tqdm.auto import tqdm
import seaborn as sns
import scipy


# In[41]:


base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = 'https://disk.yandex.ru/d/58Us0DWOzuWAjg'  # Ссылка на яндекс диск

# Получаем загрузочную ссылку
final_url = base_url + urlencode(dict(public_key=public_key))
response = requests.get(final_url)
download_url = response.json()['href']

# Получаем dataframe
groups_df = pd.read_csv(download_url, sep=';');


# In[42]:


base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = 'https://disk.yandex.ru/d/prbgU-rZpiXVYg'  # Ссылка на яндекс диск

# Получаем загрузочную ссылку
final_url = base_url + urlencode(dict(public_key=public_key))
response = requests.get(final_url)
download_url = response.json()['href']

# Получаем dataframe
active_studs_df = pd.read_csv(download_url, sep=';');


# In[43]:


base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
public_key = 'https://disk.yandex.ru/d/84hTmELphW2sqQ'  # Ссылка на яндекс диск

# Получаем загрузочную ссылку
final_url = base_url + urlencode(dict(public_key=public_key))
response = requests.get(final_url)
download_url = response.json()['href']

# Получаем dataframe
checks_df = pd.read_csv(download_url, sep=';');


# In[44]:


groups_df.head(3)


# In[45]:


groups_df.shape


# In[46]:


active_studs_df.head(3)


# In[47]:


active_studs_df.shape


# In[48]:


checks_df.head(3)


# In[49]:


checks_df.shape


# In[50]:


# Проверка на дубликаты
checks_df.duplicated(subset='student_id').sum()


# In[51]:


# Проверка на дубликаты
groups_df.duplicated().sum()


# In[52]:


# Смотрим распределение по группам
groups_df.grp.value_counts()


# In[53]:


# Пропущенных значений нет
groups_df.isnull().sum()


# In[54]:


# Посмотрим на выбросы с помощью "ящика с усами"
sns.boxplot(x=checks_df['rev'])


# In[55]:


# Посмотрим поближе
checks_df[checks_df.rev > checks_df.rev.quantile(0.995)]


# Есть выброс. Удалять его при данных условиях будет некорректно. Нужно узнать откуда он появился и либо удалить его, либо заменить на 99.5 перцентиль, либо оставить. В нашей задаче допустим, что это супер пользователь и такой выброс возможен.

# In[56]:


#Склеим все данные и добавим колонку isActive, чтобы видеть активных пользователей
full_df = groups_df.rename(columns={'id' : 'student_id'}).merge(active_studs_df, how='outer', on='student_id').merge(checks_df, how='outer', on='student_id')
full_df['isActive'] = 0
full_df['isActive'] = np.where([i in active_studs_df.student_id.values for i in full_df.student_id],1, 0)
full_df


# In[57]:


# Посмотри на различные зависимости
pairgridplot = sns.PairGrid(full_df)
pairgridplot.map(plt.scatter)


# In[58]:


# Пользователи, которые совершали покупку, но не были активны в дни проведения эксперимента. 
# Возможно, все дело в подписочной системе и плата списывается регулярно по подписке.
# Такие пользователи для анализа A/B теста не нужны, так как они не видели новую версию сайта
full_df.query('rev == rev and isActive == 0')


# In[59]:


# Есть пользователи, которые не принадлежат нашим группам A и B. Один из них делал покупку. Но ничего страшного, 
# видимо не все пользователи были распределены на группы. Возможно это новые пользователи.
# Такие пользователи для анализа A/B теста не нужны.
full_df.query('grp != grp')


# In[60]:


# Нас интересуют пользователи, которые заходили на платформу 
active_studs_groups_df = groups_df.query('id in @active_studs_df.student_id').rename(columns={'id' : 'student_id'})
# Добавим данные по оплате. Если не оплачивал, ставим 0.
ab_df = active_studs_groups_df.merge(checks_df, how='left', on='student_id').fillna(0)


# In[61]:


ab_df.head()


# In[62]:


# Посмотрим на выбросы. Все тот же выброс в группе B. И еще выбросы в группе A.
sns.boxplot(data=ab_df.query('rev > 0'), x=ab_df.query('rev > 0')['rev'], y='grp')


# In[63]:


# Посмотрим на выбросы, будем знать, что они есть. Но удалять их не будем.
ab_df.query('rev > 0 and grp == "A" and rev > 2500')


# In[64]:


# Добавим колонку с платежной информацией(если покупал 1, иначе 0)
ab_df['isPay'] = (ab_df.rev > 0)*1


# In[65]:


# Таблица с метриками CR, ARPU и ARPAU по группам
metric_df = ab_df.groupby('grp', as_index=0).agg({'isPay' : ['mean', 'sum'], 'rev' : ['mean', 'sum']}). rename(columns={'isPay': 'CR', 'rev' : 'ARPAU'})
metric_df.columns = ['grp', 'CR','sum_users','ARPAU', 'sum_rev']
metric_df['ARPPU'] = metric_df.sum_rev/metric_df.sum_users
metric_df = metric_df.drop(['sum_users', 'sum_rev'], axis=1)


# In[66]:


metric_df


# Среднее(на активного пользователя и на платящего человека) в группе B больше. Но нас больше интересует конверсия в покупку. Потому что способ оплаты скорее влияет на конверсию, чем на сумму покупки. 

# Конверсия в покупку в группе B меньше. Проверим статистически значимы ли результаты.
# Проверять будем с помощью критерия хи-квадрат, так как данные являются номинативными.
# 
# H0 : конверсия в группах не различается, а наблюдаемые различия случайны;
# 
# H1 : конверсия в группах различается
# 
# Для бизнеса увеличение конверсии показывает, то насколько больше стало платящих пользователей(то насколько привлекательнее стал продукт). (Очень полезно)

# In[67]:


ab_df.head(3)


# In[68]:


# Посмотрим стат значимы ли различия в конверсиях
expected, observed, stats = pg.chi2_independence(ab_df, x = 'grp', y = 'isPay', correction=False)


# In[69]:


# То какое распределение ожидается, если изменений нет
expected


# In[70]:


# То какое распределение у нас в группах
observed


# In[71]:


stats.round(3)


# Хи-квадрат Пирсона показал p-value > 0.05, что говорит нам о том, что статистически значимых различий между группами нет. А значит, что мы не можем сказать, что конверсия изменилась

# Проверим различаются ли стат значимо дисперсии в группах

# In[72]:


# var в группах по колонке rev
ab_df.groupby('grp', as_index=0).agg({'rev' : 'var'})


# Для проверки равенства дисперсий в группах по колонке rev воспользуемся критерием Фишера
# 
# H0: D_a/D_b = 1
# 
# H1: D_a/D_b != 1
# 
# Равенство дисперсий нам важно для применения критерия Стьюдента

# In[147]:


# Тест Фишера
def test_F(a, b):
    F = a.var()/b.var()
    p_value = scipy.stats.f.cdf(F, len(a)-1, len(b)-1)
    return np.min([p_value, 1-p_value], axis=0)
    


# In[148]:


# p_value
test_F(ab_df.query('grp == "A"').rev, ab_df.query('grp == "B"').rev)


# У нас есть все основания отвергнуть H0 о равенстве дисперсий.
# 
# Проверим дисперсий для значений rev > 0.

# In[149]:


# var в группах по колонке rev без 0
ab_df.query('rev > 0').groupby('grp', as_index=0).agg({'rev' : 'var'})


# Для проверки равенства дисперсий в группах по колонке rev(где rev > 0) воспользуемся критерием Фишера
# 
# H0: D_a/D_b = 1
# 
# H1: D_a/D_b != 1
# 
# Равенство дисперсий нам важно для применения критерия Стьюдента

# In[150]:


# p_value
test_F(ab_df.query('grp == "A" and rev > 0').rev, ab_df.query('grp == "B" and rev > 0').rev)


# У нас есть основания отвергнуть H0 о равенстве дисперсий.

# In[151]:


A_df = ab_df.query('grp == "A"')
B_df = ab_df.query('grp == "B"')


# In[152]:


A_df.rev.hist();


# In[153]:


B_df.rev.hist();


# In[154]:


# Распределения ассиметричны.
# Попробуем нормальзовать с помощью логарифма
# Прибавим 0.01, так как логарифм от 0 не берется
np.log(A_df.rev + 0.01).hist();
# Логарифм не помогает


# Проверим различаются ли стат значимо ARPAU в группах.
# 
# Метрика: ARPAU
# 
# H0: ARPAU в группах равны, а наблюдаемые различия случайны;
# 
# H1: ARPAU в группах различаются
# 
# Для бизнеса изменение метрики ARPAU показывает, на сколько больше мы стали зарабытавать с одного пользователя (очень полезно)

# Для проверки гипотез воспользуемся бутстрапом. Метод затратен, но для наших данных вычислений много времени не займет

# In[89]:


# Функция Бутстрапа

def get_bootstrap(
    data_column_1, # числовые значения первой выборки
    data_column_2, # числовые значения второй выборки
    boot_it = 10000, # количество бутстрап-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
):
    boot_data = []
    for i in tqdm(range(boot_it)): # извлекаем подвыборки
        samples_1 = data_column_1.sample(
            len(data_column_1), 
            replace = True # параметр возвращения
        ).values
        
        samples_2 = data_column_2.sample(
            len(data_column_1), 
            replace = True
        ).values
        
        boot_data.append(statistic(samples_1)-statistic(samples_2)) # mean() - применяем статистику
        
    pd_boot_data = pd.DataFrame(boot_data)
        
    left_quant = (1 - bootstrap_conf_level)/2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])
        
    p_1 = norm.cdf(
        x = 0, 
        loc = np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_2 = norm.cdf(
        x = 0, 
        loc = -np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2
        
    # Визуализация
    _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
    for bar in bars:
        if bar.get_x() <= quants.iloc[0][0] or bar.get_x() >= quants.iloc[1][0]:
            bar.set_facecolor('red')
        else: 
            bar.set_facecolor('grey')
            bar.set_edgecolor('black')
    
    plt.style.use('ggplot')
    plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
    plt.xlabel('boot_data')
    plt.ylabel('frequency')
    plt.title("Histogram of boot_data")
    plt.show()
       
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value}


# In[90]:


result = get_bootstrap(data_column_1=A_df.rev,
    data_column_2=B_df.rev,
    boot_it = 100000,
    statistic = np.mean,
    bootstrap_conf_level = 0.95)


# In[91]:


result['p_value']


# P-value > 0.05
# Значит у нас нет оснований отвергнуть H0.
# 

# In[92]:


# Оценим метрику ARPPU 
ab_df.query('rev > 0').groupby('grp', as_index=0).agg({'rev' : 'mean'})


# In[54]:


# Распределение revenue для группы A
ab_df.query('rev > 0 and grp == "A"')['rev'].hist();


# In[55]:


# Распределение revenue для группы B
ab_df.query('rev > 0 and grp == "B"')['rev'].hist();


# Метрика: ARPPU
# 
# H0: ARPPU в группах равны, а наблюдаемые различия случайны;
# 
# H1: ARPPU в группах не равны
# 
# Изменения в метрике дает понять на сколько платящие пользователи стали платить больше(или меньше). Т.е отслеживая эту метрику, мы интересуемся только платящими пользователями. Для бизнеса очень интересно, так как платящие пользователи - это те, кто приносит нам деньги.
# P.S Немного очевидный абзац.
# 
# Для проверки гипотез воспользуемся бутстрапом, потому что он требовательный к условиям и на наших данных много времени не займет. С нашими данными займет немного времени.

# In[94]:


result_for_ARPPU = get_bootstrap(
    ab_df.query('rev > 0 and grp == "A"').rev, # числовые значения первой выборки
    ab_df.query('rev > 0 and grp == "B"').rev, # числовые значения второй выборки
    boot_it = 100000, # количество бутстрап-подвыборок
    statistic = np.mean, # интересующая нас статистика
    bootstrap_conf_level = 0.95 # уровень значимости
);


# In[40]:


result_for_ARPPU['p_value']


# p_value < 0.05
# Если наш alpha = 0.05, то мы отклоняем нулевую гипотезу и принимаем альтернативную, т.е считаем, что ARPPU в группах различаются

# Видим, что расспределения отличаются. У группы B стало сильно больше покупок за 2000+- у.е. Стоит посмотреть на данные в разных разрезах и попытаться найти группу пользователей, которым "понравилось" обновление или наоборот. И возможно выкатить новую версию на определенную группу пользователей.

# Очень странный пик около 2000 у.е. Нужно узнать почему так. Может определенная группа пользователей(которые покупали продукт за 2000 у.е. стали покупать больше и на них повлияла новая версия продукта. Например, эта группа пользователей пожилые люди и с новой механикой оплаты им легче оплатить продукт. Или наша система сплитования сломана.

# # Итог
# 
# 
# Целевая метрика: конверсия.
# Мы можем с определенной увереностью сказать, что метрика не изменилась.
# 
# То же самое и с ARPAU. 
# 
# Метрика ARPPU. С определенной увереностью можем утверждать, что метрика выросла.
# 
# Вывод: несмотря на увеличение метрики ARPPU, новую механику пока не стоит запускать на всех пользователей. Стоит посмотреть на данные в разных разрезах, может определенная группа пользователей стала платить больше или меньше. И уже после выкатить изменение на определенную группу пользователей. Либо собрать еще данные, возможно изменения в конверсии и в ARPAU меньше, чем мы можем обнаружить. При этом тут, конечно, стоит обратить внимание на проблему подглядывания. Если мы правильно делаем A/B тест, то перед началом мы должны были провести A/A тест, далее выбрать alpha, можность теста и MDE и на этих данных определить размер выборки. Если все условия выполнены и настал тот день 'X', когда надо принимать решение и посмотреть на разные разрезы пользователей невозможно, то выкатить обновление. Так как статистика говорит, что платящие пользователи стали платить больше, а другие метрики не прокрасились. 

# # Задание 3. Python
# 3.1 Задача
# 
# Реализуйте функцию, которая будет автоматически подгружать информацию из дополнительного файла groups_add.csv (заголовки могут отличаться) и на основании дополнительных параметров пересчитывать метрики.
# Реализуйте функцию, которая будет строить графики по получаемым метрикам.

# Функции реализовал в отдельном файле. Далее импортировал этот файл

# In[43]:


# Все результаты будем хранить в списке
all_result = [metric_df]


# In[44]:


'''
На вход функции подается: 
group_df - датафрейм со старыми данными,
active_studs_df - датафрейм с активными в дни проведения эксперимента пользователями,
checks_df - датафрейм с оплатами
link_to_file - ссылка до новых данных,
sep_in_file - разделитель в файле

Функция возвращает:
1) Обновленный датафрейм group_df,
2) Пересчитанную таблицу с метриками
'''
def add_data_to_groups(groups_df, active_studs_df, checks_df,  link_to_file='default', sep_in_file=','):
    if link_to_file == 'default':
        base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
        public_key = 'https://disk.yandex.ru/d/3aARY-P9pfaksg'  # Ссылка на яндекс диск

        # Получаем загрузочную ссылку
        final_url = base_url + urlencode(dict(public_key=public_key))
        response = requests.get(final_url)
        download_url = response.json()['href']
    else:
        download_url = link_to_file
    # Получаем dataframe
    groups_add_df = pd.read_csv(download_url, sep=sep_in_file)
    
    #Проверка на дупликаты
    if groups_add_df.duplicated().sum() != 0:
        print('В новых данных есть дупликаты')
        
    # Проверка на пропущенные значения
    if groups_add_df.isnull().sum().sum():
        print('В новых данных есть пропущенные значения')
    
    # Добавим новые данные в таблицу groups_df
    groups_df = pd.DataFrame(np.concatenate((groups_df.values, groups_add_df.values), axis=0))
    groups_df.columns = ['id', 'grp']
    
    if groups_df.id.nunique() != groups_df.id.count():
        print('New data have old users')
    
    # Нас интересуют пользователи, которые заходили на платформу 
    active_studs_groups_df = groups_df.query('id in @active_studs_df.student_id').rename(columns={'id' : 'student_id'})
    
    # Добавим данные по оплате. Если не оплачивал, ставим 0.
    ab_df = active_studs_groups_df.merge(checks_df, how='left', on='student_id').fillna(0)
    
    # Добавим колонку с платежной информацией(если покупал True, иначе False)
    ab_df['isPay'] = (ab_df.rev > 0)*1
    
    # Считаем метрики
    metric_df = ab_df.groupby('grp', as_index=0).agg({'isPay' : ['mean', 'sum'], 'rev' : ['mean', 'sum']}).     rename(columns={'isPay': 'CR', 'rev' : 'ARPAU'})
    metric_df.columns = ['grp', 'CR','sum_users','ARPAU', 'sum_rev']
    metric_df['ARPPU'] = metric_df.sum_rev/metric_df.sum_users
    metric_df = metric_df.drop(['sum_users', 'sum_rev'], axis=1)
    
    #Выводим
    return groups_df, metric_df


# In[45]:


new_groups_df, new_metric_df =     add_data_to_groups(groups_df, active_studs_df, checks_df,  link_to_file='default', sep_in_file=',')


# In[46]:


new_metric_df


# In[47]:


all_result.append(new_metric_df)


# In[48]:


'''
На вход функции подается: 
all_result - список из датафреймов метрик,
name_metric - название метрики

Функция возвращает barplot по определенной метрике, где по х(условно, сколько дней прошло с начала эксперимента), 
по y - значение метрики.
'''
def print_metric(all_result, name_metric='CR'):
    A_data = [i[name_metric][0] for i in all_result]
    B_data = [i[name_metric][1] for i in all_result]
    # Строим график
    ax = pd.DataFrame({'A': A_data, 'B' : B_data}).plot.bar(figsize=[9, 7], title=name_metric, rot=0)
    return ax


# In[49]:


#CR
print_metric(all_result, name_metric='CR');


# In[50]:


# График ARPU 
print_metric(all_result, name_metric='ARPAU');


# In[51]:


# График ARPAU  
print_metric(all_result, name_metric='ARPPU');


# In[ ]:




