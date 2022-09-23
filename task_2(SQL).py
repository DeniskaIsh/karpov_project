#!/usr/bin/env python
# coding: utf-8

# # Задание 2. SQL

# In[1]:


import pandahouse as ph


# In[2]:


# Словарь с нужными параметрами для подключения к Clickhouse
# для подключения к default:
connection_default = {'host': 'http://clickhouse.beslan.pro:8080',
'database':'default',
'user':'student',
'password':'dpo_python_2020'
}


# 2.1 Очень усердные ученики.
# 
# 2.1.1 Условие
# 
# Образовательные курсы состоят из различных уроков, каждый из которых состоит из нескольких маленьких заданий. Каждое такое маленькое задание называется "горошиной".
# 
# Назовём очень усердным учеником того пользователя, который хотя бы раз за текущий месяц правильно решил 20 горошин.

# 2.1.2 Задача
# 
# Дана таблица default.peas.
# 
# Необходимо написать оптимальный запрос, который даст информацию о количестве очень усердных студентов за март 2020 года.
# 
# NB! Под усердным студентом мы понимаем студента, который правильно решил 20 задач за текущий месяц.

# In[3]:


# Пишем запрос
# В таблице нет данных за март 2020 года, поэтому запрос возвращает 0!
q = '''
SELECT COUNT(a.st_id) as count_diligent_st
FROM
(
    SELECT st_id
    FROM default.peas
    WHERE
        timest >= '2020-03-01' AND
        timest <= '2020-03-31'
    GROUP BY st_id
    HAVING SUM(correct) >= 20
) a
'''
# отправляем запрос и записываем результат в пандасовский датафрейм
q_first = ph.read_clickhouse(query=q, connection=connection_default)
q_first


# 2.2 Оптимизация воронки
# 
# 2.2.1 Условие
# 
# Образовательная платформа предлагает пройти студентам курсы по модели trial: студент может решить бесплатно лишь 30 горошин в день. Для неограниченного количества заданий в определенной дисциплине студенту необходимо приобрести полный доступ. Команда провела эксперимент, где был протестирован новый экран оплаты.

# 2.2.2 Задача
# 
# Дана таблицы: default.peas, default.studs и default.final_project_check:
# 
# Необходимо в одном запросе выгрузить следующую информацию о группах пользователей:
# 
# ARPU, 
# ARPAU,
# CR в покупку, 
# СR активного пользователя в покупку,
# CR пользователя из активности по математике (subject = ’math’) в покупку курса по математике
# 
# ARPU считается относительно всех пользователей, попавших в группы.
# 
# Активным считается пользователь, за все время решивший больше 10 задач правильно в любых дисциплинах.
# 
# Активным по математике считается пользователь, за все время решивший 2 или больше задач правильно по математике.

# In[36]:


q = '''
SELECT 
    test_grp,
    SUM(df.money)/uniqExact(ds.st_id) as ARPU,
    SUMIf(df.money, active.isActive > 10)/uniqIf(ds.st_id, active.isActive > 10) as ARPAU,
    round(100*uniqIf(ds.st_id, df.money > 0)/uniqExact(ds.st_id), 2) as CR,
    round(100*uniqIf(ds.st_id, active.isActive > 10 and df.money > 0)/uniqIf(ds.st_id, active.isActive > 10) ,2) as CR_active,
    round(100*uniqIf(ds.st_id, active.isActiveMath > 1 and df.subject = 'Math' and df.money > 0) /
        uniqIf(ds.st_id, active.isActiveMath > 1), 2) as CR_active_math
FROM default.studs as ds
LEFT JOIN (
            SELECT 
                st_id, SUM(correct) as isActive, SUMIf(correct, subject = 'Math') as isActiveMath
            FROM default.peas
            GROUP BY st_id
            ) as active  
ON ds.st_id = active.st_id
LEFT JOIN default.final_project_check as df
ON ds.st_id = df.st_id
GROUP BY test_grp
'''
# отправляем запрос и записываем результат в пандасовский датафрейм
q_second = ph.read_clickhouse(query=q, connection=connection_default)
q_second


# In[ ]:




