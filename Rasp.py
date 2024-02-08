from webbrowser import open_new
import os
import time
import zipfile
import pandas as pd
import shutil

dirs = '1 курс; 2 курс; 3 курс; 4 курс'.split('; ')
downloads = 'C:/Users/user/Downloads'
if '342#23d' in os.listdir(downloads):
        shutil.rmtree(f'{downloads}/342#23d')
for i in dirs:
    if f'{i}.zip' in os.listdir(downloads):
        os.remove(f'{downloads}/{i}.zip')

#открываем страницу с расписание и скачиваем его в виде zip-архива
open_new('https://www.dropbox.com/sh/4el13d3ot8ud6hr/AAC_QIpK5LvvV_OSsfyq8f7Ja/1%20курс?dl=1')
open_new('https://www.dropbox.com/sh/4el13d3ot8ud6hr/AABKLxApIPgNx4b-QH1vFQrta/2%20курс?dl=1')
open_new('https://www.dropbox.com/sh/4el13d3ot8ud6hr/AABy0G71vhCIGtSlOspq-hgla/3%20курс?dl=1')
open_new('https://www.dropbox.com/sh/4el13d3ot8ud6hr/AABryVOLWnjVTG70x68Pvx1Fa/4%20курс?dl=1')

for i in dirs:
    while f'{i}.zip' not in os.listdir(downloads):
        time.sleep(1)

# Создание рабочего каталога
workdir = 'C:/Users/user/Downloads/342#23d'

# Создание рабочего каталога с данными из архива с расписанием
downloads = 'C:/Users/user/Downloads'
workdir = 'C:/Users/user/Downloads/342#23d'

if not os.path.exists(workdir):
    os.makedirs(workdir)

dirs = '1 курс.zip; 2 курс.zip; 3 курс.zip; 4 курс.zip'.split('; ')
for file in dirs:
    with zipfile.ZipFile(f'{downloads}/{file}', 'r') as zip_ref:
        zip_ref.extractall(workdir)

# Создание попки с отсортированными и профильтрованными расписаниями
dirs = '1 курс; 2 курс; 3 курс; 4 курс'
dirs = dirs.split('; ')
storage = 'C:/Users/user/Downloads/342#23d'

# Оставляем только актуальное
last_date = sorted([i.split(' ')[-1][:-5] for i in os.listdir(storage)])[-1]

for file in os.listdir(storage):
    if last_date not in file:
        os.remove(f'{storage}/{file}')

usl = '226,229,233,230,232,Круглова,Мелешко,Сауткин Ф.В.,Рогинский,Буга,Митянин,Лазаренко,Федоринчик,Яковчик,Буга,Коротеева,Основы информационной био'.split(
    ',')


class Lesson:
    collection = []

    def __init__(self, day, time, group, name, course):
        self.day = str(day).split(' ')[0]
        self.time = time
        self.group = group
        self.program = course + ' курс'
        self.name = str(name)

        if 'логия и экол' in str(self.name) and 'Спец' not in str(self.name):
            self.program = 'Основы зоологии'
        elif 'оологии' in str(self.name) and 'Спец' not in str(self.name):
            self.program = 'Основы зоологии'
        elif 'оология' in str(self.name) and 'Спец' not in str(self.name):
            self.program = 'Зоология'
        else:
            1 + 1

        if ('Спецкурс' in str(self.name) or 'Спецпрактикум' in str(self.name)) and 'зоо' in str(self.name):
            st = self.name.split(',')
            for i in range(len(st)):
                if 'Спецкурс' in st[i]:
                    last = 'Спецкурс'
                elif 'Спецпрактикум' in st[i]:
                    last = 'Спецпрактикум'
                else:
                    st[i] = last + st[i]
            self.name = [i for i in st if 'зоо' in i][0]

    def clear():
        newcol = []
        for i in Lesson.collection:
            if str(i.name) != 'nan':
                for us in usl:
                    if us in i.name:
                        newcol.append(i)
                        break
        Lesson.collection = newcol


Lesson.collection = []

# читаем датафрейм и меняем его
for file in os.listdir(storage):
    df = pd.read_excel(f'{storage}/{file}', header=0)
    df.columns = df.iloc[0]
    for x in range(len(list(df.iteritems())[0])):
        if str(list(df.iteritems())[x]) != 'nan':
            df = df[x:]
            break

    data = [i for i in df.iteritems()]

    date = [i for i in data[0][1]]

    for i in range(len(date)):
        if str(date[i]) != 'nan':
            day = str(date[i])
        else:
            date[i] = day

    time = [i for i in data[1][1]]
    data = data[2:]

    for col in data:
        group = col[0]
        for obq_id in range(len(col[1])):
            col_data = [i for i in col[1]]
            lesson = col_data[obq_id]
            day = date[obq_id]
            t = time[obq_id]
            course = str(file)[0]
            a = Lesson(day, t, group, lesson, course)
            Lesson.collection.append(a)

Lesson.clear()

if 'replace.xlsx' in os.listdir():
    df = pd.read_excel('replace.xlsx')
    a = [list(df.loc[i].values) for i in range(len(df))]
    cor = [[a[i][0], str(a[i][1]).replace('nan', '')] for i in range(len(a))]
    cor.extend([['  ', ' '], ['  ', ' ']])

    for i in range(len(Lesson.collection)):
        for rep in cor:
            Lesson.collection[i].name = Lesson.collection[i].name.replace(rep[0], rep[1])

# Создаем таблицу с расписанием
df = pd.DataFrame(columns='День,Время,Основы зоологии,Зоология,1 курс,2 курс,3 курс,4 курс'.split(','))
df['День'] = [i.split(' ')[0] for i in date]
df['Время'] = [i for i in time]
# for para in Lesson.collection:
for para in Lesson.collection:

    if pd.isna(df.loc[(df['День'] == para.day) & (df['Время'] == para.time), para.program]).values[0] == True:
        df.loc[(df['День'] == para.day) & (df['Время'] == para.time), para.program] = para.name
    elif para.name in df.loc[(df['День'] == para.day) & (df['Время'] == para.time), para.program].to_string():
        pass
    else:
        df.loc[(df['День'] == para.day) & (df['Время'] == para.time), para.program] = df.loc[(df['День'] == para.day) & (df['Время'] == para.time), para.program] + ';' + para.name

df = df[df['День'] != 'Дата']
df = df[df['День'] != '4']
df

df.to_excel(f"C:/Users/user/Desktop/output.xlsx")