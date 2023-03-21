
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

os.chdir(r'{}'.format(input("Enter path ")))


def save(name='', fmt='png'): #Сохраняет изображения
    pwd = os.getcwd() #получает путь папки, в которую можно сохранить

    iPath1 = '.\pictures' 
    iPath2 = '.\pictures\{}'.format(fmt)
   
    if not os.path.exists(iPath1): # если пути к папке с изображениями нет, создает новую
        os.mkdir(iPath1)
    
    if not os.path.exists(iPath2): # если пути к папке с изображениями в формате png нет, создает новую
            os.mkdir(iPath2)
    os.chdir(iPath2)
    plt.savefig('{}.{}'.format(name, fmt), fmt='png')
    os.chdir(pwd)
    
def patch_calculations(currents_list, size, reverse_potential): #по полученным данным проводит расчеты
    max_current = currents_list[-1]
    min_current = currents_list[0]
    d_mkm = float(2.4 * size)
    s_mkm = float(d_mkm ** 2 * 3.1416)
    s_m2 = float(s_mkm / 1000000000000) # ток при данном напряжении делится на эту величину
    i_s = float(max_current / s_m2)
    G_ps = float(max_current / 95 / 1000)
    GS_m2 = float(G_ps /  s_m2 / 1000000000000)
    G_ps_mkm2 =  float(G_ps / s_mkm)
    
    modified_currents = []

    data_frame_patch.loc[len(data_frame_patch.index )] = [str(col), reverse_potential,      #сохраняет вычяисленные данные в формате dataframe
                                                          min_current, max_current, size,
                                                          d_mkm, s_mkm, s_m2, i_s, G_ps,
                                                          GS_m2, G_ps_mkm2]
    return data_frame_patch 

def plot_drawer(Voltage, currents_list, col): # отрисовка графика V/I
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Voltage, currents_list, color ='black', linewidth=2) 
    plt.xticks(Voltage)
    ax.set_xlabel('Voltage')
    ax.set_ylabel('Current')
    ax.set_title(col)
    plt.grid()
    return ax

def reverce_potential_predictor(Voltage, currents_list): # расчет потенциала реверсии (V(ось х) при токе 0(y=0))
    x = Voltage
    y = currents_list
    x = sm.add_constant (x)
    model = sm. OLS (y, x). fit () # использование модели линейной регрессии для вычисления функции графика линейной функции
    coefs = model.params
    b = coefs[0]
    k = coefs[1]
    reverse_potential = round(-b / k)
    return  reverse_potential

def dataframe_to_excel(data_frame_patch, excel_Name): # сохраняет посчитанные значения в excel
    writer = pd.ExcelWriter('{}.xlsx'.format(excel_Name), engine='xlsxwriter') #название выходного файла
    data_frame_patch.to_excel(writer, sheet_name='Patch data', index=False) # название листа(тоже вывести в переменную надо)
    writer.close()
    data_frame_patch.drop(data_frame_patch.index, inplace=True)
    print("Excel file saved")

Voltage = [-180,-155,-130,-105,-80,-55,-30,-5,20,45,70,95 ]
ser = pd.Series(Voltage)

potentials = {}

data_frame_patch = pd.DataFrame({"Название записи": [], "Потенциал реверсии" : [], "Минимальный ток": [],
                                 "Максимальный ток": [], "Размер": [], "D mkm":[], "S mkm^2": [],
                                 "S m^2":[], "I/S (mA/m^2)":[], "G pS": [], "G S/m^2":[],
                                 "G pS/mkm^2":[] })



Multisample = pd.read_excel('Multisample.xlsx') # импорт файла Excel

columns_in_dataset = list(Multisample.columns)
data = list(Multisample)
rows = len(columns_in_dataset)
print("Количество записей: ", rows)

row = 0
for col in columns_in_dataset: #цикл перебирает данные по названиям записей(колонкам) из датафрейма
    
    currents_list = list(Multisample[col]) #получение колонки токов и размера клетки из датафрейма по названию колонки и запись их в отдельный список   
    size = currents_list[0]   
    currents_list.pop(0) #удаление значения размера клетки из списка токов
    
    plot_drawer(Voltage, currents_list, col)
    save(name = str(col), fmt='png')
    reverse_potential = reverce_potential_predictor(Voltage, currents_list)
    patch_calculations(currents_list, size, reverse_potential)
    

print(data_frame_patch)
excel_Name = "OutputData"
dataframe_to_excel(data_frame_patch, excel_Name)


#C:\Users\computer\Documents\Visual Studio 2019\patch data

