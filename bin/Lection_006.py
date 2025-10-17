#!/usr/bin/env python
# coding: utf-8


# для начала импортируем библиотеки, которые нам сегодня понадобятся
import numpy as np
import matplotlib.pyplot as plt



# Создадим функцию, которая позволяет выводить на экран графики типа y = f(x)
def plot_graph (x_data, y_data, x_label, y_label, curve_label = None):
    '''
    !!!!!!! подобные комментарии не обязательны, но желательны:
    
    Функция позволяет вывести график y = f(x)
    
    Параметры
    ------------
    x_data - вектор данных которые откладываются по оси X. Может быть либо переменной типа список, либо столбец в формате NumPy
    y_data - вектор данных которые откладываются по оси X. Может быть либо переменной типа список, либо столбец в формате NumPy
    x_label - название вектора данных по оси Х. Должен быть переменной типа string
    y_label - название вектора данных по оси Y. Должен быть переменной типа string
    curve_label = название кривой. По умолчание None. Если не None, то должен быть переменной типа string

    Возвращает
    -------------
    None
    '''
    # Визуализируем результаты
    # создаем "базу" для графика
    plt.figure(figsize=(6, 4))
    plt.plot (x_data, y_data, '-', color = 'black', label = curve_label)
    # Добавляем названия осей
    plt.xlabel (x_label, fontsize = 14)
    plt.ylabel (y_label, fontsize = 14)
    # добавляем для красоты сетку
    plt.grid()
    # если curve_label не None, то выводим название графика
    if curve_label:
        plt.legend()

# создадим функцию, которая ищет максимальное значение по y
def get_max_point (x_data, y_data):
    '''
    Функция ищет максимальное значение переменной y и соответствующее ей значение x, f а так же номер индекса этих двух чисел 

    Параметры
    ------------
    x_data - вектор данных которые откладываются по оси X. Может быть либо переменной типа список, либо столбец в формате NumPy
    y_data - вектор данных которые откладываются по оси X. Может быть либо переменной типа список, либо столбец в формате NumPy

    Возвращает
    -------------    
    y_max, x - оба типа float 
    ind_max -  индекс, соответсвенно переменная типа integer 
    '''
    # Если параметры пришли не в виде массива NumPy, то преобразоввывем их
    x_data, y_data = [np.array(item) if not isinstance(item, np.ndarray) else item for item in [x_data, y_data]]

    y_max = y_data.max()
    ind_max = y_data.argmax()
    x = x_data [ind_max]

    return y_max, x, ind_max 



# Создадим базовый класс от которого потом будем делать два или более дочерних
class Data_xy:
    """
    Класс для анализа и визуализации данных y = f(x)
    """
    
    def __init__(self, x_data, y_data, data_name = 'Default data'):
        """
        Инициализация класса с данными
        
        Параметры
        ------------
        x_data - вектор данных по оси X
        y_data - вектор данных по оси Y
        """
        # Преобразуем в numpy arrays если это необходимо
        self.x_data = np.array(x_data) if not isinstance(x_data, np.ndarray) else x_data
        self.y_data = np.array(y_data) if not isinstance(y_data, np.ndarray) else y_data

        self.data_name = data_name
        
        # Проверяем, что длины массивов совпадают
        if len(self.x_data) != len(self.y_data):
            raise ValueError("Длины x_data и y_data должны совпадать")
            
    # декоратор @staticmethod позволяет использовать метод не создавая эксземпляр класса
    @staticmethod
    def plot_graph (x_data, y_data, x_label, y_label, curve_label = None):
        '''    
        Функция позволяет вывести график y = f(x)
        
        Параметры
        ------------
        x_data - вектор данных которые откладываются по оси X. Может быть либо переменной типа список, либо столбец в формате NumPy
        y_data - вектор данных которые откладываются по оси X. Может быть либо переменной типа список, либо столбец в формате NumPy
        x_label - название вектора данных по оси Х. Должен быть переменной типа string
        y_label - название вектора данных по оси Y. Должен быть переменной типа string
        curve_label = название кривой. По умолчание None. Если не None, то должен быть переменной типа string
    
        Возвращает
        -------------
        None
        '''
        # Визуализируем результаты
        # создаем "базу" для графика
        plt.figure(figsize=(6, 4))
        plt.plot (x_data, y_data, '-', color = 'black', label = curve_label)
        # Добавляем названия осей
        plt.xlabel (x_label, fontsize = 14)
        plt.ylabel (y_label, fontsize = 14)
        # добавляем для красоты сетку
        plt.grid()
        # если curve_label не None, то выводим название графика
        if curve_label:
            plt.legend()
    
    def get_max_point(self, x_data, y_data, print_variables = False):
        '''
        Функция ищет максимальное значение переменной y и соответствующее ей значение x, 
        а так же номер индекса этих двух чисел 

        Возвращает
        -------------    
        y_max, x - оба типа float 
        ind_max - индекс, соответственно переменная типа integer 
        '''
                
        y_max = y_data.max()
        ind_max = y_data.argmax()
        x = x_data[ind_max]
        
        if print_variables:
            print (f'Максимальное значение по оси y: {y_max:.2f}')
            print (f'Соответсвующее ему значение по оси x: {x:.2f}')
            print (f'Соответсвующий индекс: {ind_max}')
        
        return y_max, x, ind_max



# создадим класс для расчета мех характеристик и хранения данных по инженерным деформациям и напряжениям
class Mech_properties (Data_xy):
    
    """класс для расчета инженерных напряжений/деформаций и механических свойств"""
    
    def __init__(self, x_data, y_data, L0, A0, data_name='Default data'):
        super().__init__(x_data, y_data, data_name)

        self.L0 = L0 # начальная длина образца по ISO 6892-1:2019
        self.A0 = A0 # площадь поперечного сечения образца


        ### Переменные для хранения напряжений/деформаций
        self.stress_eng, self.strain_eng = None, None

        ### Переменные для хранения механических характеристик
        self.Rm, self.Ag, self.E, self.yield_stress = None, None, None, None

        ### Индекс при Rm
        self.indx_Rm = None
        

    # Поясним, что есть x, а что есть y в данном случае
    # Декоратор @property позволяет "переименовывать" атрибуты
    @property
    def stroke (self): 
        return self.x_data
    @property
    def load (self): 
        return self.y_data



    def get_strain_eng (self):
        '''
        Расчет инженерной деформации по формуле из Лекции №2
        '''

        self.strain_eng = (self.stroke/self.L0)*100

        return self.strain_eng # т.к. мы внутри функции расчитываем переменную self.strain_eng то можно было бы обойтись без
                                # return. Но нам удобно будет, если все-таки функция будет напрямую возвращать переменную 
                                # чуть позже, поэтому оставим ее 

    def get_stress_eng (self):
        '''
        Расчет инженерного напряжения по формуле из Лекции №2
        '''

        self.stress_eng = self.load/self.A0

        return self.stress_eng


    def get_E (self, lower_elastic_limit = 0, upper_elastic_limit = 0.05):
        '''
        Функция пытается расчитать модуль Юнга. Если вдруг инженерные напряжения и деформации не были расчитаны, то функция
        выкидывает ошибку
        '''
        try:
            filtr =  filtr = (self.strain_eng >= lower_elastic_limit) & (self.strain_eng <= upper_elastic_limit)
            stress_elastic = self.stress_eng[filtr]
            strain_elastic = self.strain_eng[filtr]

            strain_elastic = strain_elastic/100 # переходим от процентов к долям от единицы
            
            numerator = np.sum(stress_elastic * strain_elastic)
            denominator = np.sum(strain_elastic ** 2)
            self.E = numerator / denominator
            
            #self.E, intercept  = np.polyfit(strain_elastic, stress_elastic, deg=0)
            
            return self.E
        
        except Exception:
            print (f'Ошибка {Exception}')
            print ('Судя по всему инженерные деформация и напряжения не расчитаны или расчитаны неправильно.')
            raise

    

    def get_Rm (self):
        '''
        Функция пытается расчитать предел прочности. Если вдруг инженерные напряжения и деформации не были расчитаны, то функция
        выкидывает ошибку
        '''

        try:
            self.Rm, _, self.indx_Rm = self.get_max_point (self.strain_eng, self.stress_eng)
            return self.Rm
        
        except Exception:
            print (f'Ошибка {Exception}')
            print ('Судя по всему инженерная деформация не расчитана или расчитана неправильно.')
            raise


    
    def get_Ag (self):
        '''
        Функция пытается расчитать предел прочности. Если вдруг инженерные напряжения и деформации не были расчитаны, то функция
        выкидывает ошибку
        '''
        # нам в любом случае понядобятся модуль Юнга и предел прочности, так что если еще их нет, то их нужно посчитать
        if not self.E:
            _ = self.get_E()
        
        if not self.Rm:
            _  = self.get_Rm()

        self.Ag = self.strain_eng[self.indx_Rm]/100 - self.Rm/self.E
        self.Ag = self.Ag*100

        return self.Ag

    def get_yield_stress (self, strain_limit = 0.2):
        '''
         Функция пытается расчитать предел текучести. Если вдруг инженерные напряжения и деформации не были расчитаны, то функция
        выкидывает ошибку
        '''
        if not self.E:
            _ = self.get_E()

        filtr = self.strain_eng/100 - self.stress_eng/self.E >= strain_limit/100
        self.yeild_stress = self.stress_eng[filtr][0]
        return self.yeild_stress
        

    def get_properties (self):
        '''
        Функция пытается расчитать все мех. характеристики данного класса.  Если вдруг инженерные напряжения и деформации не 
        были расчитаны, то функция выкидывает ошибку
        '''
        if not self.Ag:
            self.Ag = self.get_Ag()
        if not self.yield_stress:
            self.yield_stress = self.get_yield_stress()
        return {'Модуль Юнга': self.E, 
                'Предел текучести': self.yield_stress, 
                'Предел прочтности': self.Rm, 
                'Равномерное удлинение': self.Ag}



# создадим класс для расчета истинных напряжений и деформаций, а так же апроксимации кривой упрочнения
class Flow_stress (Data_xy):
    
    """класс для расчета истинных напряжений и деформаций, а так же апроксимации кривой упрочнения"""
    
    def __init__(self, x_data, y_data, yield_stress, data_name='Default data'):
        super().__init__(x_data, y_data, data_name)

        self.yield_stress = yield_stress

        self.strain_true = None
        self.stress_true = None

        self.flow_stress = None

    # Поясним, что есть x, а что есть y в данном случае
    @property
    def strain_eng (self): 
        return self.x_data
    @property
    def stress_eng (self): 
        return self.y_data



    def get_strain_true (self):
        '''
        Расчет истинной деформации по формуле из Лекции №2
        '''

        self.strain_true = np.log(1 + self.strain_eng/100)

        return self.strain_true

    
    def get_stress_true (self):
        '''
        Расчет истинного напряжения по формуле из Лекции №2
        '''

        self.stress_true = self.stress_eng*(1 + self.strain_eng/100)

        return self.stress_true


    def get_flow_stress (self):
        '''
        Создание NumPy array с кривой сопротивления пластической деформации
        столбец #0 - истинная пластическая деформация
        столбец #1 - истинное пластическое напряжение
        '''
        try:
            stress_max, _, indx_stress_max = self.get_max_point(self.strain_true, self.stress_true)
            filtr01 = self.strain_true <= self.strain_true[indx_stress_max]
            filtr02 = self.stress_true >= self.yield_stress
            filtr = filtr01&filtr02
            self.flow_stress = np.array([self.strain_true[filtr], self.stress_true[filtr]])
            return self.flow_stress
        except Exception:
            print (f'Ошибка {Exception}')
            print ('Судя по всему истинные деформация и напряжения не расчитаны или расчитаны неправильно.')
            raise
        
        

if __name__ == "__main__":
    load_stroke_data = np.loadtxt ('/home/alexander/Документы/GitHub/tension_test_calculations/dz/dz001/Fairuzova/3_20.txt',delimiter='\t')
    load_stroke_data = load_stroke_data [:, 1:] # удаляем первый столбец (это время, насколько я понял)
    # размеры образца
    d0 = 9.2 # диаметр
    L0 = 15 # длина образца
    # плащадь сечения образца
    A0 = (np.pi*d0**2)/2

    # Создаем объект класса Mech_properties
    mech_properties = Mech_properties (load_stroke_data [:, 0], load_stroke_data [:, 1], L0, A0)

    # посмотрим на график
    mech_properties.plot_graph(mech_properties.stroke, mech_properties.load, 'Ход деформирования, мм', 'Сила деформирвоания, Н')
    # посчитаем инженерные напряжения/деформации и посмотрим на график
    _ = mech_properties.get_strain_eng() # отдельная переменная нам не нужно, так так массив с инженерной деформацией 
                                            #будет храниться в переменной класса
    _ = mech_properties.get_stress_eng()
    
    # посмотрим на график
    mech_properties.plot_graph(mech_properties.strain_eng, mech_properties.stress_eng, 'Инженерная деформация, %', 'Инженерное напряжение, МПа')
    # Расчет модуля Юнга
    #E, intercept = mech_properties.get_E( lower_elastic_limit = 1., upper_elastic_limit = 3)

    


