#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


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
            
            
    @staticmethod 
    def plot_experiment_vs_model(data_exp, data_model, x_label, y_label):
        '''
        вывод графика сравнения модельных данных от экспериментальных
        '''        
        
        plt.figure(figsize=(6, 4))
        plt.plot (data_model[0], data_model[1], '-', color = 'black', label = 'Модель')
        plt.scatter (data_exp[0], data_exp[1], color = 'red', label = 'Экспериментальные данные')
        # Добавляем названия осей
        plt.xlabel (x_label, fontsize = 14)
        plt.ylabel (y_label, fontsize = 14)
        # добавляем для красоты сетку
        plt.grid()
        # добавляем легенду
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


# создадим класс для пересчета силы деформирования в напряжения и деформации
class Stress_strain (Data_xy):
    '''
    Класс Stress_strain: преобразование данных испытаний (усилие-перемещение) в напряжения и деформации

    ПОРЯДОК РАБОТЫ:
    
    1. ИНИЦИАЛИЗАЦИЯ:
       - Передача: данные перемещения (x_data), нагрузки (y_data), геометрии образца (L0, A0, Le)
       - Автоматическое определение типа испытания: сжатие (Le < L0) или растяжение (Le > L0)
       - Инициализация массивов для результатов
    
    2. РАСЧЕТ ЖЕСТКОСТИ МАШИНЫ (опционально):
       - get_stifness(): расчет жесткости испытательной машины по последней точке данных
       - Используется только если жесткость неизвестна
    
    3. РАСЧЕТ ИНЖЕНЕРНЫХ ВЕЛИЧИН:
       - get_strain_eng(): инженерная деформация с учетом упругой деформации машины
       - get_stress_eng(): инженерное напряжение (нагрузка / начальная площадь)
    
    4. РАСЧЕТ ИСТИННЫХ ВЕЛИЧИН:
       - get_strain_true(): истинная деформация (логарифмическая)
       - get_stress_true(): истинное напряжение с учетом изменения площади
    
    ТИПОВАЯ ПОСЛЕДОВАТЕЛЬНОСТЬ:
    1. Создание объекта
    2. get_stifness() (если нужно)
    3. get_strain_eng() -> get_stress_eng()
    4. get_strain_true() -> get_stress_true()
    
    ОСОБЕННОСТИ:
    - Учет жесткости испытательной машины
    - Поддержка растяжения и сжатия
    - Поэтапные расчеты с сохранением промежуточных результатов
    '''
    def __init__(self, x_data, y_data, L0, A0, Le,
                 k= np.inf,
                 data_name='Default data'):
        super().__init__(x_data, y_data, data_name)
        self.L0 = L0 # начальная длина образца
        self.A0 = A0 # площадь поперечного сечения образца
        self.Le = Le # конечный размер образца
        
        if Le < L0:
            self.test_type = 'compression'
        elif Le > L0:
            self.test_type = 'tension'
        else: raise ValueError ('Конечная и начальная длины \
                                образцов не могут быть равны')

        ### Переменные для хранения напряжений/деформаций
        self.stress_eng, self.strain_eng = None, None
        self.stress_true, self.strain_true = None, None
        
        ### Переменная для хранения жесткости испытательной машины
        self.k = k # по умолчанию равна бесконечности, т.е. абсолютно твердое тело
        
        ### Переменная для хранения упругой деформации машины при 
        # данном ходе деформирования
        self.machine_deformation = np.zeros(len(x_data))
        
        # переменная для хранения изменения длины образца
        self.specimen_deformation = np.zeros(len(self.x_data))
        
        # переменная для хранения текущей длины образца
        self.L = np.zeros(len(self.x_data))

    @property
    def stroke (self): 
        return self.x_data
    @property
    def load (self): 
        return self.y_data

    def get_stifness (self):
        '''
        метод для расчета жесткости испытательной машины
        '''
        if self.test_type == 'tension':
            pass
        else:
            if isinstance (self.Le, (int, float)):
                # Рассчитываем изменение размера образца
                delta_L = self.L0 - self.Le
                # Рассчитываем упругую деформацию машины
                elastic_machine_displasement = self.stroke[-1] - delta_L
                # Рассчитываем жеткость машины
                self.k = self.load[-1]/elastic_machine_displasement
                return self.k       
            else:
                raise ValueError ("Неподдерживаемый тип данных для переменной self.Le (конечная длина образц)")

    def get_strain_eng (self):
        '''
        Расчет инженерной деформации по формуле из Лекции №2
        '''

        # Рассчитываем деформацию машины для каждой точки
        self.machine_deformation = self.load/self.k

        # Рассчитываем деформацию образца
        self.specimen_deformation = self.stroke - self.machine_deformation
        # пересчитываем текущую длину образца
        self.L = self.L0-self.specimen_deformation
                
        self.strain_eng = (self.specimen_deformation/self.L0)*100

        return self.strain_eng # т.к. мы внутри функции расчитываем переменную self.strain_eng то можно было бы обойтись без
                                # return. Но нам удобно будет, если все-таки функция будет напрямую возвращать переменную 
                                # чуть позже, поэтому оставим ее 

    def get_stress_eng (self):
        '''
        Расчет инженерного напряжения по формуле из Лекции №2
        '''

        self.stress_eng = self.load/self.A0

        return self.stress_eng


    def get_strain_true (self):
        '''
        Расчет истинной деформации по формуле из Лекции №2
        '''
        # если инженерная деформация ранее не была посчитана, то 
        # посчитаем ее
        if self.strain_eng is None:
            self.strain_eng = self.get_strain_eng()
            
                
        if self.test_type == 'compression':
            self.strain_true = abs(np.log(1 - self.strain_eng/100))
        elif self.test_type == 'tension':
            self.strain_true = np.log(1 + self.strain_eng/100)
        else: 
            raise ValueError ('Что-то пошло не так!')

        return self.strain_true

    
    def get_stress_true (self):
        '''
        Расчет истинного напряжения по формуле из Лекции №2
        '''
        # Если инженерные напряжения и деформация не были ранее посчитаны,
        # то посчитаем их
        if self.strain_eng is None:
            self.strain_eng = self.get_strain_eng()
             
        if self.stress_eng is None:
            self.stress_eng = self.get_stress_eng()

        
        
        if self.test_type == 'compression':
            self.stress_true = self.stress_eng*(1 - self.strain_eng/100)
        elif self.test_type == 'tension':
            self.stress_true = self.stress_eng*(1 + self.strain_eng/100)
        else: 
            raise ValueError ('Что-то пошло не так!')

        return self.stress_true
    
    @staticmethod
    def get_flow_curve(strain: np.ndarray, stress: np.ndarray, 
                       yield_point: float, strain_max: float):
        '''
        Метод обрезает график истинное напряжение-деормации  слева (по 
        пределу текучести (аргумент yield_point)) и справа (по деформации 
        начала образования шейки (аргумент strain_max))
        '''
        
        # создаем фильтр
        filtr = (stress >= yield_point) & (strain < strain_max)
        
        # фильтруем данные
        strain_filtered = strain[filtr]
        stress_filtered = stress[filtr]
        
        
        # объединяем напряжение и деформацю в один массив
        flow_curve = np.array([strain_filtered, stress_filtered]).T
        
        # сместим график в 0 по оси деформации
        flow_curve[:, 0] = flow_curve[:, 0] - flow_curve[0, 0]
        
        return flow_curve 
        
        
        
        
        


# создадим класс для расчета мех характеристик
class Mech_properties (Data_xy):
    
    """класс для расчета механических свойств"""
    
    def __init__(self, x_data, y_data, data_name='Default data'):
        super().__init__(x_data, y_data, data_name)

        ### Переменные для хранения механических характеристик
        self.Rm, self.Ag, self.E, self.yield_stress = None, None, None, None

        ### Индекс при Rm
        self.indx_Rm = None

    # Поясним, что есть x, а что есть y в данном случае
    @property
    def strain (self): 
        return self.x_data
    @property
    def stress (self): 
        return self.y_data


    def get_E (self, elastic_limit = 0.05):
        '''
        Функция пытается расчитать модуль Юнга. Если вдруг инженерные напряжения и деформации не были расчитаны, то функция
        выкидывает ошибку
        '''
        try:
            filtr = self.strain_eng <= elastic_limit
            stress_elastic = self.stress_eng[filtr]
            strain_elastic = self.strain_eng[filtr]

            strain_elastic = strain_elastic/100 # переходим от процентов к долям от единицы
            
            numerator = np.sum(stress_elastic * strain_elastic)
            denominator = np.sum(strain_elastic ** 2)
            self.E = numerator / denominator
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
        Функция расчитывает предел текучести
        '''
        if not self.E:
            _ = self.get_E()

        filtr = self.strain_eng/100 - self.stress_eng/self.E >= strain_limit/100
        self.yeild_stress = self.stress_eng[filtr][0]
        return self.yeild_stress
        

    def get_properties (self):
        '''
        Функция пытается расчитать все мех. характеристики данного класса.
        '''
        if not self.Ag:
            self.Ag = self.get_Ag()
        if not self.yield_stress:
            self.yield_stress = self.get_yield_stress()
        return {'Модуль Юнга': self.E, 
                'Предел текучести': self.yield_stress, 
                'Предел прочтности': self.Rm, 
                'Равномерное удлинение': self.Ag}


if __name__ == "__main__":
    # считываем данные
    brass_array = np.loadtxt ('../dz/dz001/Fairuzova/brass.csv', delimiter=',', encoding='utf-8-sig')
    # Исходные размеры образца
    brass_data = {'Le': 8.8, 'd0': 13.6, 'L0': 17.9 }
    
    A0 = (np.pi*brass_data['d0']**2)/4
    
     # Создаем экземпляр класса
    brass_stress_strain = Stress_strain (brass_array[:,0], brass_array[:,1], 
                                   brass_data['L0'], A0, Le=brass_data['Le'])
    
    _ = brass_stress_strain.get_stifness()
    _ = brass_stress_strain.get_strain_eng()
    _ = brass_stress_strain.get_stress_eng()
    brass_stress_strain.plot_graph(brass_stress_strain.strain_eng, brass_stress_strain.stress_eng, 
                                 'Инженерная деформация, %', 
                                 'Инженерное напряжение, МПа')
    
    _ = brass_stress_strain.get_strain_true()
    _ = brass_stress_strain.get_stress_true()
    brass_stress_strain.plot_graph(brass_stress_strain.strain_true, brass_stress_strain.stress_true, 
                                 'Истинная деформация', 
                                 'Истинное напряжение, МПа')
    
#%%    
    steel_array = np.loadtxt('../data/load_stroke_data.txt', delimiter='\t', encoding='utf-8-sig')
    # Исходные размеры образца
    steel_data = {'Le': 80 + steel_array[-1, 0], 'a0': 1.5, 'b0':20, 'L0': 80 }

    A0 = steel_data['a0']*steel_data['b0']

    steel_stress_strain = Stress_strain (steel_array[:,0], steel_array[:,1], 
                                   steel_data['L0'], A0, Le=steel_data['Le'])
    
    #_ = steel_stress_strain.get_stifness()
    #_ = steel_stress_strain.get_strain_eng()
    #_ = steel_stress_strain.get_stress_eng()
    #steel_stress_strain.plot_graph(steel_stress_strain.strain_eng, steel_stress_strain.stress_eng, 
                                # 'Инженерная деформация, %', 
                                 #'Инженерное напряжение, МПа')
    
    _ = steel_stress_strain.get_strain_true()
    _ = steel_stress_strain.get_stress_true()
    steel_stress_strain.plot_graph(steel_stress_strain.strain_true, steel_stress_strain.stress_true, 
                                 'Истинная деформация', 
                                 'Истинное напряжение, МПа')
    
    
    # выделим кривую упрочнения
    flow_stress = Stress_strain.get_flow_curve(steel_stress_strain.strain_true, 
                                               steel_stress_strain.stress_true, 
                                               200., 0.3)
    
    # посмотрим на нее
    steel_stress_strain.plot_graph(flow_stress[:, 0], flow_stress[:, 1], 
                                 'Истинная деформация', 
                                 'Истинное напряжение, МПа')

    
    
    
    
    