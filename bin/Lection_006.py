#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
from scipy.stats import linregress
from scipy.interpolate import interp1d


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
    def plot_graphs (list_of_x_data, list_of_y_data, list_of_curve_labels , x_label, y_label):
        '''    
        Функция позволяет вывести график y = f(x)
        
        Параметры
        ------------
        list_of_x_data - список векторов которые откладываются по оси X. 
        list_of_y_data - список векторов которые которые откладываются по оси Y.
        list_of_curve_labels - список имен кривых
        x_label - название вектора данных по оси Х. Должен быть переменной типа string
        y_label - название вектора данных по оси Y. Должен быть переменной типа string
        curve_label = название кривой. По умолчание None. Если не None, то должен быть переменной типа string
    
        Возвращает
        -------------
        None
        '''
        # Визуализируем результаты
        # создадим список из цветов для кривых
        colors = plt.cm.tab10(range(len(list_of_x_data)))
        
        # создаем "базу" для графика
        plt.figure(figsize=(6, 4))
        for i in range (len(list_of_x_data)):
            x_data, y_data = list_of_x_data[i], list_of_y_data [i]
            curve_label =  list_of_curve_labels[i]
            plt.plot (x_data, y_data, '-', color = colors[i], label = curve_label)
        
        # Добавляем названия осей
        plt.xlabel (x_label, fontsize = 14)
        plt.ylabel (y_label, fontsize = 14)
        # добавляем для красоты сетку
        plt.grid()
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
                return self.k, elastic_machine_displasement     
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
                
        strain_eng = (self.specimen_deformation/self.L0)*100
        
        self.strain_eng = np.asarray(strain_eng, dtype=float)

        return self.strain_eng # т.к. мы внутри функции расчитываем переменную self.strain_eng то можно было бы обойтись без
                                # return. Но нам удобно будет, если все-таки функция будет напрямую возвращать переменную 
                                # чуть позже, поэтому оставим ее 

    def get_stress_eng (self):
        '''
        Расчет инженерного напряжения по формуле из Лекции №2
        '''

        stress_eng = self.load/self.A0
        
        self.stress_eng = np.asarray(stress_eng, dtype=float)

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
            strain_true = np.abs(np.log(1 - self.strain_eng/100))
        elif self.test_type == 'tension':
            strain_true = np.log(1 + self.strain_eng/100)
        else: 
            raise ValueError ('Что-то пошло не так!')


        self.strain_true = np.asarray(strain_true, dtype=float)

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
            stress_true = self.stress_eng*(1 - self.strain_eng/100)
        elif self.test_type == 'tension':
            stress_true = self.stress_eng*(1 + self.strain_eng/100)
        else: 
            raise ValueError ('Что-то пошло не так!')

        self.stress_true = np.asarray(stress_true, dtype=float)

        return self.stress_true
    

        
        
# создадим класс для расчета мех характеристик
class Flow_curve (Data_xy):
    
    """класс для расчета механических свойств"""
    
    def __init__(self, x_data, y_data, data_name='Default data'):
        super().__init__(x_data, y_data, data_name)
        
        self.flow_curve = None
        self.flow_strart_indx = None

    # Поясним, что есть x, а что есть y в данном случае
    @property
    def strain_true (self): 
        return self.x_data
    @property
    def stress_true (self): 
        return self.y_data


    def interpolate_data(self, num_points=1000, kind='cubic', 
                          x_min=None, x_max=None):
        """
        Интерполирует кривую истинное напряжение–деформация для получения более плотной и гладкой зависимости.
        
        Параметры:
            num_points : int, optional
                Количество точек в интерполированной кривой. По умолчанию 500.
            kind : str, optional
                Тип интерполяции: 'linear', 'quadratic', 'cubic'. По умолчанию 'cubic'.
            x_min, x_max : float, optional
                Диапазон деформации для интерполяции. Если не заданы — используются минимум и максимум исходных данных.
            data_name_suffix : str, optional
                Суффикс для имени нового объекта (добавляется к self.data_name).
        
        Возвращает:
            Flow_curve
                Новый объект класса Flow_curve с интерполированными данными.
        """
        x = self.x_data
        y = self.y_data

        if len(x) < 2:
            raise ValueError("Недостаточно точек для интерполяции (требуется >= 2).")
        
        # Упорядочиваем данные по x (на случай, если они не упорядочены)
        sorted_idx = np.argsort(x)
        x_sorted = x[sorted_idx]
        y_sorted = y[sorted_idx]

        # Удаляем дубликаты по x 
        unique_idx = np.unique(x_sorted, return_index=True)[1]
        x_clean = x_sorted[unique_idx]
        y_clean = y_sorted[unique_idx]

        if len(x_clean) < 2:
            raise ValueError("После удаления дубликатов осталось менее 2 точек.")

        # Определяем диапазон интерполяции
        x_min = x_min if x_min is not None else x_clean.min()
        x_max = x_max if x_max is not None else x_clean.max()

        # Генерируем новые x-значения
        x_new = np.linspace(x_min, x_max, num_points)

        # Создаём интерполянт
        try:
            interpolator = interp1d(x_clean, y_clean, kind=kind, fill_value="extrapolate")
        except ValueError as e:
            raise ValueError(f"Ошибка при создании интерполянта: {e}")

        # Вычисляем y_new
        y_new = interpolator(x_new)
        
        self.x_data = x_new
        self.y_data = y_new
        


    def find_plastic_start(self, stress_threshold, 
                       max_r2=0.995, min_points=5):
        
        """
        Находит индекс начала пластической деформации (конец упругого участка).
        
        Параметры:
            stress_threshold : float — минимальное напряжение для начала поиска (МПа)
            max_r2 : float — R^2, выше которого считаем участок линейным
            min_points : int — минимальное число точек для аппроксимации
        
        Возвращает:
            idx_plastic_start : int — индекс первой точки, где начинается нелинейность
        """
        strain_true, stress_true = self.strain_true, self.stress_true
    
    
        # 1. Фильтруем по напряжению
        filtr = stress_true >= stress_threshold
        if not np.any(filtr):
            raise ValueError("Нет данных с напряжением >= заданного порога")
    
        idxs = np.where(filtr)[0]
        strain_sub = strain_true[filtr]
        stress_sub = stress_true[filtr]
        
        n = len(stress_sub)
        if n < min_points:
            raise ValueError("Недостаточно точек после фильтрации")
        
        # 2. Перебираем возможные "концы" линейного участка
        best_end = min_points - 1  # хотя бы min_points точек
        for i in range(min_points, n):
            # Аппроксимируем прямую на участке [0:i]
            slope, intercept, r_value, p_value, std_err = linregress(strain_sub[:i], stress_sub[:i])
            r2 = r_value ** 2
            
            # Если R² упал ниже порога — линейность нарушена
            if r2 < max_r2:
                best_end = i - 1
                break
        else:
            # Если весь участок линейный — считаем, что пластичность не началась
            best_end = n - 1
    
        
        self.flow_strart_indx = int(idxs[best_end] + 1)
        
                    
        # Возвращаем индекс в исходном массиве
        return self.flow_strart_indx  # +1 — чтобы начать СЛЕДУЮЩУЮ точку как пластическую
    
    
    def find_plastic_start_derivative(self, stress_threshold, 
                                  min_points_for_derivative=5, 
                                  derivative_threshold_factor=0.1):
        """
        Находит начало пластического упрочнения через производную.
        Ищет, где производная напряжения по деформации падает ниже порога.
    
        Параметры:
            stress_threshold : float — минимальное напряжение для начала поиска (МПа)
            min_points_for_derivative : int — мин. точек для расчёта производной
            derivative_threshold_factor : float — доля от начальной производной, ниже которой начинается нелинейность
    
        Возвращает:
            idx_plastic_start : int — индекс начала упрочнения
        """
        strain_true, stress_true = self.strain_true, self.stress_true
    
        # Фильтруем по напряжению
        mask = stress_true >= stress_threshold
        if not np.any(mask):
            raise ValueError("Нет данных с напряжением >= заданного порога")
    
        idxs = np.where(mask)[0]
        strain_sub = strain_true[mask]
        stress_sub = stress_true[mask]
    
        if len(stress_sub) < min_points_for_derivative:
            raise ValueError("Недостаточно точек после фильтрации")
    
        # Считаем производную: dσ/dε
        d_sigma_d_epsilon = np.gradient(stress_sub, strain_sub)
    
        # Начальная производная (упругий модуль или модуль на площадке)
        initial_slope = np.mean(d_sigma_d_epsilon[:min_points_for_derivative])
    
        # Порог: например, 10% от начальной производной
        threshold = derivative_threshold_factor * initial_slope
    
        # Находим первую точку, где производная падает ниже порога
        plastic_idx_local = np.where(d_sigma_d_epsilon < threshold)[0]
    
        if len(plastic_idx_local) == 0:
            # Не нашли — считаем, что упрочнения нет
            idx_global = idxs[-1]
        else:
            idx_local = plastic_idx_local[0]
            idx_global = idxs[idx_local]
    
        self.flow_strart_indx = int(idx_global)
        return self.flow_strart_indx
    
    

    
    def set_flow_curve(self):
        """
        Формирует кривую течения (напряжение-деформация после начала пластической деформации),
        обрезая упругую часть и смещая начало пластической части в 0 по оси деформации.
    
        Метод:
        - Если индекс начала пластической деформации (self.flow_strart_indx) ещё не вычислен,
          вызывает метод find_plastic_start() для его определения.
        - Извлекает подмассивы истинной деформации и истинного напряжения,
          начиная с индекса self.flow_strart_indx.
        - Объединяет их в массив размером (N, 2), где:
            - столбец 0 — истинная деформация (смещённая в 0),
            - столбец 1 — истинное напряжение.
        - Смещение деформации: вычитается первое значение деформации из всех значений,
          чтобы кривая начиналась с 0 по оси X.
    
        Атрибуты:
            self.flow_curve : ndarray, shape (N, 2)
                Кривая течения в формате [[ε_0, σ_0], [ε_1, σ_1], ...],
                где ε — смещённая истинная деформация, σ — истинное напряжение.
    
        Возвращает:
            self.flow_curve : ndarray
                Кривая течения (см. выше).
        """
        if not isinstance(self.flow_strart_indx, int):
            raise ValueError("Еще не найдена точка начала пластической деформации. Воспользуетесь методом find_plastic_start")
    
        if self.flow_strart_indx >= len(self.strain_true):
            raise ValueError("Индекс начала пластической деформации выходит за пределы массива.")
    
        strain_plastic = self.strain_true[self.flow_strart_indx:]
        stress_plastic = self.stress_true[self.flow_strart_indx:]
    
        if len(strain_plastic) == 0 or len(stress_plastic) == 0:
            raise ValueError("После отсечения упругой части не осталось данных для кривой течения.")
    
        # Объединяем в массив (N, 2)
        self.flow_curve = np.column_stack((strain_plastic, stress_plastic))
    
        # Смещаем деформацию в 0
        self.flow_curve[:, 0] -= self.flow_curve[0, 0]
    
        return self.flow_curve
 
    @staticmethod
    def get_flow_curve_manually(strain: np.ndarray, stress: np.ndarray, 
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
    
    def __init__(self, x_data, y_data, stress_type = 'eng', data_name='Default data'):
        super().__init__(x_data, y_data, data_name)
        
        if stress_type in ['eng', 'true']:
            self.stress_type = stress_type
        else:
            print ('Атрибут stress_type может быть только eng либо true. Было принято дефолтное значение eng')
            self.stress_type = 'eng'

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


    def get_E (self, strain_limit_left = 0, strain_limit_right = 0.05):
        '''
        Функция пытается расчитать модуль Юнга. Если вдруг инженерные напряжения и деформации не были расчитаны, то функция
        выкидывает ошибку
        '''
        
        strain = self.strain
        stress = self.stress
              
        
        try:
            filtr = (strain <= strain_limit_right)&(strain >= strain_limit_left)
            stress_elastic = stress[filtr]
            strain_elastic = strain[filtr]
            

            if self.stress_type == 'eng':
                strain_elastic = strain_elastic/100 # переходим от процентов к долям от единицы
              
            self.E = (stress_elastic[-1] - stress_elastic[0])/(strain_elastic[-1] - strain_elastic[0])
            
            return self.E
        
        except Exception:
            print (f'Ошибка {Exception}')
            print ('Судя по всему деформация и напряжения не расчитаны или расчитаны неправильно.')
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
        
        strain = self.strain
        stress = self.stress
        
        if not self.E:
            _ = self.get_E()
            
        if self.stress_type == 'eng':
            strain = strain/100 # переходим от процентов к долям от единицы

        
        filtr = strain - stress/self.E >= strain_limit
        self.yeild_stress = stress[filtr][0]
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


#%%
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
    
    
    stifnesses = {'brass test': brass_stress_strain.k}
    
    
    
#%%    
    steel_array = np.loadtxt('../dz/dz001/Fairuzova/3_20.csv', delimiter=',', encoding='utf-8-sig')
    # Исходные размеры образца
    steel_data = {'Le': 4.14,  'd0':9.20, 'L0': 14.9 }

    A0 = (np.pi*steel_data['d0']**2)/4

    steel_stress_strain = Stress_strain (steel_array[:,0], steel_array[:,1], 
                                   steel_data['L0'], A0, Le=steel_data['Le'])
    
    # посчитаем жесткость
    _ = steel_stress_strain.get_stifness()
    stifnesses['steel test'] = steel_stress_strain.k
    
    
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
    flow_stress = Flow_curve.get_flow_curve_manually(steel_stress_strain.strain_true, 
                                               steel_stress_strain.stress_true, 
                                               420., 1.2)
    
    # посмотрим на нее
    steel_stress_strain.plot_graph(flow_stress[:, 0], flow_stress[:, 1], 
                                 'Истинная деформация', 
                                 'Истинное напряжение, МПа')
    
    
    # Создадим объект для расчета мех. характиристик (модуля Юнга и предела текучести)
    mech_properties = Mech_properties (steel_stress_strain.strain_true, steel_stress_strain.stress_true, 
                                       stress_type = 'true')
    
    # Посмотрим внимательнее на "около" упругую часть графика
    filtr01 = mech_properties.strain <= 0.1
    steel_stress_strain.plot_graph(steel_stress_strain.strain_true[filtr01], steel_stress_strain.stress_true[filtr01], 'Истинная деформация', 'Истинное напряжение, Н')
    _ = mech_properties.get_E(strain_limit_left = 0.017, strain_limit_right = 0.022)
    
    print (f'Модуль Юнга равен: {mech_properties.E:.2f} МПа')
    

    
    
    