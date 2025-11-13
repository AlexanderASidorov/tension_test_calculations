#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np



class CochranTable:
    '''
    Класс для определения критического значения критерия Кохрана
    '''
    
    
    def __init__(self):
        
       # Таблица Кохрена для alpha = 0.05
        # f:    1      2      3      4      5      6      7      8      9      10     16     36     144    ∞
        self.table = np.array([
            [0.9985, 0.9750, 0.9392, 0.9057, 0.8584, 0.8534, 0.8332, 0.8159, 0.8010, 0.7880, 0.7341, 0.6602, 0.5813, 0.5000],  # n = 2
            [0.9669, 0.8709, 0.7977, 0.7457, 0.7071, 0.6771, 0.6530, 0.6333, 0.6167, 0.6025, 0.5466, 0.4748, 0.4031, 0.3333],  # n = 3
            [0.9065, 0.7679, 0.6841, 0.6287, 0.5895, 0.5598, 0.5365, 0.5175, 0.5017, 0.4884, 0.4366, 0.3720, 0.3093, 0.2500],  # n = 4
            [0.8412, 0.6838, 0.5981, 0.5440, 0.5063, 0.4783, 0.4564, 0.4387, 0.4241, 0.4118, 0.3645, 0.3066, 0.2513, 0.2000],  # n = 5
            [0.7808, 0.6161, 0.5321, 0.4803, 0.4447, 0.4184, 0.3980, 0.3817, 0.3682, 0.3568, 0.3135, 0.2612, 0.2119, 0.1667],  # n = 6
            [0.7271, 0.5612, 0.4800, 0.4307, 0.3907, 0.3726, 0.3555, 0.3384, 0.3254, 0.3154, 0.2756, 0.2277, 0.1833, 0.1429],  # n = 7
            [0.6798, 0.5157, 0.4377, 0.3910, 0.3595, 0.3362, 0.3185, 0.3043, 0.2926, 0.2829, 0.2462, 0.2022, 0.1616, 0.1250],  # n = 8
            [0.6385, 0.4775, 0.4027, 0.3584, 0.3286, 0.3067, 0.2901, 0.2768, 0.2659, 0.2568, 0.2226, 0.1820, 0.1446, 0.1111],  # n = 9
            [0.6020, 0.4450, 0.3733, 0.3311, 0.3029, 0.2823, 0.2666, 0.2541, 0.2439, 0.2353, 0.2032, 0.1655, 0.1308, 0.1000],  # n = 10
            [0.5410, 0.3924, 0.3264, 0.2880, 0.2624, 0.2439, 0.2299, 0.2187, 0.2098, 0.2020, 0.1737, 0.1403, 0.1100, 0.0833],  # n = 12
            [0.4709, 0.3346, 0.2758, 0.2419, 0.2195, 0.2034, 0.1911, 0.1815, 0.1736, 0.1671, 0.1429, 0.1144, 0.0889, 0.0677],  # n = 15
            [0.3894, 0.2705, 0.2205, 0.1921, 0.1735, 0.1602, 0.1501, 0.1422, 0.1357, 0.1303, 0.1108, 0.0879, 0.0675, 0.0500],  # n = 20
            [0.3434, 0.2354, 0.1907, 0.1656, 0.1493, 0.1374, 0.1286, 0.1216, 0.1160, 0.1113, 0.0942, 0.0743, 0.0567, 0.0417],  # n = 24
            [0.2929, 0.1980, 0.1593, 0.1377, 0.1237, 0.1137, 0.1061, 0.1002, 0.0958, 0.0921, 0.0771, 0.0604, 0.0457, 0.0333],  # n = 30
            [0.2370, 0.1576, 0.1259, 0.1082, 0.0968, 0.0887, 0.0827, 0.0780, 0.0745, 0.0713, 0.0595, 0.0462, 0.0347, 0.0250],  # n = 40
            [0.1737, 0.1131, 0.0895, 0.0766, 0.0682, 0.0623, 0.0583, 0.0552, 0.0520, 0.0497, 0.0411, 0.0316, 0.0234, 0.0167],  # n = 60
            [0.0998, 0.0632, 0.0495, 0.0419, 0.0371, 0.0337, 0.0312, 0.0292, 0.0279, 0.0266, 0.0218, 0.0165, 0.0120, 0.0083],  # n = 120
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]   # n = ∞
        ])
        
        self.n_values = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 24, 30, 40, 60, 120, float('inf')]  # число опытов
        self.f_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 36, 144, float('inf')]  # f = m - 1
                            

    
        
    def get_value(self, n_samples, n_repeats):
        """Получить критическое значение для заданных n и f"""
        self.n_samples, self.n_repeats  = n_samples, n_repeats
        
        self.f = n_repeats - 1
        
        if self.n_samples not in self.n_values:
            raise ValueError(f"n_samples должен быть в {self.n_values}")
        if self.f not in self.f_values:
            raise ValueError(f"n_repeats должен быть таким, чтобы f=n_repeats-1 был в {self.f_values}")
        
        n_idx = self.n_values.index(self.n_samples)
        f_idx = self.f_values.index(self.f)
        
        self.cochran_max = float(self.table[n_idx, f_idx])
        
        return self.cochran_max
    
class StudentTable:
    '''
    Класс для определения критического значения t-критерия Стьюдента
    '''
    
    def __init__(self):
        
        # Таблица t-критерия Стьюдента для α = 0.05 (двусторонний)
        self.table = np.array([
            12.71,  # f = 1
            4.30,   # f = 2
            3.18,   # f = 3
            2.78,   # f = 4
            2.57,   # f = 5
            2.45,   # f = 6
            2.36,   # f = 7
            2.31,   # f = 8
            2.26,   # f = 9
            2.23,   # f = 10
            2.20,   # f = 11
            2.18,   # f = 12
            2.16,   # f = 13
            2.14,   # f = 14
            2.13,   # f = 15
            2.12,   # f = 16
            2.11,   # f = 17
            2.10,   # f = 18
            2.09,   # f = 19
            2.09,   # f = 20
            2.08,   # f = 21
            2.07,   # f = 22
            2.07,   # f = 23
            2.06,   # f = 24
            2.06,   # f = 25
            2.06,   # f = 26
            2.05,   # f = 27
            2.05,   # f = 28
            2.05,   # f = 29
            2.04,   # f = 30
            2.02,   # f = 40
            2.00,   # f = 60
            1.98,   # f = 120
            1.96    # f = ∞
        ])
        
        self.f_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
                        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 
                        29, 30, 40, 60, 120, float('inf')]
    
    def get_value(self, n_samples, n_repeats):
        """Получить критическое значение для заданного числа степеней свободы"""
        
        self.n_samples, self.n_repeats  = n_samples, n_repeats
        self.f = n_samples*(n_repeats-1)
        
        
        if self.f not in self.f_values:
            raise ValueError(f"degrees_of_freedom должен быть в {self.f_values}")
        
        f_idx = self.f_values.index(self.f)
        
        self.student_min = float(self.table[f_idx])
        
        return self.student_min

class FisherTable:
    '''
    Класс для определения критического значения F-критерия Фишера (α = 0.05)
    '''
    
    def __init__(self):
        # Таблица F-критерия Фишера для α = 0.05
        # f2 (по горизонтали): 1, 2, 3, 4, 5, 6, 8, 12, 24, ∞
        self.table = np.array([
            [161.40, 199.50, 215.70, 224.60, 230.20, 234.00, 238.90, 243.90, 249.00, 254.30],  # f1 = 1
            [18.51,  19.00,  19.16,  19.25,  19.30,  19.33,  19.37,  19.41,  19.45,  19.50],   # f1 = 2
            [10.13,  9.55,   9.28,   9.12,   9.01,   8.94,   8.84,   8.74,   8.64,   8.53],    # f1 = 3
            [7.71,   6.94,   6.59,   6.39,   6.26,   6.16,   6.04,   5.91,   5.77,   5.63],    # f1 = 4
            [6.61,   5.79,   5.41,   5.19,   5.05,   4.95,   4.82,   4.68,   4.53,   4.36],    # f1 = 5
            [5.99,   5.14,   4.76,   4.53,   4.39,   4.28,   4.15,   4.00,   3.84,   3.67],    # f1 = 6
            [5.59,   4.74,   4.35,   4.12,   3.97,   3.87,   3.73,   3.57,   3.41,   3.23],    # f1 = 7
            [5.32,   4.46,   4.07,   3.84,   3.69,   3.58,   3.44,   3.28,   3.12,   2.93],    # f1 = 8
            [5.12,   4.26,   3.86,   3.63,   3.48,   3.37,   3.23,   3.07,   2.90,   2.71],    # f1 = 9
            [4.96,   4.10,   3.71,   3.48,   3.33,   3.22,   3.07,   2.91,   2.74,   2.54],    # f1 = 10
            [4.84,   3.98,   3.59,   3.36,   3.20,   3.09,   2.95,   2.79,   2.61,   2.40],    # f1 = 11
            [4.75,   3.88,   3.49,   3.26,   3.11,   3.00,   2.85,   2.69,   2.50,   2.30],    # f1 = 12
            [4.67,   3.80,   3.41,   3.18,   3.02,   2.92,   2.77,   2.60,   2.42,   2.21],    # f1 = 13
            [4.60,   3.74,   3.34,   3.11,   2.96,   2.85,   2.70,   2.53,   2.35,   2.13],    # f1 = 14
            [4.54,   3.68,   3.29,   3.06,   2.90,   2.79,   2.64,   2.48,   2.29,   2.07],    # f1 = 15
            [4.49,   3.63,   3.24,   3.01,   2.85,   2.74,   2.59,   2.42,   2.24,   2.01],    # f1 = 16
            [4.45,   3.59,   3.20,   2.96,   2.81,   2.70,   2.55,   2.38,   2.19,   1.96],    # f1 = 17
            [4.41,   3.55,   3.16,   2.93,   2.77,   2.66,   2.51,   2.34,   2.15,   1.92],    # f1 = 18
            [4.38,   3.52,   3.13,   2.90,   2.74,   2.63,   2.48,   2.31,   2.11,   1.88],    # f1 = 19
            [4.35,   3.49,   3.10,   2.87,   2.71,   2.60,   2.45,   2.28,   2.08,   1.84],    # f1 = 20
            [4.32,   3.47,   3.07,   2.84,   2.68,   2.57,   2.42,   2.25,   2.05,   1.81],    # f1 = 21
            [4.30,   3.44,   3.05,   2.82,   2.66,   2.55,   2.40,   2.23,   2.03,   1.78],    # f1 = 22
            [4.28,   3.42,   3.03,   2.80,   2.64,   2.53,   2.38,   2.20,   2.00,   1.76],    # f1 = 23
            [4.26,   3.40,   3.01,   2.78,   2.62,   2.51,   2.36,   2.18,   1.98,   1.73],    # f1 = 24
            [4.24,   3.38,   2.99,   2.76,   2.60,   2.49,   2.34,   2.16,   1.96,   1.71],    # f1 = 25
            [4.22,   3.37,   2.98,   2.74,   2.59,   2.47,   2.32,   2.15,   1.95,   1.69],    # f1 = 26
            [4.21,   3.35,   2.96,   2.73,   2.57,   2.46,   2.30,   2.13,   1.93,   1.67],    # f1 = 27
            [4.20,   3.34,   2.95,   2.71,   2.56,   2.44,   2.29,   2.12,   1.91,   1.65],    # f1 = 28
            [4.18,   3.33,   2.93,   2.70,   2.54,   2.43,   2.28,   2.10,   1.90,   1.64],    # f1 = 29
            [4.17,   3.32,   2.92,   2.69,   2.53,   2.42,   2.27,   2.09,   1.89,   1.62],    # f1 = 30
            [4.08,   3.23,   2.84,   2.61,   2.45,   2.34,   2.18,   2.00,   1.79,   1.52],    # f1 = 40
            [4.00,   3.15,   2.76,   2.52,   2.37,   2.25,   2.10,   1.92,   1.70,   1.39],    # f1 = 60
            [3.92,   3.07,   2.68,   2.45,   2.29,   2.17,   2.02,   1.83,   1.61,   1.25],    # f1 = 120
            [3.84,   2.99,   2.60,   2.37,   2.21,   2.09,   1.94,   1.75,   1.52,   1.00]     # f1 = ∞
        ])
        
        self.f1_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                         21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 40, 60, 120, float('inf')]  # степени свободы числителя
        
        self.f2_values = [1, 2, 3, 4, 5, 6, 8, 12, 24, float('inf')]  # степени свободы знаменателя
    
    
    def get_value(self, n_samples, n_repeats, coefs):
        """
        Получить критическое значение F-критерия для заданных степеней свободы
        """
        self.n_samples, self.n_repeats, self.coefs  = n_samples, n_repeats, coefs
        
        self.f2 = int(self.n_repeats - 1)
        n_non_zero_coefs = np.count_nonzero(self.coefs)
        
        self.f1 = int(self.n_samples - n_non_zero_coefs) 
        
        
        if self.f1 not in self.f1_values:
            raise ValueError(f"f1 должен быть в {self.f1_values}")
        if self.f2 not in self.f2_values:
            raise ValueError(f"f2 должен быть в {self.f2_values}")
        
        f1_idx = self.f1_values.index(self.f1)
        f2_idx = self.f2_values.index(self.f2)
        
        self.fisher_max = float(self.table[f1_idx, f2_idx])
        
        return self.fisher_max

class Full_factorial_matrix:
    '''
    Класс для построения матрицы планирования полного факторного эксперимента
    '''
    def __init__(self, n_factors):
        
        if type(n_factors) is not int:
            raise TypeError ('Переменная n_factors должна быть типа int')
        if n_factors <= 1:
            raise ValueError ('Переменная n_factors должна быть типа int и больше 1')
        self.n_factors = n_factors
        
        self.matrix_normalized = self.set_full_factorial_matrix()
        self.n_samples = self.matrix_normalized.shape[0]
 
        
        
    def set_full_factorial_matrix(self):
        '''
        Метод для построения матрицы планирвоания ПФЭ в кодированном масштабе
        '''
        
        n_factors = self.n_factors
        # необходимое количество экспериментов
        n_experiments = 2 ** n_factors
        
        # заготовка для матрицы ПФЭ
        matrix = []
        
        for i in range(n_experiments):
            # Преобразуем номер опыта в двоичное число
            binary = format(i, f'0{n_factors}b')
            # 0 → -1, 1 → +1 и разворачиваем порядок
            row = [1 if bit == '1' else -1 for bit in binary][::-1]
            matrix.append(row)
        
        # Инвертируем первый столбец
        matrix = np.array(matrix)
        matrix[:, 0] = -matrix[:, 0]  # меняем знак у первого столбца
        
        return matrix

    
class Regression:
    '''
    Класс для расчета коэффициентов линейной регрессии при ПФЭ
    '''
    
    def __init__(self,  matrix_normalized, target_values):
        
        for item in [matrix_normalized, target_values]:
            if type(item) is not np.ndarray:
                raise TypeError ('Переменная {item} должна быть типа массива NumPy')
                
        if matrix_normalized.shape[0] != target_values.shape[0]:
            raise ValueError ('Матрица планирования и матрица целеыой переменной имеют разное количество строк!')
                
        self.matrix_normalized = matrix_normalized
        self.target_values = target_values 
        
        self.target_values_avg = np.mean(target_values, axis = 1)
        
        self.n_repeats = self.target_values.shape[1]
        self.n_samples = self.target_values.shape[0]
        
        cochran = CochranTable()
        self.cochran_max = cochran.get_value(self.n_samples, self.n_repeats)
        
        student = StudentTable()
        self.student_max = student.get_value(self.n_samples, self.n_repeats)
        
  
    def get_S2 (self):
        '''
        Метод для расчета построчной дисперсии целевой переменной и 
        дисперсии воспроизводимости
        '''
        self.S2 = np.var(self.target_values, axis=1, ddof=1)
        self.S2_vospr = float(np.mean(self.S2))
        
        # Расчетный критерий Кохрена
        self.cochran = float(np.max(self.S2) / np.sum(self.S2))
        
        if self.cochran < self.cochran_max:
            print("✅ Дисперсии однородны. Можно продолжать анализ.")
        else:
            print("❌ Дисперсии неоднородны. Модель может быть ненадёжной.")
        
        
        

        
    def get_coefficients (self):
        '''
        Метод для расчета коэффициентов линейной регрессии
        '''
        
        # Оставляйте как есть - это стандартный подход для DOE
        X = np.column_stack([np.ones(self.n_samples), self.matrix_normalized])
        y = self.target_values_avg
        
        XTX = X.T @ X
        XTX_inv = np.linalg.inv(XTX)
        self.coefs = XTX_inv @ X.T @ y
        return self.coefs
  
    
    def check_coefficients (self):
        '''
        Проверка значимости коэффициентов
        '''
        
        required_attrs = ['coefs', 'S2_vospr', 'n_samples', 'student_max']
        
        # Проверяем все необходимые атрибуты
        if not all(hasattr(self, attr) for attr in required_attrs):
            raise ValueError("Не все необходимые атрибуты рассчитаны.")
    
        
        # если коэффициенты уже посчитаны, то можно проверять их значимость
        SE_b = np.sqrt(self.S2_vospr / self.n_samples)
        self.student = np.abs(self.coefs / SE_b)
        for i in range(len(self.coefs)):
            # если расчетный коэффициент меньше табличного, то коэффициент не 
            # значм и его можно обнулить
            if self.student[i] < self.student_max:
                self.coefs[i] = 0.0
            else:
                pass
 




if __name__ == '__main__':        
        
    matrix = Full_factorial_matrix(3)
    

    matrix_4 = matrix.matrix_normalized
    
    
    
    
 #%% 
    target_values = y_repeated = np.array([[238.5, 237.0, 239.2],  # Опыт 1: C=0.10%, h=0.7 мм, θ=0°  → DC01
                                           [160.9, 161.4, 158.0],  # Опыт 2: C=0.06%, h=0.7 мм, θ=0°  → DC04
                                           [240.2, 241.9, 239.0],  # Опыт 3: C=0.10%, h=1.2 мм, θ=0° → DC01
                                           [183.1, 185.4, 187.8],  # Опыт 4: C=0.06%, h=1.2 мм, θ=0° → DC04
                                           [241.4, 244.8, 242.3],  # Опыт 5: C=0.10%, h=0.7 мм, θ=90° → DC01
                                           [165.6, 163.2, 165.5],  # Опыт 6: C=0.06%, h=0.7 мм, θ=90° → DC04
                                           [243.1, 243.1, 241.2],  # Опыт 7: C=0.10%, h=1.2 мм, θ=90° → DC01
                                           [201.9, 198.3, 202.1]])  # Опыт 8: C=0.06%, h=1.2 мм, θ=90° → DC04
    
    regression = Regression(matrix_4, target_values)
    
    

    regression.get_S2()
    
    cochran = CochranTable()
    cochran_coef = cochran.get_value(regression.n_samples, regression.n_repeats)
    
    student = StudentTable()
    student_coef = student.get_value(regression.n_samples, regression.n_repeats)
    
    _ = regression.get_coefficients()
    
    _ = regression.check_coefficients()
    
    
    fisher = FisherTable()
    fisher_coef = fisher.get_value(regression.n_samples, regression.n_repeats, regression.coefs)























