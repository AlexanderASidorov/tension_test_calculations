import json
import numpy as np
import matplotlib.pyplot as plt
from Lection_006 import Data_xy


from scipy.optimize import differential_evolution
from scipy.optimize import curve_fit




class Curve (Data_xy):
    '''
    Класс для хранения кривой сопротивления пластической деформации
    '''
    def __init__(self, x_data, y_data, temperature, strain_rate, colour, data_name='Default data', line_type = '-'):
        super().__init__(x_data, y_data, data_name)
        
        self.colour = np.array(colour) / 255.0
        self.line_type = line_type
        
        self.temperature = temperature
        self.strain_rate = strain_rate
        
    @property
    def strain (self): 
        return self.x_data
    @property
    def stress (self): 
        return self.y_data
    @property
    def curve_name (self):
        return self.data_name

class SetOfCurves:
    
    '''
    Класс для хранения семейства кривых сопротивления пластической дефомрации
    '''
    def __init__(self, dict_of_curves):
        self.curves = dict_of_curves
        self.line_types = ['-', '--', ':', '*', 'x', 'o', '-.']
        
    def get_all_process_conditions (self):
        '''
        собирает все температуры, деформации, скорости деформации и напряжения
        в один массив
        '''
        temps = []
        strain_rates = []
        strains = []
        stresses = []
        
        for curve in self.curves.values():
            for i in range(len(curve.strain)):
                temps.append(curve.temperature)
                strain_rates.append(curve.strain_rate)
                strains.append(curve.strain[i])
                stresses.append(curve.stress[i])
        
        self.data = np.array([temps, strain_rates, strains, stresses]).T
        
        return self.data
    
    @staticmethod
    # функция для уравнения Хензеля–Шпиттеля
    def hensel_spittel(T, strain, strain_rate, A, m1, m2, m3, m4, m5, m6, m7, m9):
        """Вычисляет напряжение течения по эмпирической модели Хензеля–Шпиттеля. 
        Args:
            T : температура, К.
            strain: Истинная (логарифмическая) степень деформации (должна быть > 0).
            strain_rate: Скорость деформации, с⁻¹ (должна быть > 0).
            A (float): Материалозависимая константа, МПа.
            m1, m2, ..., m9 (float): Эмпирические коэффициенты модели (безразмерные, кроме A).
        Returns:
            array_like: Напряжение течения σ_f, МПа.
        """
        term1 = A
        term2 = np.exp(m1 * T)
        term3 = T**m9
        term4 = strain**m2
        term5 = np.exp(m4 / strain)
        term6 = (1 + strain)**(m5 * T)
        term7 = np.exp(-m6 * strain)
        term8 = strain_rate**m3
        term9 = strain_rate**(m7 * T)
        return term1 * term2 * term3 * term4 * term5 * term6 * term7 * term8 * term9
    
    @staticmethod 
    # функция-обертка для подстановки с метод curve_fit
    def model(X, A, m1, m2, m3, m4, m5, m6, m7, m9):
        T, strain_rate, strain = X
        return SetOfCurves.hensel_spittel(T, strain, strain_rate, A, m1, m2, m3, m4, m5, m6, m7, m9)
    
    @staticmethod 
    def r2_score(y_true, y_pred):
        '''
        Расчет коэффициента детерминации
        '''
        
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
        

   
  

if __name__ == '__main__':
    # Загрузка кривых сопротивления пластической деформации из JSON-файла
    with open('../data/OT4_1.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Создаём словарь куда будем записывать кривые
    curves = {}
  
    # создадим список возможных типов линии
    line_types = ['-', '--', ':', '*', 'x', 'o', '-.']
   
    # создадим множество куда будем записывать скорости деформации на которых определены кривые
    strain_rates = set()
    temperatures = set()
    
    for dataset in data['datasetColl']:
        name = dataset['name']  # имя кривой, например: "20 0.4" (20 градусов при скорости деформации 0.5 1/с)
        colour = dataset['colorRGB']
        points = dataset['data']  # список точек входящих в данную кривую
        temperature, strain_rate = name.split() # извлечем из имени кривой данные о температуре и скорости деформации
        temperature = float(temperature)
        strain_rate = float(strain_rate)
        strain_rates.add(round(strain_rate, 5))
        temperatures.add(round(temperature, 1))
        
        # Извлекаем значения [деформация, напряжение] и конвертируем в numpy массив
        values = [point['value'] for point in points]  # список списков
        xy_data = np.array(values) # массив координат
        
        # создаем объект класса Curve
        curve = Curve (xy_data[:, 0], xy_data[:, 1], temperature, strain_rate, colour, data_name=name)
        # добавляем его в словарь curves
        curves[name] = curve
        
    curves_object = SetOfCurves (curves)
    data_array = curves_object.get_all_process_conditions()
    
    # Настройка графика
    plt.figure(figsize=(10, 6))
    
    # будем итерировать по словарю
    for curve in curves.values():
        
        # найдем номер скорости деформации из множества strain_rates 
        indx = list(strain_rates).index(curve.strain_rate) 
        
        # в соответсвии с этим номером назначим тип кривой
        curve.line_type = line_types[indx]
                
        plt.plot (curve.strain, curve.stress, color = curve.colour, linestyle =curve.line_type,  label = curve.curve_name)
    
    # Оформление
    plt.xlabel('Истинная деформация')
    plt.ylabel('Истинное напряжение, МПа')
    plt.grid(True, linestyle='-', alpha=0.6)
    plt.legend(title='     T (°C) | ε̇ (1/с)')
    plt.tight_layout()
    plt.show()

#%% Попытка расчета коэффициентов уравнения 
    X = (data_array[:, 0], data_array[:, 1], data_array[:, 2])
    stress = data_array[:, 3]
    
    p0 = [10.0133, -0.0024, -0.2885, -0.3952,	-0.0623, -0.0035, 1.8931, 0.0007, 0.7280]
    
    popt, pcov = curve_fit(SetOfCurves.model, X, stress, p0=p0, maxfev=10000, ftol=1e-8, xtol=1e-8)
    print("Оптимальные коэффициенты:")
    params = ['A', 'm1', 'm2', 'm3', 'm4', 'm5', 'm7', 'm8', 'm9']
    for name, val in zip(params, popt):
        print(f"  {name} = {val:.6g}")
 
    # Предсказание модели
    # Предсказание модели
    sigma_pred = SetOfCurves.model(X, *popt)

    # Истинные (экспериментальные) значения
    sigma_true = stress
    
    # Кэоэффициент детерминации
    R2 = SetOfCurves.r2_score(sigma_true,  sigma_pred)

    # Диапазон для линии y = x
    min_val = min(sigma_true.min(), sigma_pred.min())
    max_val = max(sigma_true.max(), sigma_pred.max())

    # Построение parity plot
    plt.figure(figsize=(8, 8))
    plt.scatter(sigma_true, sigma_pred, alpha=0.7, edgecolors='k', s=30)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальное совпадение')
    plt.plot ([], [], ' ', label = f'R² = {R2:.2f}')
    
    plt.xlabel('Экспериментальное напряжение, МПа')
    plt.ylabel('Предсказанное напряжение, МПа')
    plt.title('Parity plot: Модель vs Эксперимент')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.axis('equal')  # чтобы угол был именно 45°
    plt.tight_layout()
    plt.show()






















    
    
