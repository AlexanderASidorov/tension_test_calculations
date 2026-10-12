# Ось X: Деформация (в %)
deformations = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

# Ось Y: Напряжение (в кгс/мм^2)
# Кривая 1 (нижняя пунктирная линия)
stresses_kgf_1 = [45, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

# Кривая 2 (верхняя сплошная линия)
stresses_kgf_2 = [50, 58, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86]

# Коэффициент перевода из кгс/мм^2 в МПа
stresses_mpa_1 = [round(s * 9.80665, 2) for s in stresses_kgf_1]
stresses_mpa_2 = [round(s * 9.80665, 2) for s in stresses_kgf_2]

with open('Stress_1.txt', 'w', encoding='utf-8') as f:
    f.write('Деформация_%,Напряжение_МПа_1,Напряжение_МПа_2,\n')
    for i in range(len(stresses_mpa_1)):
        f.write(f'{deformations[i]:.1f}             {stresses_mpa_1[i]:.1f}            {stresses_mpa_2[i]:.1f}\n')

print('Stress_1.txt')
