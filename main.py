import pandas as pd
from pandas import DataFrame

from pcm import PairwiseComparisonMatrix

import numpy as np

car_criteria = np.array([2, 5, 0.20, 5, 5, 0.25,
                         3, 1, 3, 0.3, 0.25,
                         1, 2, 1, 0.25,
                         0.3, 0.33, 9,
                         9, 0.11,
                         3])

car_alternatives = np.array(
    # cost, fuel consumption (l/100km), speed (km/h), power, ground clearance (ml), comfort of the cabin, brand
    [[3, 7, 1.5, 1.33, 1.25, 0.6, 1.33],  # Hyundai Solaris vs Kia Rio
     [0.20, 9, 0.75, 1.33, 2.5, 0.75, 0.8],  # Hyundai Solaris vs Chevrolet Cruze
     [0.33, 0.33, 1, 1, 1.67, 1.5, 1],  # Hyundai Solaris vs Renault Logan
     [9, 5, 2.5, 0.75, 1.25, 1, 2],  # Hyundai Solaris vs Лада Веста
     [0.11, 0.11, 0.60, 1, 1.67, 0.75, 1.33],  # Hyundai Solaris vs Volkswagen Tiguan
     [0.33, 7, 0.50, 1, 2, 1.25, 0.6],  # Kia Rio
     [0.40, 9, 0.67, 0.75, 1.33, 2.5, 0.75],
     [3, 0.14, 0.5, 1, 1, 1.67, 1.5],
     [0.14, 5, 0.40, 0.75, 1.33, 1.25, 1],
     [3, 9, 0.14, 0.75, 0.67, 2, 1.25],  # Chevrolet Cruze
     [0.33, 0.11, 1, 1, 0.5, 1.33, 2.5],
     [3, 7, 2.5, 0.75, 0.75, 0.67, 2],
     [0.11, 1, 0.75, 0.75, 1, 0.5, 5],  # Renault Logan
     [3, 3, 0.33, 1.33, 0.11, 0.33, 2.5],
     [5, 0.33, 0.80, 0.33, 5, 0.33, 9]]  # Лада Веста
    # Volkswagen Tiguan
)

criteria_comparsion = PairwiseComparisonMatrix(car_criteria, car_alternatives)

#criteria_comparsion.calculate_criteria()

#criteria_comparsion.display_matrix(0)
#criteria_comparsion.consistency_ratio(priority_index=criteria_comparsion.find_priority_index())
# Создание пустой матрицы типа DataFrame для расчета лучшего варианта по алгоритму AHP
df_ahp = pd.DataFrame()
# Создание пустой матрицы типа DataFrame для расчета лучшего варианта по алгоритму AHP+
df_ahp_plus = pd.DataFrame()
# Создание пустой матрицы типа DataFrame для расчета веса по среднему, а потом по максимальному значению
priority_index_alt_modify = pd.DataFrame()
# Создание пустой матрицы типа DataFrame для расчета веса по среднему
priority_index_alt = pd.DataFrame()

for i in range(np.shape(car_alternatives)[1]):
    # Создание матриц парных сравнений альтернатив
    criteria_comparsion.calculate_alternatives(i)
    # Вывод МПС альтернатив
    criteria_comparsion.display_matrix(i)
    print("\nПроверка согласованности МПС альтернатив:")
    # Проверка согласованности каждой МПС альтернатив
    criteria_comparsion.consistency_ratio()
    # Расчет весов альтернатив по среднему
    priority_index_alt = criteria_comparsion.find_priority_index()
    # Расчет весов альтернатив по среднему, а потом по максимальному значению
    priority_index_alt_modify = criteria_comparsion.avoid_rank_reversal()
    # Складывание весов альтернатив, рассчитанных по среднему в матрицу
    df_ahp = pd.concat([df_ahp, priority_index_alt], ignore_index=True)
    print('\nМатрица весов альтернатив, расчитанных по среднему:\n', df_ahp)
    # Складывание весов альтернатив, рассчитанных по среднему, а потом по макс. зн. в матрицу
    df_ahp_plus = pd.concat([df_ahp_plus, priority_index_alt_modify], ignore_index=True)
    print('\nМатрица весов альтернатив, расчитанных по среднему, а потом по максимальному значению:\n', df_ahp_plus)

# Создание матрицы парных сравнений критериев
criteria_comparsion.calculate_criteria()
# Вывод МПС критериев
criteria_comparsion.display_matrix(0)
# Проверка согласованности МПС критериев
criteria_comparsion.consistency_ratio()
# Расчет приоритетов сравнимаемых критериев
priority_index_criteria: DataFrame = criteria_comparsion.find_priority_index()
# Транспонсирование ВК критериев
priority_index_criteria = priority_index_criteria.transpose()
# Транспонсирование весов альтернатив следуя AHP для удовлетворения правилу умножения матриц (6х7 и 7х1)
df_ahp = df_ahp.transpose()
# Транспонсирование весов альтернатив следуя AHP+ для удовлетворения правилу умножения матриц (6х7 и 7х1)
df_ahp_plus = df_ahp_plus.transpose()
# Расчет лучшего варианта по алгоритму AHP (умножением весов критерив и весов альтернатив)
finally_result = np.dot(df_ahp, priority_index_criteria)
# Приведение матрицы к типу DataFrame
finally_result_pd = pd.DataFrame(finally_result, index=['Hyundai Solaris', 'Kia Rio', 'Chevrolet Cruze', 'Renault Logan', 'Лада Веста', 'Volkswagen Tiguan'])
# Создание имени колонки
finally_result_pd.columns = ['Vectors']
print(f'\nРасчет лучшего варианта по алгоритму AHP:\n {finally_result_pd}')
# Расчет лучшего варианта по алгоритму AHP+ (умножением весов критериев и весов альтернатив, пересчитанных по среднему, а потом по максимальному значению)
finally_result_plus = np.dot(df_ahp_plus, priority_index_criteria)
# Приведение матрицы к типу DataFrame
finally_result_plus_pd = pd.DataFrame(finally_result_plus, index=['Hyundai Solaris', 'Kia Rio', 'Chevrolet Cruze', 'Renault Logan', 'Лада Веста', 'Volkswagen Tiguan'])
# Создание имени колонки
finally_result_plus_pd.columns = ['Vectors']
print(f'\nРасчет лучшего варианта по алгоритму AHP+:\n {finally_result_plus_pd}')

