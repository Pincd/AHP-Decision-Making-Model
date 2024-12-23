import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.spatial.distance import squareform


class PairwiseComparisonMatrix:

    def __init__(self, arr, car_alternatives: np.ndarray):
        self.arr = arr
        self.arr2 = car_alternatives
        self.comparison_matrix = None

    def calculate_alternatives(self, column):
        # car_values = self.arr2[:, 0]
        criteria_values = (squareform(self.arr2[:, column])).astype(float)
        row, col = np.diag_indices(criteria_values.shape[0])
        criteria_values[row, col] = np.ones(criteria_values.shape[0])
        for i in range(0, len(row)):
            for j in range(0, len(col)):
                if j < i:
                    criteria_values[i, j] = (1 / criteria_values[i, j])
        self.comparison_matrix = np.asarray(criteria_values)

    def calculate_criteria(self):
        # car_values = self.arr2[:, 0]
        criteria_values = (squareform(self.arr)).astype(float)
        row, col = np.diag_indices(criteria_values.shape[0])
        criteria_values[row, col] = np.ones(criteria_values.shape[0])
        for i in range(0, len(row)):
            for j in range(0, len(col)):
                if j < i:
                    criteria_values[i, j] = (1 / criteria_values[i, j])
        self.comparison_matrix = np.asarray(criteria_values)

    def find_priority_index(self):
        arr = pd.DataFrame(self.comparison_matrix)
        sum_array = np.array(arr.sum(numeric_only=True))
        cell_by_sum = arr.div(sum_array, axis=1)
        weight = pd.DataFrame(cell_by_sum.mean(axis=1), index=arr.index, columns=['priority index'])
        weight: DataFrame = weight.transpose()
        return weight

    def avoid_rank_reversal(self):
        arr = pd.DataFrame(self.comparison_matrix)
        sum_array = np.array(arr.sum(numeric_only=True))
        cell_by_sum = arr.div(sum_array, axis=1)
        sum_normalize_arr = np.array(cell_by_sum.sum(axis=1))
        max_index = sum_normalize_arr.max()
        for i in range(len(sum_normalize_arr)):
            sum_normalize_arr[i] = sum_normalize_arr[i]/max_index
        weight: DataFrame = pd.DataFrame(sum_normalize_arr)
        weight: DataFrame = weight.transpose()
        return weight

    def consistency_ratio(self):
        arr_ahp = self.get_matrix()
        # Сумма каждой колонки матрицы
        sum_array = np.array(arr_ahp.sum(numeric_only=True))
        # Случайная согласованность
        random_matrix = {1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32,
                         8: 1.14, 9: 1.45, 10: 1.49}
        #mult_in_square = pd.DataFrame()
        # Пустая матрица
        wvector = np.array([np.float64(0)]*len(sum_array))
        # Произведение каждой строки в матрице
        mult_in_square = arr_ahp.product(axis=1)
        for i in range(len(mult_in_square)):
            # Возведение произведения каждой строки в корень для расчета ненормализированного вектора
            mult_in_square[i] = np.power(mult_in_square[i], 1/6)
            # Деление ненормализованного вектора строки на сумму колонки для расчета нормализованного вектора
            wvector[i] = mult_in_square[i]/sum_array[i]
        lambda_max = 0
        for i in range(len(wvector)):
            # Приближенное определение собственного числа МПС лямба max
            lambda_max += wvector[i]*sum_array[i]
        # Расчет индекса согласованности
        consistency_index = (lambda_max-len(arr_ahp))/(len(arr_ahp)-1)
        tmp_value = 0
        for i in range(len(random_matrix)):
            if (len(arr_ahp)==i):
                tmp_value = random_matrix[i]
        # Расчет отношения согласованности
        consistency_ratio = consistency_index/tmp_value
        print(f'\nThe Consistency Ratio is: {consistency_ratio}')
        if consistency_ratio < 0.1:
            print('The model is consistent')
        else:
            print('The model is not consistent')

    def get_matrix(self):
        if self.comparison_matrix is None:
            raise ValueError("Matrix not calculated")
        return pd.DataFrame(self.comparison_matrix)

    def display_matrix(self, attr_num):
        pcm = self.get_matrix()
        priority_index = self.find_priority_index()
        if (len(pcm)==6):
            print(f'\nМатрица попарных сравнений для критерия C{attr_num + 1}:')
            print(pcm, '\n')
            print(f'Весовой столбец критерия C{attr_num + 1}:')
            print(priority_index)
        else:
            print(f'\nМатрица попарных сравнений для критериев:')
            print(pcm, '\n')
            print(f'Весовой столбец критериев')
            print(priority_index)

