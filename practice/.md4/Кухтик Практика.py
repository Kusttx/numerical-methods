# compare_11.py
# 11 задач: сравнение обычного цикла Python и NumPy.

import time
import numpy as np


rng = np.random.default_rng(42)


def measure(fn, x, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(x)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


# Задача 1. Сумма массива

def loop_sum(arr):
    total = 0.0
    for x in arr:
        total += x
    return total


def numpy_sum(arr):
    return arr.sum()


print("\nЗАДАНИЕ 1. Сумма массива")
for n in (10**5, 10**6, 10**7):
    a = rng.normal(size=n)
    print(f"\nn = {n}")
    r_loop, t_loop = measure(loop_sum, a)
    r_numpy, t_numpy = measure(numpy_sum, a)
    print(f"цикл:   {t_loop * 1000:10.2f} мс")
    print(f"NumPy:  {t_numpy * 1000:10.2f} мс")
    print(f"ускорение: {t_loop / t_numpy:.1f}×")
    print(f"результаты совпадают: {np.isclose(r_loop, r_numpy)}")


# Задача 2. Среднее и стандартное отклонение

def loop_mean_std(arr):
    total = 0.0
    squares = 0.0
    for x in arr:
        total += x
        squares += x * x
    mean = total / len(arr)
    std = (squares / len(arr) - mean * mean) ** 0.5
    return mean, std


def numpy_mean_std(arr):
    return arr.mean(), arr.std()


print("\nЗАДАНИЕ 2. Среднее и стандартное отклонение")
a = rng.normal(size=1_000_000)
r_loop, t_loop = measure(loop_mean_std, a)
r_numpy, t_numpy = measure(numpy_mean_std, a)
print(f"цикл:   {t_loop * 1000:10.2f} мс")
print(f"NumPy:  {t_numpy * 1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"совпадают до 1e-12: {np.allclose(r_loop, r_numpy, atol=1e-12, rtol=0)}")
print("цикл:", r_loop)
print("NumPy:", r_numpy)


# Задача 3. Поэлементное преобразование: sin(x) + x**2

def loop_transform(arr):
    result = []
    for x in arr:
        result.append(float(np.sin(x)) + x * x)
    return result


def numpy_transform(arr):
    return np.sin(arr) + arr**2


print("\nЗАДАНИЕ 3. Поэлементное преобразование")
a = rng.normal(size=1_000_000)
r_loop, t_loop = measure(loop_transform, a)
r_numpy, t_numpy = measure(numpy_transform, a)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")
print("Список Python использует больше памяти на хранение объектов и ссылок.")
print(f"Память массива NumPy: {r_numpy.nbytes / 1024**2:.2f} МБ")
print("Выражение NumPy может создавать временные массивы.")


# Задача 4. Условная замена (clipping)

def loop_clip(arr):
    result = []
    for x in arr:
        if x < 0:
            result.append(0.0)
        elif x > 1:
            result.append(1.0)
        else:
            result.append(x)
    return result


def numpy_clip(arr):
    return np.clip(arr, 0, 1)


print("\nЗАДАНИЕ 4. Ограничение значений от 0 до 1")
a = rng.normal(size=1_000_000)
r_loop, t_loop = measure(loop_clip, a)
r_numpy, t_numpy = measure(numpy_clip, a)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")
print("изменено элементов:", np.sum((a < 0) | (a > 1)))


# Задача 5. Сумма по строкам матрицы

def loop_sum_rows(mat):
    result = [0.0] * mat.shape[0]
    for i in range(mat.shape[0]):
        total = 0.0
        for j in range(mat.shape[1]):
            total += mat[i, j]
        result[i] = total
    return result


def numpy_sum_rows(mat):
    return mat.sum(axis=1)


print("\nЗАДАНИЕ 5. Сумма по строкам")
m = rng.normal(size=(100_000, 10))
r_loop, t_loop = measure(loop_sum_rows, m)
r_numpy, t_numpy = measure(numpy_sum_rows, m)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")


# Задача 6. Среднее по столбцам и нормировка

def loop_normalize(mat):
    result = mat.copy()
    rows, columns = mat.shape
    means = [0.0] * columns

    for j in range(columns):
        total = 0.0
        for i in range(rows):
            total += mat[i, j]
        means[j] = total / rows

    for i in range(rows):
        for j in range(columns):
            result[i, j] -= means[j]
    return result


def numpy_normalize(mat):
    return mat - mat.mean(axis=0)


print("\nЗАДАНИЕ 6. Среднее по столбцам и нормировка")
m = rng.normal(size=(100_000, 10))
r_loop, t_loop = measure(loop_normalize, m)
r_numpy, t_numpy = measure(numpy_normalize, m)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")
print("средние после нормировки:", r_numpy.mean(axis=0))


# Задача 7. Поиск максимума и его индекса

def loop_max(arr):
    max_val = arr[0]
    max_idx = 0
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_val, max_idx


def numpy_max(arr):
    return arr.max(), arr.argmax()


print("\nЗАДАНИЕ 7. Максимум и его индекс")
a = rng.normal(size=1_000_000)
r_loop, t_loop = measure(loop_max, a)
r_numpy, t_numpy = measure(numpy_max, a)
print(f"цикл:   {t_loop * 1000:10.2f} мс")
print(f"NumPy:  {t_numpy * 1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {r_loop == r_numpy}")

a_nan = a.copy()
a_nan[0] = np.nan
print("np.nanmax для массива с NaN:", np.nanmax(a_nan))


# Задача 8. Фильтрация по порогу

def loop_filter(arr):
    return [x for x in arr if x > 0]


def numpy_filter(arr):
    return arr[arr > 0]


print("\nЗАДАНИЕ 8. Фильтрация положительных значений")
a = rng.normal(size=1_000_000)
r_loop, t_loop = measure(loop_filter, a)
r_numpy, t_numpy = measure(numpy_filter, a)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")
print(f"память массива NumPy: {r_numpy.nbytes / 1024**2:.2f} МБ")
print("Список Python занимает больше памяти из-за ссылок и отдельных объектов float.")


# Задача 9. Замена отрицательных значений их модулем (in-place)

def loop_abs(arr):
    for i in range(len(arr)):
        if arr[i] < 0:
            arr[i] = -arr[i]
    return arr


def numpy_abs(arr):
    np.abs(arr, out=arr)
    return arr


print("\nЗАДАНИЕ 9. Замена отрицательных значений")
a = rng.normal(size=1_000_000)
a_loop = a.copy()
a_numpy = a.copy()
r_loop, t_loop = measure(loop_abs, a_loop)
r_numpy, t_numpy = measure(numpy_abs, a_numpy)
print(f"цикл:   {t_loop * 1000:10.2f} мс")
print(f"NumPy:  {t_numpy * 1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.array_equal(r_loop, r_numpy)}")
print("оба варианта изменяют переданный массив на месте.")


# Задача 10. Комбинированная маска

def loop_range(arr):
    return [x for x in arr if -1 < x < 1]


def numpy_range(arr):
    return arr[(arr > -1) & (arr < 1)]


print("\nЗАДАНИЕ 10. Комбинированная маска")
a = rng.normal(size=1_000_000)
r_loop, t_loop = measure(loop_range, a)
r_numpy, t_numpy = measure(numpy_range, a)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")
print("В NumPy скобки вокруг сравнений обязательны.")


# Задача 11. Расстояния между точками

def loop_distances(data):
    X, Y = data
    n = X.shape[0]
    m = Y.shape[0]
    result = np.empty((n, m))

    for i in range(n):
        for j in range(m):
            dx = X[i, 0] - Y[j, 0]
            dy = X[i, 1] - Y[j, 1]
            dz = X[i, 2] - Y[j, 2]
            result[i, j] = (dx * dx + dy * dy + dz * dz) ** 0.5
    return result


def numpy_distances(data):
    X, Y = data
    return np.sqrt(((X[:, None, :] - Y[None, :, :])**2).sum(axis=2))


print("\nЗАДАНИЕ 11. Расстояния между точками")
X = rng.normal(size=(1000, 3))
Y = rng.normal(size=(1000, 3))
data = (X, Y)
r_loop, t_loop = measure(loop_distances, data)
r_numpy, t_numpy = measure(numpy_distances, data)
print(f"цикл:   {t_loop*1000:10.2f} мс")
print(f"NumPy:  {t_numpy*1000:10.2f} мс")
print(f"ускорение: {t_loop / t_numpy:.1f}×")
print(f"результаты совпадают: {np.allclose(r_loop, r_numpy)}")
print("форма матрицы расстояний:", r_numpy.shape)
print("размер временного массива (1000, 1000, 3):",
      1000 * 1000 * 3 * 8 / 1024**2, "МБ")

