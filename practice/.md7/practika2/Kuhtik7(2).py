
import math
import numpy as np
import matplotlib.pyplot as plt


def bisection(f, a, b, eps=1e-9, max_iter=200):
    fa, fb = f(a), f(b)

    if fa * fb > 0:
        raise ValueError(
            f"Нет смены знака на [{a}, {b}]: f(a)={fa}, f(b)={fb}"
        )

    journal = []

    for k in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        length = b - a

        journal.append({
            "iter": k,
            "a": a,
            "b": b,
            "c": c,
            "f(c)": fc,
            "length": length
        })

        if fc == 0:
            return c, journal, "f_zero"

        if length < eps:
            return c, journal, "ok"

        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

    return c, journal, "max_iter"


def print_journal(journal, title=""):
    if title:
        print("\n", title)

    for row in journal:
        print(
            "Итерация:", row["iter"],
            "a =", row["a"],
            "b =", row["b"],
            "c =", row["c"],
            "f(c) =", row["f(c)"],
            "длина =", row["length"]
        )


# Задание 1

k = 0.5

def f(x):
    return math.cos(x) - k * x


root, j, s = bisection(f, 0, 2, eps=1e-9)

print_journal(j, "Результат для k = 0.5")

print("Корень:", root)
print("f(корень):", f(root))
print("Статус:", s)
print("Количество итераций:", len(j))


# Задание 2

print("\nРезультаты для разных значений k")

for k in [0.5, 1, 2, 5]:

    def f(x):
        return math.cos(x) - k * x

    root, journal, status = bisection(f, 0, 2, eps=1e-9)

    print("k:", k)
    print("Корень:", root)
    print("Количество итераций:", len(journal))
    print("|f(корень)|:", abs(f(root)))
    print("Статус:", status)


# Задание 3

k_values = np.arange(0.1, 10.1, 0.1)
roots = []

for k in k_values:

    def f(x):
        return math.cos(x) - k * x

    root, journal, status = bisection(f, 0, 2, eps=1e-9)

    roots.append(root)


plt.plot(k_values, roots)
plt.xlabel("k")
plt.ylabel("Корень")
plt.title("Зависимость корня от параметра k")
plt.grid()
plt.show()


# Задание 4

k = 0.5

def f(x):
    return math.cos(x) - k * x


exact_root, j_exact, s_exact = bisection(
    f, 0, 2, eps=1e-14
)

eps_values = [1e-3, 1e-6, 1e-9, 1e-12]

iterations = []
errors = []

print("\nРезультаты для разной точности")

for eps in eps_values:

    root, journal, status = bisection(f, 0, 2, eps=eps)

    error = abs(root - exact_root)

    theoretical = math.ceil(math.log2(2 / eps))

    iterations.append(len(journal))
    errors.append(error)

    print("Точность:", eps)
    print("Корень:", root)
    print("Количество итераций:", len(journal))
    print("Фактическая ошибка:", error)
    print("Теоретическое количество итераций:", theoretical)


plt.plot(eps_values, iterations, marker="o")
plt.xscale("log")
plt.gca().invert_xaxis()
plt.xlabel("Точность epsilon")
plt.ylabel("Количество итераций")
plt.title("Зависимость итераций от точности")
plt.grid()
plt.show()


# Задание 5

k = 0

def f(x):
    return math.cos(x) - k * x


print("\nПроверка при k = 0")

print("f(0):", f(0))
print("f(2):", f(2))

root, journal, status = bisection(f, 0, 2, eps=1e-9)

print("Корень:", root)
print("Количество итераций:", len(journal))
print("Статус:", status)
