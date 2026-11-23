
import math
import matplotlib.pyplot as plt


def newton(f, df, x0, eps=1e-12, max_iter=100):
    journal = []
    x = x0
    for k in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if dfx == 0:
            return x, journal, "zero_derivative"
        x_new = x - fx / dfx
        journal.append({
            "iter": k,
            "x": x,
            "f(x)": fx,
            "f'(x)": dfx,
            "x_new": x_new,
            "step": abs(x_new - x),
        })
        if abs(x_new - x) < eps or abs(fx) < eps:
            return x_new, journal, "ok"
        x = x_new
    return x, journal, "max_iter"


def secant(f, x0, x1, eps=1e-12, max_iter=100):
    journal = []
    x_prev, x_curr = x0, x1
    f_prev, f_curr = f(x_prev), f(x_curr)

    for k in range(1, max_iter + 1):
        if f_curr - f_prev == 0:
            return x_curr, journal, "zero_denominator"

        x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)

        journal.append({
            "iter": k,
            "x_prev": x_prev,
            "x_curr": x_curr,
            "f_curr": f_curr,
            "x_new": x_new,
            "step": abs(x_new - x_curr),
        })

        if abs(x_new - x_curr) < eps or abs(f_curr) < eps:
            return x_new, journal, "ok"

        x_prev, f_prev = x_curr, f_curr
        x_curr = x_new
        f_curr = f(x_curr)

    return x_curr, journal, "max_iter"


def bisection(f, a, b, eps=1e-12, max_iter=200):
    fa, fb = f(a), f(b)

    if fa * fb > 0:
        raise ValueError("Нет смены знака")

    for k in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)

        if fc == 0 or (b - a) < eps:
            return c, k

        if fa * fc < 0:
            b = c
        else:
            a = c
            fa = fc

    return c, k


def print_journal_newton(journal, title=""):
    if title:
        print(f"\n=== {title} ===")

    print(" k |       x |          f(x) |         f'(x) |        шаг")
    print("-" * 70)

    for row in journal:
        print(
            row["iter"],
            "x =", row["x"],
            "f(x) =", row["f(x)"],
            "f'(x) =", row["f'(x)"],
            "шаг =", row["step"]
        )


def print_journal_secant(journal, title=""):
    if title:
        print(f"\n=== {title} ===")

    print(" k | x_prev | x_curr | f_curr | шаг")
    print("-" * 70)

    for row in journal:
        print(
            row["iter"],
            "x_prev =", row["x_prev"],
            "x_curr =", row["x_curr"],
            "f_curr =", row["f_curr"],
            "шаг =", row["step"]
        )


print("Проверка при k = 1")

k = 1
f = lambda x: math.exp(x) - k * x
df = lambda x: math.exp(x) - k

print("Минимум функции:", f(0))
print("Действительных корней нет")


print("\nПроверка при k = e")

k = math.e
f = lambda x: math.exp(x) - k * x
df = lambda x: math.exp(x) - k

print("f(1):", f(1))
print("f'(1):", df(1))

root, journal, status = newton(f, df, x0=1)

print("Ньютон:")
print("Корень:", root)
print("Итераций:", len(journal))
print("Статус:", status)

print("Бисекция на [0, 2]:")
print("f(0):", f(0))
print("f(2):", f(2))
print("Смены знака нет")


for k in [3, 5]:
    f = lambda x, k=k: math.exp(x) - k * x
    df = lambda x, k=k: math.exp(x) - k

    if k == 3:
        newton_starts = [0.5, 2.0]
        secant_starts = [(0, 1), (1, 2)]
        intervals = [(0, 1), (1, 2)]
    else:
        newton_starts = [0.2, 2.5]
        secant_starts = [(0, 1), (2, 3)]
        intervals = [(0, 1), (2, 3)]

    print("\n\nРЕЗУЛЬТАТЫ ДЛЯ k =", k)

    for i, x0 in enumerate(newton_starts, 1):
        root_n, jn, status_n = newton(f, df, x0=x0)

        print_journal_newton(
            jn,
            f"Ньютон, корень №{i}, x0 = {x0}"
        )

        print("Корень:", root_n)
        print("Итераций:", len(jn))
        print("Статус:", status_n)

    for i, (x0, x1) in enumerate(secant_starts, 1):
        root_s, js, status_s = secant(f, x0=x0, x1=x1)

        print_journal_secant(
            js,
            f"Секущие, корень №{i}, x0 = {x0}, x1 = {x1}"
        )

        print("Корень:", root_s)
        print("Итераций:", len(js))
        print("Статус:", status_s)

    for i, (a, b) in enumerate(intervals, 1):
        root_b, n_b = bisection(f, a, b)

        print("\nБисекция, корень №", i)
        print("Интервал:", a, b)
        print("Корень:", root_b)
        print("Итераций:", n_b)


print("\nГрафик зависимости корней от k")

k_values = []
roots_1 = []
roots_2 = []

current_k = math.e

while current_k <= 10:
    k = current_k
    f = lambda x, k=k: math.exp(x) - k * x

    if abs(k - math.e) < 1e-12:
        root_1 = 1
        root_2 = 1
    else:
        root_1, _ = bisection(f, 0, 1, eps=1e-10)
        root_2, _ = bisection(f, 1, 5, eps=1e-10)

    k_values.append(k)
    roots_1.append(root_1)
    roots_2.append(root_2)

    current_k += 0.1


plt.figure(figsize=(10, 6))
plt.plot(k_values, roots_1, label="Первый корень")
plt.plot(k_values, roots_2, label="Второй корень")
plt.xlabel("k")
plt.ylabel("Корень")
plt.title("Зависимость корней от параметра k")
plt.grid(True)
plt.legend()
plt.show()