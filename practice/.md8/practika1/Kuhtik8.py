
import math
import numpy as np
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
            "step": abs(x_new - x)
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
            "step": abs(x_new - x_curr)
        })

        if abs(x_new - x_curr) < eps or abs(f_curr) < eps:
            return x_new, journal, "ok"

        x_prev, f_prev = x_curr, f_curr
        x_curr = x_new
        f_curr = f(x_curr)

    return x_curr, journal, "max_iter"


def print_journal_newton(journal, title=""):
    if title:
        print("\n", title)

    for row in journal:
        print(
            "Итерация:", row["iter"],
            "x =", row["x"],
            "f(x) =", row["f(x)"],
            "f'(x) =", row["f'(x)"],
            "шаг =", row["step"]
        )


def print_journal_secant(journal, title=""):
    if title:
        print("\n", title)

    for row in journal:
        print(
            "Итерация:", row["iter"],
            "x_prev =", row["x_prev"],
            "x_curr =", row["x_curr"],
            "f_curr =", row["f_curr"],
            "шаг =", row["step"]
        )


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


f = lambda x: x**3 - 2*x + 2
df = lambda x: 3*x**2 - 2

eps = 1e-12

x = np.linspace(-3, 3, 400)

plt.plot(x, f(x))
plt.axhline(0, color="black")
plt.grid()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("График функции x^3 - 2x + 2")
plt.show()


root_n0, jn0, status_n0 = newton(f, df, x0=0, eps=eps)

print_journal_newton(jn0, "Ньютон: x0 = 0")
print("Корень:", root_n0)
print("Итераций:", len(jn0))
print("Статус:", status_n0)


root_n, jn, status_n = newton(f, df, x0=-2, eps=eps)

print_journal_newton(jn, "Ньютон: x0 = -2")
print("Корень:", root_n)
print("Итераций:", len(jn))
print("Статус:", status_n)


root_s, js, status_s = secant(f, x0=-2, x1=-1, eps=eps)

print_journal_secant(js, "Метод секущих: x0 = -2, x1 = -1")
print("Корень:", root_s)
print("Итераций:", len(js))
print("Статус:", status_s)


root_b, n_b = bisection(f, -2, -1, eps=eps)

print("\nМетод половинного деления")
print("Корень:", root_b)
print("Итераций:", n_b)


print("\nСравнение методов")

print("Ньютон, x0 = 0:", root_n0, "итераций:", len(jn0))
print("Ньютон, x0 = -2:", root_n, "итераций:", len(jn))
print("Секущие:", root_s, "итераций:", len(js))
print("Бисекция:", root_b, "итераций:", n_b)

