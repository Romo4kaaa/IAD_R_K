import pandas as pd
from IPython.display import display
import random
import numpy as np
import matplotlib.pyplot as plt

N = 30

# Генеруємо першу випадкову послідовність з чисел в діапазоні від 0 до 99
x_prob = [random.randint(0, 99) for _ in range(N)]
y_prob = [random.choice([0, 99]) for _ in range(N)]

print("Перша випадкова послідовність:", x_prob)
print("Друга випадкова послідовність:", y_prob)

# ---------------------------

def generateSequence(N:int, a0:float, d:float) -> list[float]:
    return [a0 + d * i for i in range(N)]

Xdetup = generateSequence(N, np.random.randint(-N, N), np.random.randint(1, N))
Ydetup = generateSequence(N, np.random.randint(-N, N), np.random.randint(1, N))

Xdetdown = generateSequence(N, np.random.randint(-N, N), np.random.randint(-N, 0))
Ydetdown = generateSequence(N, np.random.randint(-N, N), np.random.randint(-N, 0))

print("Xdetup = ", Xdetup)
print("Ydetup = ", Ydetup)
print("Xdetdown = ", Xdetdown)
print("Ydetdown = ", Ydetdown)

# ------------------------------------------------

x_stoch1 = np.add(x_prob, Xdetup)
x_stoch2 = np.add(x_prob, Xdetdown)

# Зміна: Робимо віднімання для y_stoch2 для від’ємної кореляції
y_stoch1 = np.add(y_prob, Ydetup)
y_stoch2 = np.subtract(y_prob, Ydetdown)

print("x_stoch1 = ", x_stoch1.tolist())
print("x_stoch2 (від'ємна кореляція) = ", x_stoch2.tolist())
print("y_stoch1 = ", y_stoch1.tolist())
print("y_stoch2 (від'ємна кореляція) = ", y_stoch2.tolist())

# Побудова графіків для x_prob та y_prob
plt.plot(x_prob, y_prob, 'go')  # зелені точки для x_prob та y_prob
plt.xlabel('x_prob')
plt.ylabel('y_prob')
plt.title('Графік для x_prob та y_prob')
plt.show()

# Побудова графіків для x_stoch1 та y_stoch1
plt.plot(x_stoch1, y_stoch1, 'bo')  # сині точки для x_stoch1 та y_stoch1
plt.xlabel('x_stoch1')
plt.ylabel('y_stoch1')
plt.title('Графік для x_stoch1 та y_stoch1')
plt.show()

# Побудова графіків для x_stoch2 та y_stoch2
plt.plot(x_stoch2, y_stoch2, 'ro')  # червоні точки для x_stoch2 та y_stoch2
plt.xlabel('x_stoch2 (від\'ємна кореляція)')
plt.ylabel('y_stoch2 (від\'ємна кореляція)')
plt.title('Графік для x_stoch2 та y_stoch2')
plt.show()

# --------------------------------------------------------

def calc(X: list[float], Y: list[float]):
    X_v = np.sum(X) / N
    Y_v = np.sum(Y) / N

    M_XY = np.sum([(X[i] - X_v) * (Y[i] - Y_v) for i in range(N)]) / N

    D_X = np.sum([(X[i] - X_v) ** 2 for i in range(N)]) / N
    D_Y = np.sum([(Y[i] - Y_v) ** 2 for i in range(N)]) / N

    R_XY = M_XY / np.sqrt(D_X * D_Y)

    return {
        "X_v": X_v,
        "Y_v": Y_v,
        "M_XY": M_XY,
        "D_X": D_X,
        "D_Y": D_Y,
        "R_XY": R_XY
    }

# Обчислення кореляції для всіх пар
result = [
    calc(x_prob, y_prob),
    calc(x_prob, x_stoch1),
    calc(x_prob, x_stoch2),
    calc(y_prob, y_stoch1),
    calc(y_prob, y_stoch2),
    calc(x_stoch1, y_stoch1),
    calc(x_stoch2, y_stoch2)
]

# Вивід результатів в таблиці
display(
    pd.DataFrame(
        result,
        index=range(1, len(result) + 1)
    )
)

# Вивід коефіцієнтів кореляції
for i in range(len(result)):
    print(i + 1, '. ', result[i]['R_XY'])
