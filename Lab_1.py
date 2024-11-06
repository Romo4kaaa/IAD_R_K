import pandas as pd
from IPython.display import display
import random
import numpy as np
import matplotlib.pyplot as plt
#Етап 1

N = 30

# Генеруєм випадкову послідовність 0-99
x_prob = [random.randint(0, 99) for _ in range(N)]
y_prob = [random.choice([0,99]) for _ in range(N)]

print("Випадкова послідовність 1:", x_prob)
print("Випадкова послідовність 2:", y_prob)

# ---------------------------
#Етап 2
def Generacia_poslidovnosti(N:int, a0:float, d:float) -> list[float]:
    return [a0 + d * i for i in range(N)]

Xdetup = Generacia_poslidovnosti(N, np.random.randint(-N, N), np.random.randint(1, N))
Ydetup = Generacia_poslidovnosti(N, np.random.randint(-N, N), np.random.randint(1, N))
Xdetdown = Generacia_poslidovnosti(N, np.random.randint(-N, N), np.random.randint(-N, 0))
Ydetdown = Generacia_poslidovnosti(N, np.random.randint(-N, N), np.random.randint(-N, 0))
print("Xdetup = ", Xdetup)
print("Ydetup = ", Ydetup)
print("Xdetdown = ", Xdetdown)
print("Ydetdown = ", Ydetdown)

# ------------------------------------------------
#Етап 3
x_stoch1 = np.add(x_prob, Xdetup)
x_stoch2 = np.add(x_prob, Xdetdown)
y_stoch1 = np.add(y_prob, Ydetup)
y_stoch2 = np.subtract(y_prob, Ydetdown)

print("x_stoch1 = ", x_stoch1.tolist())
print("x_stoch2 = ", x_stoch2.tolist())
print("y_stoch1 = ", y_stoch1.tolist())
print("y_stoch2 = ", y_stoch2.tolist())

#Графіки x_prob та y_prob
plt.plot(x_prob, y_prob, 'go')
plt.xlabel('x_prob')
plt.ylabel('y_prob')
plt.title('Графік для x_prob та y_prob')
plt.show()

#Графіки x_stoch1 та y_stoch1
plt.plot(x_stoch1, y_stoch1, 'bo')
plt.xlabel('x_stoch1')
plt.ylabel('y_stoch1')
plt.title('Графік для x_stoch1 та y_stoch1')
plt.show()

#Графіки x_stoch2 та y_stoch2
plt.plot(x_stoch2, y_stoch2, 'ro')
plt.xlabel('x_stoch2 (від\'ємна кореляція)')
plt.ylabel('y_stoch2 (від\'ємна кореляція)')
plt.title('Графік для x_stoch2 та y_stoch2')
plt.show()

# --------------------------------------------------------
#Етап 4
def calc(X: list[float], Y: list[float]):
    X_v = np.sum(X) / N
    Y_v = np.sum(Y) / N
    M_XY = np.sum([(X[i] - X_v) * (Y[i] - Y_v) for i in range(N)]) / N
    D_X = np.sum([(X[i] - X_v) ** 2 for i in range(N)]) / N
    D_Y = np.sum([(Y[i] - Y_v) ** 2 for i in range(N)]) / N
    R_XY = M_XY / np.sqrt(D_X * D_Y)

    return {
        "X_v": np.round(X_v , 2),
        "Y_v": np.round(Y_v ,2),
        "M_XY": np.round(M_XY , 2),
        "D_X": np.round(D_X ,2),
        "D_Y": np.round(D_Y,2),
        "R_XY": np.round(R_XY,2)
    }

# Обчислення кореляції для всіх пар
result = [
    calc(x_prob, y_prob),
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
#-------------------------------------------------------------------
#Етап 5
# Вивід коефіцієнтів кореляції
for i in range(len(result)):
    print(i + 1, '. ', result[i]['R_XY'])
