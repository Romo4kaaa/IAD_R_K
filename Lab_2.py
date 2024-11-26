import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#Етап 1
def lnY(lnA, alpha, lnK, beta, lnL):
    return lnA + alpha * lnK + beta * lnL
N = 20
K = np.sort(np.random.randint(1, N, N))
L = np.sort(np.random.randint(1, N, N))
Y = np.sort(np.random.normal(N, 0.05, N))
decimal = 3
print("K:", K)
print("L:",L)
print("Y:",np.round(Y, decimal))

figure = plt.figure()
ax = figure.add_subplot(projection="3d")

ax.scatter(K, L, Y)

ax.set_xlabel('K')
ax.set_ylabel('L')
ax.set_zlabel('Y')

plt.show()
#----------------------------------------------
#Етап 2
Y_log = np.log(Y)
K_log = np.log(K)
L_log = np.log(L)

print("Вектор Y_log:\n", np.round(Y_log , decimal))
print("\nВектор K_log:\n",np.round(K_log,decimal))
print("\nВектор L_log:\n", np.round(L_log,decimal))

# Побудова матриці H з стовпця одиниць, W та U
H = np.array((np.ones(N), K_log, L_log)).T

print("\nМатриця H:\n", np.round(H,decimal))

# Перевірка рангу матриці H
rank = np.linalg.matrix_rank(H)
minSize = min(H.shape)

print("\nРанг матриці -", rank)

if rank != minSize:
   print("\nМатриця H неповнорангова. Переробіть вектори K і L.")
#-----------------------------------------------------------
#Етап 3
X = np.linalg.inv(H.T @ H) @ H.T @ Y_log

X_lnA = X[0]
X_alpha = X[1]
X_beta = X[2]

# Oтримання оцінки A з X_lnA
X_A = np.exp(X_lnA)

print("A =", np.round(X_A,decimal))
print("alpha =", np.round(X_alpha,decimal))
print("beta =", np.round(X_beta,decimal))

# Знаходження Yalt на основі отриманих параметрів
Yalt = np.array([lnY(X[0], X[1], K_log[i], X[2], L_log[i]) for i in range(N)])

print("Вектор Yalt:\n", np.round(Yalt,decimal))

# Знаходження вибіркової дисперсії
sample_variance = np.sum((Y_log - Yalt) ** 2) / (N - X.shape[0])

print("\nВибіркова дисперсія = ", sample_variance)

# Знаходження квадратів дисперсій на основі вибіркової дисперсії
HTH = np.linalg.inv(H.T @ H)
c = np.array([HTH[i][i] for i in range(X.shape[0])])

print("\nЕлементи головної діагоналі:", np.round(c,decimal))

variance = sample_variance * c

print("\nКвадрати дисперсій:", variance)

# Обчислення параметра t
t = X / np.sqrt(variance)

print("\nt:", t)

# Знаходження табличного значення t_alpha

t_alpha = stats.t.ppf(q=1 - X[1], df=N - len(X))

print("\nt_alpha = ", np.round(t_alpha,decimal))

# Обчислення довірчих границь
T = t_alpha * np.sqrt(variance) * np.sqrt(c)

for i in range(len(T)):
   print('\nT', i, "=", np.round(X[i],decimal), "+-", np.round(T[i],4))

#Знаходимо lnY_upper
lnY_upper = np.sum(Y_log) / N

R2 = np.sum((Yalt - lnY_upper) ** 2) / np.sum((Y_log - lnY_upper) ** 2)

print("R^2 = ", np.round(R2,decimal))

Y_potentiated = Y_log ** np.e

print("Початкове значення Y:\n",np.round(Y,decimal))
print("Потенційоване значення Y:\n",np.round(Y_potentiated,decimal))