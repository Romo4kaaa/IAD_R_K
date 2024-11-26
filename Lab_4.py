import numpy as np
import matplotlib.pyplot as plt

# Задаємо кількість точок у кожній множині
num_points = np.random.randint(5, 11)

# Задаємо математичні середні для кожної множини так, щоб утворити трикутник
mean1 = np.array([0, 0])
mean2 = np.array([2, 0])
mean3 = np.array([1, 2])

# Задаємо дисперсію (можна вибрати довільне значення)
variance = 1.5

# Генеруємо випадкові точки для кожної множини
set1 = np.random.normal(mean1, variance, size=(num_points, 2))
set2 = np.random.normal(mean2, variance, size=(num_points, 2))
set3 = np.random.normal(mean3, variance, size=(num_points, 2))

# Відображення точок на графіку
plt.scatter(set1[:, 0], set1[:, 1], label='Set 1')
plt.scatter(set2[:, 0], set2[:, 1], label='Set 2')
plt.scatter(set3[:, 0], set3[:, 1], label='Set 3')

# Встановлюємо відстані між математичними середніми
plt.plot([mean1[0], mean2[0]], [mean1[1], mean2[1]], color='black')
plt.plot([mean2[0], mean3[0]], [mean2[1], mean3[1]], color='black')
plt.plot([mean3[0], mean1[0]], [mean3[1], mean1[1]], color='black')

# Виводимо графік
plt.title('Random Points with Triangular Centers')
plt.legend()
plt.show()

# Початкові центри кластерів
initial_centers = np.vstack([mean1, mean2, mean3])

# Відображення точок та початкових центрів на графіку
plt.scatter(set1[:, 0], set1[:, 1], label='Set 1')
plt.scatter(set2[:, 0], set2[:, 1], label='Set 2')
plt.scatter(set3[:, 0], set3[:, 1], label='Set 3')
plt.scatter(initial_centers[:, 0], initial_centers[:, 1], c='red', marker='X', label='Initial Centers')

# Виводимо графік
plt.title('Random Points with Triangular Centers and Initial Cluster Centers')
plt.legend()
plt.show()

def k_means_clustering(data, k, initial_centers, max_iterations=100):
    if max_iterations <= 0:
        raise ValueError("Max iterations should be a positive integer.")

    if k <= 0:
        raise ValueError("Number of clusters (k) should be a positive integer.")

    if initial_centers.shape != (k, data.shape[1]):
        raise ValueError("The shape of initial_centers should be (k, n_features).")

    centers = initial_centers.copy()
    n_points, n_features = data.shape

    for step in range(max_iterations):
        # Оцінка приналежності точок до кластерів
        distances = np.zeros((n_points, k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(data - centers[i], axis=1)

        # Визначення приналежності точок до кластерів
        labels = np.argmin(distances, axis=1)

        # Перерахунок центрів кластерів
        new_centers = np.array([data[labels == j].mean(axis=0) for j in range(k)])

        # Перевірка на збіг
        if np.all(centers == new_centers):
            break

        centers = new_centers

    return labels, centers, step + 1

# Об'єднуємо всі точки в один масив
all_points = np.vstack([set1, set2, set3])

# Визначаємо кількість кластерів та початкові центри
k_clusters = 3
initial_centers = np.vstack([mean1, mean2, mean3])

# Застосовуємо k-means
cluster_labels, final_centers, num_steps = k_means_clustering(all_points, k_clusters, initial_centers)

# Відображення результатів кластеризації
for i in range(k_clusters):
    cluster_points = all_points[cluster_labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')

# Відображення центрів кластерів
plt.scatter(final_centers[:, 0], final_centers[:, 1], c='red', marker='X', label='Final Centers')

plt.title('k-means Clustering Results')
plt.legend()
plt.show()


# Генеруємо початкові центри кластерів
initial_centers_away = np.array([mean1 + 0.75 * variance * np.random.randn(2),
                            mean2 + 0.75 * variance * np.random.randn(2),
                            mean3 + 0.75 * variance * np.random.randn(2)])

results_away = k_means_clustering(all_points, k_clusters, initial_centers_away)
cluster_labels_away, final_centers_away, num_steps_away = results_away

# Відображення результатів кластеризації
for i in range(k_clusters):
    cluster_points = all_points[cluster_labels_away == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
plt.scatter(final_centers_away[:, 0], final_centers_away[:, 1], c='red', marker='X', label='Final Centers')
plt.title('k-means Clustering Results (Initial Centers Away)')
plt.legend()

# Виводимо кількість кроків
print(f"Алгоритм завершився на {num_steps_away} кроках.")


initial_centers_towards = np.array([mean1 - 0.75 * variance * np.random.randn(2),
                            mean2 - 0.75 * variance * np.random.randn(2),
                            mean3 - 0.75 * variance * np.random.randn(2)])

results_towards = k_means_clustering(all_points, k_clusters, initial_centers_towards)
cluster_labels_towards, final_centers_towards, num_steps_towards = results_towards

# Відображення результатів кластеризації
for i in range(k_clusters):
    cluster_points = all_points[cluster_labels_towards == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
plt.scatter(final_centers_towards[:, 0], final_centers_towards[:, 1], c='red', marker='X', label='Final Centers')
plt.title('k-means Clustering Results (Initial Centers Towards)')
plt.legend()

plt.show()

# Виводимо кількість кроків
print(f"Алгоритм завершився на {num_steps} кроках.")


import pandas as pd
data_comparison = {
    'Випадок': ['З трикутного розподілу','Віддалено', 'Наближено'],
    'Кількість кроків': [num_steps, num_steps_away, num_steps_towards]
}

df_comparison = pd.DataFrame(data_comparison)
df_comparison.set_index('Випадок', inplace=True)

# Виведемо таблицю порівняння
df_comparison

