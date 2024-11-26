import numpy as np
import matplotlib.pyplot as plt

from prettytable import PrettyTable

# Дані
customers = {
    'C1': {'Age': 5, 'Number Purchase': 1, 'Last Purchase': 'Лялька'},
    'C2': {'Age': 4, 'Number Purchase': 2, 'Last Purchase': 'Машинка'},
    'C3': {'Age': 3, 'Number Purchase': 3, 'Last Purchase': 'Куртка'},
    'C4': {'Age': 2, 'Number Purchase': 4, 'Last Purchase': 'Кросівки'},
    'C5': {'Age': 1, 'Number Purchase': 5, 'Last Purchase': None}  # Невідома категорія покупки
}


# Функція для розрахунку відстані між C5 та іншим клієнтом
def calculate_distance(C5, Ck):
    return np.sqrt((C5['Age'] - Ck['Age']) ** 2 + (C5['Number Purchase'] - Ck['Number Purchase']) ** 2)


def visualize_neighbors(customers, distances):
    x = [customers[customer]['Age'] for customer in customers if customer != 'C5']
    y = [customers[customer]['Number Purchase'] for customer in customers if customer != 'C5']
    sizes = [1 / (distance + 1) * 100 for _, distance in distances]  # Задамо розмір відповідно до відстані

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=sizes, alpha=0.5)
    plt.scatter(customers['C5']['Age'], customers['C5']['Number Purchase'], color='red', marker='X', label='C5')

    for customer, distance in distances:
        plt.text(customers[customer]['Age'], customers[customer]['Number Purchase'], f'{customer}\n{distance:.2f}',
                 fontsize=8, ha='right', va='bottom')

    plt.xlabel('Age')
    plt.ylabel('Number Purchase')
    plt.title('Customers and Distances to C5')
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# Функція для прогнозу категорії покупки для C5
def predict_purchase(unknown_customer, customers):
    min_distance = float('inf')  # Встановлюємо початкове значення як нескінченність
    min_purchase = None

    # Розрахунок відстаней для C5 відносно інших клієнтів
    distances = []
    for key, value in customers.items():
        if key != 'C5':
            distance = calculate_distance(unknown_customer, value)
            distances.append((key, distance))

            if distance < min_distance:
                min_distance = distance
                min_purchase = value['Last Purchase']

    unknown_customer['Last Purchase'] = min_purchase
    unknown_customer['Distance'] = min_distance

    # Вивід таблиці з відстанями
    distances_table = PrettyTable()
    distances_table.field_names = ["Customer", "Distance to C5"]
    distances_table.add_rows([(customer, np.round(distance,2)) for customer, distance in distances])
    print("Table with distances:")
    print(distances_table)
    visualize_neighbors(customers, distances)


unknown_customer_C5 = customers['C5']
predict_purchase(unknown_customer_C5, customers)

table = PrettyTable()
table.field_names = ["Customer", "Age", "Number Purchase", "Last Purchase"]

for key, value in customers.items():
    table.add_row([key, value['Age'], value['Number Purchase'], value['Last Purchase']])

print(table)