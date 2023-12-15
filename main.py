import numpy as np
import matplotlib.pyplot as plt

# Вхідні дані для класів X1 та X2
X1 = np.array([(1, 4), (3, 5), (4, 5), (0, 4), (2, 4), (2, 6), (0, 3), (1, 5), (3, 4), (0, 5)])
X2 = np.array([(1, 0), (1, 1), (1, 2), (3, 2), (4, 1), (2, 1), (4, 2), (3, 3), (4, 3), (2, 2)])

# Об'єднуємо два класи в одну матрицю
X = np.vstack((X1, X2))

# Задаємо відомі мітки класів (-1 для першого класу і 1 для другого)
y = np.hstack((-np.ones(len(X1)), np.ones(len(X2))))

# Додаємо стовпець з одиницями до матриці X для обчислення вільного члена w3
X = np.column_stack((X, np.ones(len(X))))

# Знаходимо параметри w1, w2, w3 з використанням алгоритму Хо-Кашьяпа
w = np.linalg.inv(X.T @ X) @ X.T @ y

# Виводимо параметри вирішувальної функції
w1, w2, w3 = w[:-1], w[-1], w[1]

# Виводимо вирішувальну функцію
print(f"d(x) = {w1[0]} * x1 + {w2} * x2 + {w3} = 0")

# Побудова графіка
x_range = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
y_range = (-w3 - w2 * x_range) / w1[1]

plt.figure(figsize=(8, 6))
plt.scatter(X1[:, 0], X1[:, 1], label='Клас X1', marker='o', color='blue')
plt.scatter(X2[:, 0], X2[:, 1], label='Клас X2', marker='x', color='red')
plt.plot(x_range, y_range, label='Вирішувальна функція', linestyle='--', color='green')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.title('Дані та вирішувальна функція')
plt.grid(True)
plt.show()
