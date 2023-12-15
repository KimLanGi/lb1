import numpy as np
import matplotlib.pyplot as plt

# Вхідні дані для класів X1 та X2
X1 = np.array([(1, 4), (3, 5), (4, 5), (0, 4), (2, 4), (2, 6), (0, 3), (1, 5), (3, 4), (0, 5)])
X2 = np.array([(1, 0), (1, 1), (1, 2), (3, 2), (4, 1), (2, 1), (4, 2), (3, 3), (4, 3), (2, 2)])

# Додамо стовбці до кожного з масивів
X1 = np.hstack((X1, np.ones((X1.shape[0], 1))))
X2 = np.hstack((-X2, -np.ones((X2.shape[0], 1))))

# Проведемо деякі маніпуляції з масивами
unified_array = np.vstack((X1, X2)) # Уніфіцируєм масиви

V = np.matrix(unified_array) # Створюємо матрицю з уніфіковних масивів

# Проведемо деякі маніпуляції з матрицею масивів
trans = np.transpose(V) # Транспоуємо матрицю

transToV = trans * V # Помножимо матрицю на тронспонуючу

det = np.linalg.det(transToV) # Знайдем det матриці

transToV_inverse = np.linalg.inv(transToV) # Знайдемо обернену матрицю

transToV_inverseMultiplyTrans = transToV_inverse * trans # Помножимо обернену матрицю на транспоновану матрицю

# Запишем одиничний вектор y
V_size = transToV_inverseMultiplyTrans.shape[1]
y = np.ones((V_size, 1))
w = transToV_inverseMultiplyTrans * y

# Перевіримо, що V * w > 0
if (V * w > 0).all():
    print('V * w > 0')
    print("-----------------------------------------------------------------------")
else:
    print('V * w < 0')
    print("-----------------------------------------------------------------------")
    exit()

# Запишемо результат як формулу виду d(x) = w0x1 + w1x2 + w2
print(f'd(x) = {w[0, 0]}x1 + {w[1, 0]}x2 + {w[2, 0]}')

# Побудова графіка
x_range = [-4, 4]
y_range = [(-w[0, 0] * x - w[2, 0]) / w[1, 0] for x in x_range]

# Побудова графіка для вирішувальної функції
plt.scatter(X1[:, 0], X1[:, 1], label='Клас X1', marker='o', color='blue')
plt.scatter(X2[:, 0], X2[:, 1], label='Клас X2', marker='x', color='red')
plt.plot(x_range, y_range, label='Вирішувальна функція', linestyle='--', color='green')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.title('Алгоритм Хо-Кашьяпа')

plt.show()
