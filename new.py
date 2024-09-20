import numpy as np
import sympy as sp
def f_a():
    # Створюємо матрицю A
    a = np.array([[4, 3, 0],
                  [2, -1, 5],
                  [1, 2, -1]])

    print("Матриця А")
    print(a)

    # Матриця А в квадраті
    print("Матриця А в квадраті")
    func = np.dot(a, a)
    print(func)

    # Матриця А в квадраті помножена на 9
    print("Матриця А в квадраті помножена на 9")
    func = func * 9
    print(func)

    # f(A) = 9A^2 - 4I, де I - одинична матриця
    print("f(A)")
    func = func - 4 * np.eye(a.shape[0])
    print(func)

    return func



def matrix_operations():
    # Матриці A, B, C
    a = np.array([[3, -2],
                  [1, -1]])

    b = np.array([[3, 2],
                  [1, 4]])

    c = np.array([[2, 1],
                  [4, -3]])
    # Виводимо матрицю A
    print("Матриця A:")
    print(a)

    # Виводимо матрицю B
    print("Матриця B:")
    print(b)

    # Виводимо матрицю C
    print("Матриця C:")
    print(c)

    # Обчислюємо добуток матриць A і B
    print("Добуток матриць A і B (A * B):")
    a_b = np.dot(a, b)
    print(a_b)

    # Обчислюємо обернену матрицю для A * B
    print("Обернена матриця до A * B:")
    inv_a_b = np.linalg.inv(a_b)
    print(inv_a_b)

    # Обчислюємо добуток оберненої матриці (A * B)^(-1) і матриці C
    print("Добуток оберненої матриці до A * B і матриці C:")
    x = np.dot(inv_a_b, c)
    print(x)

    return x

def matrix_rank():
    # Матриця A
    a = np.array([[1, 0, 1, -1, 2],
                  [-2, 1, 0, 1, -2],
                  [2, 1, 1, 2, 3],
                  [1, 2, 2, 2, 3]])
    # Виводимо матрицю A
    print("Матриця A:")
    print(a)

    # Обчислюємо ранг матриці A
    rank = np.linalg.matrix_rank(a)
    print("Ранг матриці A:")
    print(rank)

    return rank


def solve_matrix_method():
    a = np.array([[2, 1, 1],
                  [3, 3, 2],
                  [1, -1, 1]])

    b = np.array([2, 5, -3])
    """
    Розв'язок системи рівнянь матричним методом
    """
    # Виводимо матрицю коефіцієнтів A та вектор-стовпець B
    print("Матриця коефіцієнтів A:")
    print(a)
    print("Вектор B:")
    print(b)

    # Розв'язок за допомогою матричного методу (X = A^-1 * B)
    try:
        x_matrix_method = np.linalg.solve(a, b)
        print("Розв'язок матричним методом:")
        print(x_matrix_method)
    except np.linalg.LinAlgError:
        print("Матриця A є виродженою (det(A) = 0), розв'язку немає.")
        return None

    return x_matrix_method


def solve_cramer_method():
    """
    Розв'язок системи рівнянь за допомогою формул Крамера
    """
    a = np.array([[2, 1, 1],
                  [3, 3, 2],
                  [1, -1, 1]])

    b = np.array([2, 5, -3])
    # Визначник матриці A
    det_a = np.linalg.det(a)

    if det_a == 0:
        print("Матриця A є виродженою (det(A) = 0), метод Крамера не застосовується.")
        return None

    print(f"Визначник матриці A: det(A) = {det_a}")

    n = a.shape[0]  # Кількість змінних
    x_cramer_method = np.zeros(n)

    # Формули Крамера: X_i = det(A_i) / det(A)
    for i in range(n):
        a_i = a.copy()
        a_i[:, i] = b  # Замінюємо i-тий стовпець матриці A на вектор B
        det_a_i = np.linalg.det(a_i)
        x_cramer_method[i] = det_a_i / det_a

        print(f"det(A_{i}) = {det_a_i}, x_{i} = {x_cramer_method[i]}")

    print("Розв'язок за допомогою формул Крамера:")
    print(x_cramer_method)

    return x_cramer_method

def vector_dot_product(a, b):
    # Обчислюємо скалярний добуток
    return np.dot(a, b)

def calculate_expression():
    # Оголошення змінних
    a, b = sp.symbols('a b')

    # Вираз для спрощення
    expression = (a + b) * (a - 2 * b)

    # Спрощення виразу
    simplified_expression = sp.expand(expression)

    # Виведення результату
    print(simplified_expression)

def vector_magnitude_difference():
    # Довжини векторів та кут між ними
    mag_a = 3
    mag_b = 4
    angle_phi = np.pi / 3  # кут в радіанах

    # Формула для модуля різниці двох векторів
    result = np.sqrt(mag_a**2 + mag_b**2 - 2 * mag_a * mag_b * np.cos(angle_phi))
    return result

def solve_linear_system():
    A = np.array([[3, -2, 3],
                  [5, -1, 6],
                  [11, 1, 18]])

    B = np.array([11, 13, 3])
    try:
        # Розв'язок системи рівнянь (X = A^-1 * B)
        solution = np.linalg.solve(A, B)
        return solution
    except np.linalg.LinAlgError as e:
        print(f"Помилка: {e}")
        return None


def vector_subtract(P, Q):
    """
    Віднімання векторів P і Q (P - Q)
    """
    return np.array(P) - np.array(Q)


def triangle_area():
    # Координати точок 361 -500 0 -5 30
    A = [1, -3, 0]
    B = [4, 3, 1]
    C = [-4, -3, 0]
    D = [-1, -2, 3]
    """
    Обчислення площі трикутника ABC
    """
    # Вектори AB і AC
    AB = vector_subtract(B, A)
    AC = vector_subtract(C, A)

    # Векторний добуток AB і AC
    cross_product = np.cross(AB, AC)

    # Площа трикутника
    area = 0.5 * np.linalg.norm(cross_product)
    return area


def pyramid_volume():
    # Координати точок
    A = [1, -3, 0]
    B = [4, 3, 1]
    C = [-4, -3, 0]
    D = [-1, -2, 3]
    """
    Обчислення об'єму піраміди ABCD
    """
    # Вектори AB, AC і AD
    AB = vector_subtract(B, A)
    AC = vector_subtract(C, A)
    AD = vector_subtract(D, A)

    # Векторний добуток AB і AC
    cross_product = np.cross(AB, AC)

    # Скалярний добуток (AB x AC) · AD
    volume = abs(np.dot(cross_product, AD)) / 6.0
    return volume

# Викликаємо функцію f_a 2.1.16
print("\n\n\n2.1.16")
result = f_a()

# Викликаємо функцію 2.2.16
print("\n\n\n2.2.16")
result_2 = matrix_operations()

# Викликаємо функцію для обчислення рангу 2.4.16
print("\n\n\n2.4.16")
rank_a = matrix_rank()

# Розв'язок матричним методом 3.1.16 а
print("\n\n\n3.1.16 a")
x_matrix = solve_matrix_method()

# Розв'язок за допомогою формул Крамера 3.1.16 б
print("\n\n\n3.1.16 b")
x_cramer = solve_cramer_method()

# Обчислюємо вираз 5.1.16 а
print("\n\n\n5.1.16 a")
calculate_expression()


# Обчислюємо модуль різниці векторів 5.1.16 б
print("\n\n\n5.1.16 b")
result_b = vector_magnitude_difference()
print(f"Модуль різниці векторів |a - b|: {result_b}")

# 5.3.16
print("\n\n\n5.3.16")
solution = solve_linear_system()

if solution is not None:
    print("Розв'язок системи рівнянь:")
    print(f"x_1 = {solution[0]}")
    print(f"x_2 = {solution[1]}")
    print(f"x_3 = {solution[2]}")

# Обчислюємо площу трикутника ABC 6.2.16
print("\n\n\n6.2.16")
area_ABC = triangle_area()
print(f"Площа трикутника ABC: {area_ABC}")

# Обчислюємо об'єм піраміди ABCD 6.2.16
volume_ABCD = pyramid_volume()
print(f"Об'єм піраміди ABCD: {volume_ABCD}")
