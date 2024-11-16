import numpy as np
import matplotlib.pyplot as plt


def function(x0):
    x, y = x0
    return 3 * x**2 + 0.2 * x * y + 3 * y**2


def calculate_gradient(x0):
    x, y = x0
    dx = 6 * x + 0.2 * y
    dy = 0.2 * x + 6 * y
    return np.array([dx, dy])


def gradient_descent_with_fixed_step(x0, e1, e2, M, tk=1.0):
    x_vals, y_vals, z_vals = [], [], []  # для отображения графика

    # Шаг 1
    print(
        'Шаг 1:'
        f'\tx0 = {x0}\n'
        f'\te1 = {e1}; e2 = {e2}\n'
        f'\tM = {M}\n'
        f'\ttk = {tk}\n'
    )

    # Шаг 2
    k = 0
    flag = False
    xk = np.array(x0)
    print(
        'Шаг 2\n'
        f'\tk = {k}'
    )

    while True:
        # Добавляю точки в списки для отображения их на графике
        x_vals.append(xk[0])
        y_vals.append(xk[1])
        z_vals.append(function(xk))

        # Шаг 3
        derivative_k = calculate_gradient(xk)
        print(
            f'Итерация №{k+1}\n\n'
            'Шаг 3:\n'
            f'grad f(xk) = ({derivative_k[0]}, {derivative_k[1]})\n'
        )

        normF = np.linalg.norm(derivative_k)
        print(
            'Шаг 4:\n'
            f'\t||grad f(xk)|| = {normF} \n'
            f'\t Проверяем, что {normF} < {e1}'
        )

        # Шаг 4
        if normF < e1:
            print('Критерий выполнен. Конец вычеслений.')
            break
        print('Критерий не выполнен, переходим к Шагу 5.\n')

        print(
            'Шаг 5:\n'
            f'\tПроверяем {k} >= {M}'
        )

        # Шаг 5
        if k >= M:
            print('Неравентство выполнено, расчеты окончены.')
            break
        print('Неравенсство не выполнено, переходим к шагу 6\n')

        # Шаг 6
        tk = tk
        while True:

            # Шаг 7
            xk_1 = xk - tk * derivative_k
            print(
                'Шаг 7\n'
                f'\tx(k+1) = ({xk_1[0]}, {xk_1[1]})'
            )

            # Проверка условия спуска
            if function(xk_1) < function(xk):
                break
            tk /= 2  # Уменьшаем шаг, если условие не выполнено

        # Шаг 7
        norm = np.linalg.norm(xk_1 - xk)

        # Шаг 8
        print(
                '\nШаг 8:\n'
                '\t||x(k+1) - x(k)|| < e2  and |f(x(k+1) - f(xk) < e2)|'
            )
        if norm < e2 and abs(function(xk_1) - function(xk)) < e2:
            if flag:
                print(f'x* равен ({xk_1[0]}, {xk_1[1]})')
                print('Оба условия выполнены, расчеты окончены.\n')
                xk = xk_1
                break
            flag = True
            xk = xk_1

        # Шаг 8(б)
        else:
            flag = False
            print(f'Одно из условий не выполнено, кладем k = {k+1}.')
            xk = xk_1
            k += 1

    print(
        f'Ответ: xk = ({xk[0]:.4f},'
        f'{xk[1]:.4f}),'
        f'f(xk) = {function(xk):.4f}, k = {k+1}'
    )
    return x_vals, y_vals, z_vals


def graf_display_fixed_step(x0, e1, e2, M, tk=1.0):
    # Получаем траекторию градиентного спуска
    x_vals, y_vals, z_vals = gradient_descent_with_fixed_step(
        x0, e1, e2, M, tk
    )

    # Построение 3D графика функции
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Сетка значений x и y
    x_range = np.linspace(-1, 1, 100)
    y_range = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = function([X, Y])

    # Построение поверхности функции
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    # Построение траектории градиентного спуска
    ax.plot(
        x_vals, y_vals, z_vals, color='r',
        marker='o', markersize=5,
        label='Path of Gradient Descent'
    )
    ax.scatter(
        x_vals[-1], y_vals[-1], z_vals[-1],
        color='red', s=50,
        label='Minimum Found'
    )

    # Настройки графика
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(X, Y)')
    ax.set_title('Gradient Descent Path on 3D Surface with Fixed Step')
    ax.legend()

    plt.show()
