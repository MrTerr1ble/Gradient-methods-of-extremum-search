from contextlib import redirect_stdout
from gradient_method_with_a_constant_step import (
    gradient_descent_with_constant_step, graf_display
)
from the_fastest_gradient_descent import (
    gradient_descent_with_fixed_step, graf_display_fixed_step
)


def main():
    x0 = (0, 0.5)
    e1 = 0.15
    e2 = 0.2
    M = 10
    with open('output.txt', 'w', encoding='utf-8') as f, redirect_stdout(f):
        # print('Метод градиентного спуска с постоянным шагом')
        # gradient_descent_with_constant_step(x0, e1, e2, M)
        print('\n\nМетод наискорейшего градиентного спуска.')
        gradient_descent_with_fixed_step(x0, e1, e2, M)
    # graf_display(x0, e1, e2, M)
    graf_display_fixed_step(x0, e1, e2, M)


if __name__ == '__main__':
    main()
