'''
Тест производной через преобразование Фурье
По итогу - появляются какие то странные артефакты на концах графика
Пока что не знаю, что их вызывает
Буду пока использовать Метод конечных разностей : https://en.wikipedia.org/wiki/Finite_difference_method
Коэффициенты формул численного дифференцирования : https://en.wikipedia.org/wiki/Finite_difference_coefficient
'''

import numpy as np
import matplotlib.pyplot as plt

def fourier_derivative_2nd(f, dx):
    # Length of vector f
    nx = np.size(f)
    # Initialize k vector up to Nyquist wavenumber 
    kmax = np.pi / dx
    dk = kmax / (nx / 2)
    k = np.arange(float(nx))
    k[: int(nx/2)] = k[: int(nx/2)] * dk 
    k[int(nx/2) :] = k[: int(nx/2)] - kmax
    
    # Fourier derivative
    ff = np.fft.fft(f)
    ff = (1j*k)**0 * ff
    df_num = np.real(np.fft.ifft(ff))
    return np.array(df_num)


def finite_len_1st_derivative(f, dx):
    deriv_out = []
    deriv_out.append(0)

    for i in range(len(f)-2):
        deriv_out.append((f[i+1] - f[i])/dx)
    
    deriv_out.append(0)
    return np.array(deriv_out)


def finite_len_2nd_derivative(f, dx):
    deriv_out = []
    deriv_out.append(0)

    for i in range(len(f)-2):
        deriv_out.append((f[i+1] - 2*f[i] + f[i-1])/dx**2)
    
    deriv_out.append(0)
    return np.array(deriv_out)


FUNC1 = lambda x: np.sin(x) + x
FUNC2 = lambda x: np.e**np.sin(x)

cur_func = FUNC1

X_MIN, X_MAX = 0, 100
X_INCREM = 1
x_range = X_MAX - X_MIN
x_num_segm = int(x_range/X_INCREM)

x_span = np.linspace(0, X_MAX, x_num_segm)
f_span = cur_func(x_span)
deriv_f_fourier = fourier_derivative_2nd(f_span, X_INCREM)
deriv_f_finite_len = finite_len_2nd_derivative(f_span, X_INCREM)

error = f_span - deriv_f_fourier
print(error)



plt.plot(x_span, f_span, 'D')
plt.plot(x_span, deriv_f_fourier)
plt.plot(x_span, error)

# plt.plot(x_span, deriv_f_finite_len)
plt.show()