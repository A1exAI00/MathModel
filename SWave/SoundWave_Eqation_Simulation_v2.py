'''
Симуляция уравнения U_tt'' + sigma U_t' = c^2(U_xx'' + U_yy'') + f(x,t)

Для взятия производной по координатам используется преобразование Фурье
Код для этого взят с https://krischer.github.io/seismo_live_build/html/Computational%20Seismology/The%20Pseudospectral%20Method/ps_fourier_acoustic_1d.html

Для взятия производной во времени используется метод конечных разностей
В частности формула f_xx = (1 * f[i-h]  -  2 * f[i]  +  1 * f[i+h]) / (h**2), где h - шаг дискретизации координат

UPD 11.02.22: нашел отличную статью https://www.hindawi.com/journals/tswj/2014/285945/

DONE: перейти от анимации на matplotlib к сохранению картинок
DONE: добавить демпферную зону
DONE:
TODO: переписать так, чтоб f(x,t) учитывалось в методе Field.increment(), а не где то вне
TODO: вписать создание объекта Field прямо в объекте Field_simulation, а не передавать ему при создании
TODO: придумать новую цветовую политру, а то ничего не видно 
TODO: легенда с подписями силы сигнала от цвета пикселя

'''


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


import numpy as np
import os
from PIL import Image 


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


class Field():
    ''' Объект среды распространения волны '''
    def __init__(self, num, dl, dt, c, sigma):
        self.num, self.dl = num, dl
        self.dt = dt
        self.c = c
        self.sigma = sigma
        self.values = np.zeros((num, num))
        self.prev = self.values
        self.prev2 = self.values
        self.prev3 = self.values
    
    def set_values(self, values):
        self.values, self.prev = values, values
    
    def get_values(self):
        return self.values
    
    def set_one_point(self, pos, val):
        x, y = pos
        self.values[x, y] = val

    def calc_derivative_field(self, x, power):
        ''' Рассчёт второй производной по координатам 
        Разбивает 2-мерную матрицу на 1-мерные массив, который передаёт функции calc_fourier_2nd_derivative
        
        Parameters
        ----------
        x: bool
            Если True, то выдаст матрицу производных в каждой точке по X
            Если False, то по Y '''

        def calc_fourier_derivative(f, dx, power):
            # Length of vector f
            nx = len(f)
            # Initialize k vector up to Nyquist wavenumber 
            kmax = np.pi / dx
            dk = kmax / (nx / 2)
            k = np.arange(float(nx))
            k[: int(nx/2)] = k[: int(nx/2)] * dk 
            k[int(nx/2) :] = k[: int(nx/2)] - kmax
            # Fourier derivative
            ff = np.fft.fft(f)
            ff = (1j*k)**power * ff
            df_num = np.real(np.fft.ifft(ff))
            return df_num

        output_derivative = np.zeros((self.num, self.num), dtype=np.float32)
        if x:
            for i in range(self.num):
                slice = self.values[i, :]
                slice[0], slice[-1] = 0, 0

                tmp_deriv = calc_fourier_derivative(slice, self.dl, power)
                for j in range(self.num):
                    output_derivative[i, j] = tmp_deriv[j]
        else:
            for i in range(self.num):
                slice = self.values[:, i]
                slice[0], slice[-1] = 0, 0

                tmp_deriv = calc_fourier_derivative(slice, self.dl, power)
                for j in range(self.num):
                    output_derivative[j, i] = tmp_deriv[j]
        return output_derivative
    
    def increment(self, absorbing=True):
        ''' Функция рассчёта следующего шага симуляции '''
        if not absorbing:
            next_step = self.dt**2 * self.c * (self.calc_derivative_field(x=True, power=2) + self.calc_derivative_field(x=False, power=2)) + 2 * self.values - self.prev
        else:
            next_step = self.dt**2 * (self.c * (self.calc_derivative_field(x=True, power=2) + self.calc_derivative_field(x=False, power=2)) - self.sigma*(self.values - self.prev)/self.dt ) + 2 * self.values - self.prev
        self.prev, self.values = self.values, next_step


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


class Field_simulation():
    ''' Класс симуляции '''
    def __init__(self, field_obj, num, dl, homogeneous=True, redraw_iter=5):
        self.field = field_obj
        self.dl = dl

        self.homogeneous = homogeneous
        self.num_iterations_to_redraw = redraw_iter
    
    def start_animation_png_save(self, num_iterations, max_val):
        t = 0
        name_t = 0
        while t < num_iterations:
            self.field.increment()
            if not self.homogeneous:
                if t < 400:
                    self.field.set_one_point((int(self.field.num/2), int(self.field.num/2)), np.sin(t/20)/10)
            if t % self.num_iterations_to_redraw == 0:
                save_image_matrix(self.field.get_values(), max_val, name_t)
                name_t += 1
            t += 1


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def save_image_matrix(matrix, val_extr, index):
    tmp_pic = Image.new('RGB', (NUM,NUM), 'black')
    tmp_pic_colors = tmp_pic.load()
    for i in range(NUM):
        for j in range(NUM):
            current_martix_val = matrix[i,j]
            if current_martix_val > val_extr:
                tmp_pic_colors[i,j] = (255,0,0)
                continue
            if current_martix_val < -val_extr:
                tmp_pic_colors[i,j] = (0,0,255)
                continue
            bounded_val = current_martix_val/val_extr*255
            if bounded_val > 0:
                tmp_pic_colors[i,j] = (int(bounded_val),0,0)
            else:
                tmp_pic_colors[i,j] = (0, int(abs(bounded_val/3)), int(abs(bounded_val)))
    
    tmp_pic.save(os.getcwd() + f'\\frames\\png_{int(index):05.0f}.png')
    

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def create_parabolic_reflectors(c_matrix, P):
    for i in range(NUM):
        for j in range(NUM):
            # уравнения парабол
            case1 = [(j-P)**2 > 4*P*i, (j-P)**2 > -4*P*(i-2*P), (i-P)**2 > 4*P*j, (i-P)**2 > -4*P*(j-2*P)]
            if any(case1):
                c_matrix[i,j] = 0
    return c_matrix

def create_absorbing_edge(sigma):
    absorbing_value = 2e-1
    for i in range(NUM):
        for j in range(10):
            sigma[j, i] = absorbing_value
            sigma[i, j] = absorbing_value
            sigma[NUM-j-1, i] = absorbing_value
            sigma[i, NUM-j-1] = absorbing_value
    return sigma
    

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def main(homogeneous):
    #Создания поля скоростей распространения волны и коэфф затухания
    C = np.array([[5*1e-1 for i in range(NUM)] for j in range(NUM)])**2

    # Создание параболических отражающих стенок
    P = int(NUM/2)
    # C = create_parabolic_reflectors(C, P)

    # Создание коэфф затухания и поглощающих стенок
    sigma = np.zeros((NUM,NUM))
    sigma = create_absorbing_edge(sigma)

    # Создать поле
    field = Field(num=NUM, dl=dL, dt=dT, c=C, sigma=sigma)

    # Создать импульс
    field_val = np.zeros((NUM,NUM))
    field_val[20,20] = 3
    field_val[NUM-20,NUM-20] = 3
    # field_val[200,20] = 3
    field.set_values(field_val[:,:])

    # Запустить симуляцию
    simulation = Field_simulation(field, NUM, dL, homogeneous)
    simulation.start_animation_png_save(10000, 0.1)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


if __name__ == '__main__':
    # Задать константы
    NUM = 500
    dL = 1
    dT = 5 * 1e-1

    # Запуск симуляции с выбором режима (однородное или неоднородное уравнение)
    main(homogeneous=False)