'''
Симуляция уравнения U_tt'' = c^2(U_xx'' + U_yy'') + f(x,t)
Это первая версия программы.

Для взятия производной по координатам используется преобразование Фурье
Код для этого взят с https://krischer.github.io/seismo_live_build/html/Computational%20Seismology/The%20Pseudospectral%20Method/ps_fourier_acoustic_1d.html

Для взятия производной во времени используется метод конечных разностей
В частности формула f_xx = (1 * f[i-h]  -  2 * f[i]  +  1 * f[i+h]) / (h**2), где h - шаг дискретизации координат

Дополнительный функционал:
    1. matplotlib достаточно долго рисует картинки
        Поэтому можно сохранить состояние среды как матрицу (получится много матриц, которые занимают достаточно много места на диске)
    2. recreate_animation() воссоздаст анимацию по сохраненным матрицам

Во время просмотра анимации может показаться, что программа супер медленная
Это отчасти правда. 
На самом деле основная проблема в том, что именно matplotlib больше всего занимает времени для отрисовки 
Отсюда можно выделить следующие идеи:

TODO: найти другой движок для анимаций
TODO: переписать функцию recreate_animation(), чтоб та работала напрямую с графикой, а не скидывала всю работу на matplotlib 
TODO: существуют более точные формулы производной во времени, где для предсказания состояния среду используются более старые (более 2) состояния среды
    ссылка на рассчёт коэффициентов https://web.media.mit.edu/~crtaylor/calculator.html

UPD 11.02.22: Добавил другой способ сохранения картинок
    Ещё нашел отличную статью https://www.hindawi.com/journals/tswj/2014/285945/
    По сути, это то же самое, что я делал, но ещё и с демпфером 
'''


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image 


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


class Field():
    ''' Объект среды распространения волны '''
    def __init__(self, num, dl, dt, c):
        self.num, self.dl = num, dl
        self.dt = dt
        self.values = np.zeros((num, num))
        self.c = c
        self.prev_values = self.values
        self.second_derivative_coef = [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]
    
    def set_values(self, values):
        self.values, self.prev_values = values, values
    
    def get_values(self) -> np.ndarray:
        return self.values
    
    def set_one_point(self, pos, val):
        x, y = pos
        self.values[x, y] = val

    def calc_derivative_field(self, x):
        ''' Рассчёт второй производной по координатам 
        Разбивает 2-мерную матрицу на 1-мерные массив, который передаёт функции calc_fourier_2nd_derivative
        
        Parameters
        ----------
        x: bool
            Если True, то выдаст матрицу производных в каждой точке по X
            Если False, то по Y '''

        def calc_fourier_2nd_derivative(f, dx):
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
            ff = (1j*k)**2 * ff
            df_num = np.real(np.fft.ifft(ff))
            return df_num

        output_derivative = np.zeros((self.num, self.num), dtype=np.float32)

        if x:
            for i in range(self.num):
                slice = self.values[i, :]
                slice[0], slice[-1] = 0, 0

                tmp_deriv = calc_fourier_2nd_derivative(slice, self.dl)
                for j in range(self.num):
                    output_derivative[i, j] = tmp_deriv[j]
        else:
            for i in range(self.num):
                slice = self.values[:, i]
                slice[0], slice[-1] = 0, 0

                tmp_deriv = calc_fourier_2nd_derivative(slice, self.dl)
                for j in range(self.num):
                    output_derivative[j, i] = tmp_deriv[j]
        return output_derivative
    
    def increment(self):
        ''' Функция рассчёта следующего шага симуляции '''
        next_step = self.dt**2 * self.c * (self.calc_derivative_field(x=True) + self.calc_derivative_field(x=False)) + 2 * self.values - self.prev_values
        self.prev_values = self.values
        self.values = next_step


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


class Field_simulation():
    ''' Класс симуляции '''
    def __init__(self, field_obj, num, dl, save_figs=False, homogeneous=True, redraw_iter=5):
        self.field = field_obj
        self.dl = dl

        # Массив уровней для отображения на картинке matplotlib
        self.vals = np.linspace(-0.1, 0.1, num=150)

        # Создание поля для отображения через matplotlib
        tmp_linear_field = np.linspace(0, num-1, num=num)*self.dl
        self.X, self.Y = np.meshgrid(tmp_linear_field, tmp_linear_field)
        self.t = 0

        self.save_figs = save_figs
        self.homogeneous = homogeneous
        self.num_iterations_to_redraw = redraw_iter

    def start_animation_matplotlib(self):
        ''' Функция создания анимации matplotlib '''
        self.fig, self.ax = plt.subplots(figsize=(7,7))
        self.ax.set_aspect('equal', 'box')
        # self.fig.canvas.manager.full_screen_toggle() 

        anim = animation.FuncAnimation(self.fig, self.redraw, interval=0, blit=False)
        plt.show()

    def redraw(self, t):
        ''' Функция перерисовки картинки matplotlib '''
        self.field.increment()

        # Если уравнение неоднородное, и задано уравнение источников f(x,t), то это тут
        if not self.homogeneous:
            self.field.set_one_point((int(self.field.num/2), int(self.field.num/2)), np.sin(t/20)/10)

        # Отрисовка / сохранение картинки, если прошло self.num_iterations_to_redraw итераций 
        if t % self.num_iterations_to_redraw == 0:
            self.ax.clear()
            self.ax.contourf(self.X, self.Y, self.field.get_values(), self.vals, alpha=1)
            if self.save_figs:
                self.fig.savefig(f'C:\\Users\\Alex\\Desktop\\SWave\\frames\\hello_{int(self.t):05.0f}.png')
            self.t += 1
    
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
        
    def start_simulation(self, num_iterations):
        t = 0
        while t < num_iterations:
            np.save(f'C:\\Users\\Alex\\Desktop\\SWave\\data\\matrix_{int(t):05.0f}', np.asarray(self.field.get_values()), dtype=np.float16)
            self.field.increment()
            t += 1
        exit()


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
    
    tmp_pic.save(f'C:\\Users\\Alex\\Desktop\\SWave\\frames\\png_{int(index):05.0f}.png')
    

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def recreate_animation_matplotlib(num, dl, save_figs):
    ''' Функция воссоздания анимации по сохраненным матрицам '''

    # Массив уровней для отображения на картинке matplotlib
    vals = np.linspace(-0.1, 0.1, num=150)

    # Создание поля для отображения через matplotlib
    tmp_linear_field = np.linspace(0, num-1, num=num)*dl
    X, Y = np.meshgrid(tmp_linear_field, tmp_linear_field)
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_aspect('equal', 'box')
    t = 0
    name_t = t
    while True:
        try:
            tmp_matrix = np.load(f'C:\\Users\\Alex\\Desktop\\SWave\\data\\matrix_{int(t):05.0f}.npy')
        except:
            exit()
        if t%5 == 0:
            ax.clear()
            ax.contourf(X, Y, tmp_matrix, vals, alpha=1)
            if save_figs:
                fig.savefig(f'C:\\Users\\Alex\\Desktop\\SWave\\frames\\frame_{int(t):05.0f}.png')
            name_t += 1
        t += 1


def create_parabolic_reflectors(c_matrix, P):
    c_matrix_copy = np.copy(c_matrix)
    for i in range(NUM):
        for j in range(NUM):
            # Параболические стенки
            case1 = [(j-P)**2 > 4*P*i, (j-P)**2 > -4*P*(i-2*P), (i-P)**2 > 4*P*j, (i-P)**2 > -4*P*(j-2*P)]
            if any(case1):
                c_matrix_copy[i,j] = 0
    return c_matrix_copy


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def simulation_setup(run_animate, save_figs, homogeneous):
    
    C = np.array([[5*1e-1 for i in range(NUM)] for j in range(NUM)])**2
    omega = np.zeros((NUM,NUM))

    # Создание параболических отражающих стенок
    P = int(NUM/2)
    C = create_parabolic_reflectors(C, P)

    # Создать поле
    field = Field(num=NUM, dl=dL, dt=dT, c=C)

    # Создать импульс
    field_val = np.zeros((NUM,NUM))
    field_val[150,150] = 3
    # field_val[200,20] = 3
    field.set_values(field_val[:,:])

    # Запустить симцляцию
    simulation = Field_simulation(field, NUM, dL, save_figs, homogeneous)

    if run_animate:
        # simulation.start_animation_matplotlib()
        simulation.start_animation_png_save(10000, 0.1)
    else:
        simulation.start_simulation(2000)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------

# Задать константы
NUM = 500
dL = 1
dT = 5 * 1e-1

# Воссоздание анимации по сохраненным матрицам
# recreate_animation(num=NUM, dl=dL, save_figs=False)

# Запуск симуляции с выбором режима 
simulation_setup(run_animate=True, save_figs=True, homogeneous=False)