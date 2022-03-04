'''
Симуляция источников синусоидального сигнала на плоскости 

DONE: Закодить класс источника сигнала
DONE: Закодить поле с источниками 
DONE: Функция с итерацией времени 
DONE: Функция с итерацией по фазе источников
TODO: Подумать, есть ли что то интересное при итерации по другим параметрам источников
DONE: Попробовать сделать белый фон

Т.к. функцию можно разложить в ряд фурье, то в одной точке могут находиться несколько источников разной частоты, и суперпозиция синусоид будет результироваться в нужной нам функции

Замечание: 
Метод get_val_at_point() сам по себе работает быстро, но его очень часто вызывают, и поэтому стоит подумать над оптимизацией 
'''


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


import numpy as np
from PIL import Image
import os
import time


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


class SineWaveSource():
    def __init__(self, pos, w_freq, amplitude, phase, wave_speed):
        self.pos, self.w_freq, self.amplitude, self.phase, self.wave_speed = pos, w_freq, amplitude, phase, wave_speed
        self.wave_func = lambda r, t: self.amplitude/(r+1) * np.cos(r - self.wave_speed*t + self.phase) if r - self.wave_speed*t < 0 else 0

    def get_val_at_point(self, pos, t):
        cur_x, cur_y = pos
        self_x, self_y = self.pos
        r = np.sqrt((self_x-cur_x)**2 + (self_y-cur_y)**2)
        return self.wave_func(r, t)


class SourcesField():
    def __init__(self, pos_map, w_freq_map, amplitude_map, phase_map, wave_speed_map):
        self.sources_list = []
        self.n_sources = len(pos_map)
        for i in range(self.n_sources):
            self.sources_list.append(SineWaveSource(pos_map[i], w_freq_map[i], amplitude_map[i], phase_map[i], wave_speed_map[i]))
        
    def get_tot_val_at_point(self, pos, t):
        tot_val = sum([source.get_val_at_point(pos, t) for source in self.sources_list])
        return tot_val
    
    def get_val_matrix(self, t):
        matrix = np.zeros((WIDTH, HEIGHT))
        for i in range(WIDTH):
            for j in range(HEIGHT):
                cur_pos = (dx_global*i - WIDTH/2, dy_global*j - HEIGHT/2)
                matrix[i,j] = self.get_tot_val_at_point(cur_pos, t)
        return matrix


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def save_picture(matrix, index=0, white_background=False):
    ''' Функция сохранения матрицы matrix как картинки 
    
    Parameters
    ----------
    matrix: list of lists of strings
    index: int - индекс, с которым сохранится картинка
    time_name: bool
        Если True, то вместо индекса будет время в секундах с начала Эпохи
        Если False, то будет сохранена с индексом
    '''
    tmp_pic = Image.new('RGB', (WIDTH, HEIGHT), 'white')
    tmp_pic_colors = tmp_pic.load()
    for i in range(WIDTH):
        for j in range(HEIGHT):
            cur_val = matrix[i,j]
            if cur_val > MAX_AMPLITUDE:
                tmp_pic_colors[i,j] = (255,0,0)
                continue
            if cur_val < -MAX_AMPLITUDE:
                tmp_pic_colors[i,j] = (0,0,255)
                continue
            bound_cur_val = int(cur_val/MAX_AMPLITUDE*255)
            if white_background:
                if cur_val > 0:
                    # tmp_pic_colors[i,j] = (bound_cur_val,0,0)
                    tmp_pic_colors[i,j] = (255,255-bound_cur_val,255-bound_cur_val)
                else:
                    # tmp_pic_colors[i,j] = (0,0,abs(bound_cur_val))
                    tmp_pic_colors[i,j] = (255-abs(bound_cur_val),255-abs(bound_cur_val), 255)
            else:
                if cur_val > 0:
                    tmp_pic_colors[i,j] = (bound_cur_val,0,0)
                else:
                    tmp_pic_colors[i,j] = (0,0,abs(bound_cur_val))



    tmp_pic.save(os.getcwd() + f'\\pics1\\wave_{int(index):05.0f}.png')


def iterate_time():
    pos_map = [(5*i, 0) for i in range(-5, 6)]
    w_freq_map = [1/10 for _ in range(len(pos_map))]
    amplitude_map = [5 for _ in range(len(pos_map))]
    phase_map = np.linspace(0, 2*np.pi, num=len(pos_map))
    wave_speed_map = [5 for _ in range(len(pos_map))]

    sources_obj = SourcesField(pos_map, w_freq_map, amplitude_map, phase_map, wave_speed_map)

    for i in range(10):
        save_picture(sources_obj.get_val_matrix(i*dt_global), index=i, white_background=True)


def iterate_phase():
    i = 0
    for phase_multiplier in np.linspace(-2, 2, num=100):
        pos_map = [(5*i, 0) for i in range(-5, 6)]
        w_freq_map = [1/10 for _ in range(len(pos_map))]
        amplitude_map = [5 for _ in range(len(pos_map))]
        phase_map = np.linspace(0, 2*np.pi, num=len(pos_map)) * phase_multiplier
        wave_speed_map = [5 for _ in range(len(pos_map))]

        sources_obj = SourcesField(pos_map, w_freq_map, amplitude_map, phase_map, wave_speed_map)

        save_picture(sources_obj.get_val_matrix(100), index=i, white_background=True)
        i += 1


def my_cos(x):
    x = (x) % 2*np.pi 
    ans = 1 - x**2/2 + x**4/24 - x**6/720 + x**8/40320 - x**10/3628800 + x**12/479001600 - x**14/87178291200 + x**16/20922789888000 - x**18/6402373705728000
    return ans



# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


if __name__ == '__main__':
    st_time = time.perf_counter()

    WIDTH, HEIGHT = 500, 500
    dx_global, dy_global = 1, 1
    dt_global = 0.5

    MAX_AMPLITUDE = 1

    # iterate_time()

    iterate_phase()

    print(time.perf_counter() - st_time())