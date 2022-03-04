'''
Скрипт, симулирующий отражение луча от различных линий - окружностей, правильных многоугольников и тд.

DONE: отрисовка начальных фигур
DONE: рассчётная формула вектора отражения
DONE: перейти от коэфф k и b к вектору k_vect = (s_x, s_y) и нач точки p_start = (p_x, p_y)
DONE: починить - игнорирует окружности 
DONE: учесть что надо учитывать только одно направление прямой (вдоль, а не против вектора k_vect)
DONE: ввести возможность ручного выбора положения окр через circle_map
TODO: ввести возможность ручного выбора положения многоугольников через polygon_map
TODO: ускорить функцию draw_circle
DONE: Окружить поле квадратом с вектором нормали (0,0) и ловить пересечение с таким вектором - отрисовать луч, идущий за край картинки и break
DONE: написать "документацию"

Возможные тудушки-улучшения:
TODO: GUI для выбора положения объектов и начала луча
'''


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


import numpy as np
from PIL import Image 
import os
import time
import copy


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


class Circle():
    def __init__(self, pos, R, special=False) -> None:
        '''
        Parameters
        ----------
        pos: tuple
            pos = (x_c, y_c) - координаты центра окружности
        R: int\float
            Радиус окружности
        '''
        self.x, self.y = pos
        self.pos = pos
        self.R = R
        self.special = special
    
    def get_pos(self):
        ''' Getter для координат центра окружности (x_c, y_c) '''
        return self.pos
    
    def get_r(self):
        ''' Getter для радиуса окружности R '''
        return self.R
    
    def calc_intersec_points(self, line_info):
        ''' Метод рассчёта пересечений прямой с данной окружностью 
        
        Parameters
        ----------
        line_info: tuple
            line_info = (k, b) - коэффициенты в уравнении прямой y = k*x + b

        Reterns
        -------
        intersec_points: tuple of point_info
            point_info = (intersec_point_coords, normal_vect)
            intersec_point_coords - координаты точки пересечения прямой и окружности
            normal_vect - вектор нормали в точке пересечения 
        '''
        x_c, y_c = self.pos
        k, b = line_info
        
        # Решение системы уравнений прямой и окружности
        # Не смотря на то, что выражение под корнем м.б. < 0, numpy не выдаст ошибку, и результатом будет переменная типа np.nan
        x_1 = (-k*b+k*y_c+x_c + np.sqrt(k**2*self.R**2-k**2*x_c**2-2*k*x_c*b+2*k*x_c*y_c+2*b*y_c+self.R**2-b**2-y_c**2)) / (k**2+1)
        x_2 = (-k*b+k*y_c+x_c - np.sqrt(k**2*self.R**2-k**2*x_c**2-2*k*x_c*b+2*k*x_c*y_c+2*b*y_c+self.R**2-b**2-y_c**2)) / (k**2+1)

        intersec_points = []
        if not np.isnan(x_1):
            y_1 = k*x_1 + b
            point_info = ((x_1, y_1), self.calc_normal_vect(point=(x_1, y_1)))
            intersec_points.append(point_info)
        if not np.isnan(x_2):
            y_2 = k*x_2 + b
            point_info = ((x_2, y_2), self.calc_normal_vect(point=(x_2, y_2)))
            intersec_points.append(point_info)
        return tuple(intersec_points)
    
    def calc_normal_vect(self, point):
        ''' Рассчёт вектора нормали в точке point '''
        if self.special:
            return (0,0)
        x, y = point
        x_c, y_c = self.pos
        return ((x-x_c)/self.R, (y-y_c)/self.R)

    def calc_tangent_vect(self, point):
        ''' Рассчёт вектора касательной в точке point '''
        if self.special:
            return (0,0)
        x, y = point
        x_c, y_c = self.pos
        return ((y-y_c)/self.R, -(x-x_c)/self.R)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


class Regular_polygon():
    def __init__(self, pos, R, N_sides, angle_shift_rad=0, special=False) -> None:
        self.pos = pos
        self.R, self.N_sides = R, N_sides
        self.angle_shift_rad = angle_shift_rad
        self.internal_angle = 2*np.pi/N_sides
        self.special = special

        self.side_len = self.R * 2 * np.sin(np.pi/N_sides)

        self.vertices_points = self.calc_polygon_vertices()
        self.sides_lines = self.calc_sides_lines()
    
    def get_polygon_vertices(self):
        ''' Getter координат вершин многоугольника '''
        return self.vertices_points
    
    def calc_polygon_vertices(self):
        ''' Метод рассчёта координат вершин многоугольника '''
        x_c, y_c = self.pos
        vertices_points = []
        for i in range(self.N_sides):
            tmp_x = float(self.R*np.cos(self.internal_angle*i + self.angle_shift_rad) + x_c)
            tmp_y = float(self.R*np.sin(self.internal_angle*i + self.angle_shift_rad) + y_c)
            vertices_points.append((tmp_x, tmp_y))
        return tuple(vertices_points)
    
    def calc_sides_lines(self):
        ''' Метод рассчёта пар точек, образующих сторону многоугольника, и нормальный вектор к ней '''
        sides_info = []
        for i in range(-1, len(self.vertices_points)-1):
            tmp_first_point, tmp_second_point = self.vertices_points[i], self.vertices_points[i+1]
            x_1, y_1 = tmp_first_point
            x_2, y_2 = tmp_second_point
            if self.special:
                tangent_vect = (0,0)
            else:
                tangent_vect = ((x_2-x_1)/self.side_len, (y_2-y_1)/self.side_len)
            normal_vect = (-tangent_vect[1], tangent_vect[0])
            sides_info.append((tmp_first_point, tmp_second_point, normal_vect))
        return sides_info
    
    def calc_intersec_points(self, line_info):
        ''' Метод нахождения точек пересечения прямой с многоугольником 
        
        Parameters
        ----------
        line_info: tuple
            line_info = (k, b) - коэффициенты в уравнении прямой y = k*x + b
        '''
        k, b = line_info
        intersec_points = []
        for i in range(len(self.sides_lines)):
            p_1, p_2, normal_vect = self.sides_lines[i]
            x_1, y_1 = p_1
            x_2, y_2 = p_2
            x_intersec = (-x_1*b+x_1*y_2+x_2*b-x_2*y_1)/(k*x_1-k*x_2-y_1+y_2)
            y_intersec = k*x_intersec + b

            if x_intersec <= max(x_1, x_2) and x_intersec >= min(x_1,x_2):
                if y_intersec <= max(y_1, y_2) and y_intersec >= min(y_1,y_2): 
                    intersec_points.append(((x_intersec, y_intersec), normal_vect))
        return intersec_points


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def draw_circle(circle_object, matrix):
    ''' Функция отрисовки окружности circle_object на matrix 
    Пробегает все точки матрицы и проверяет, удволетворяет ли какая то из них уравнению кольца с толщиной ring_width

    Можно ускорить, заранее ограничив квадрат, в котором изменяются индексы i и j

    Parameters
    ----------
    circle_object: объект класса Circle
    matrix: list of lists of strings
    
    Returns
    -------
    mutated matrix
    '''
    ring_width = 1
    x_c, y_c = circle_object.get_pos()
    R = circle_object.get_r()
    R_2 = R**2
    R_dR_2 = (R+ring_width)**2
    for i in range(WIDTH):
        for j in range(HEIGHT):
            if abs(i-x_c) > R*1.3:
                continue
            if abs(j-y_c) > R*1.3:
                continue
            if not (i-x_c)**2 + (j-y_c)**2 > R_2:
                continue
            if not (i-x_c)**2 + (j-y_c)**2 < R_dR_2:
                continue
            matrix[i][j] = 'black'
    return matrix


def draw_polygon(polygon_object, matrix):
    ''' Функция отрисовки многоугольника polygon_object на matrix 
    Для каждой пары точек многоугольника вызывает функцию draw_line()
    
    Parameters
    ----------
    polygon_object: объект класса Regular_polygon
    matrix: list of lists of strings
    
    Returns
    -------
    mutated matrix
    '''
    vertices = polygon_object.get_polygon_vertices()
    for i in range(-1, len(vertices)-1):
        draw_line(vertices[i], vertices[i+1], matrix)
    return matrix


def draw_line(p1, p2, matrix, color='black'):
    ''' Функция отрисовки отрезка по двум точкам p1 и p2 в цвете color на матрице matrix '''
    x1, y1 = p1
    x2, y2 = p2
    x1, x2 = int(x1), int(x2)
    y1, y2 = int(y1), int(y2)
    dx = x2 - x1
    dy = y2 - y1

    # Если угол меньше 45 градусов
    if abs(dx) > abs(dy):
        # Если dx > 0:
        if x1 < x2:
            for x in range(x1, x2):
                y = round(y1 + dy*(x-x1)/dx)
                case = (x >= WIDTH, x < 0, y >= HEIGHT, y < 0)
                if not any(case):
                    matrix[x][y] = color
        # Если dx <= 0:
        else:
            for x in range(x2, x1):
                y = round(y1 + dy*(x-x1)/dx)
                case = (x >= WIDTH, x < 0, y >= HEIGHT, y < 0)
                if not any(case):
                    matrix[x][y] = color
    # Если угол больше 45 градусов
    else:
        # Если dy > 0:
        if y1 < y2:
            for y in range(y1, y2):
                x = round(x1 + dx*(y-y1)/dy)
                case = (x >= WIDTH, x < 0, y >= HEIGHT, y < 0)
                if not any(case):
                    matrix[x][y] = color
        # Если dy <= 0:
        else:
            for y in range(y2, y1):
                x = round(x1 + dx*(y-y1)/dy)
                case = (x >= WIDTH, x < 0, y >= HEIGHT, y < 0)
                if not any(case):
                    matrix[x][y] = color
    return matrix


def save_picture(matrix, index=0, time_name=True):
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
            cur_val = matrix[i][j]
            if cur_val == 'white':
                tmp_pic_colors[i,j] = (255,255,255)
                continue
            if cur_val == 'black':
                tmp_pic_colors[i,j] = (0,0,0)
                continue
            if cur_val == 'red':
                tmp_pic_colors[i,j] = (255,0,0)
                continue
    if time_name:
        tmp_pic.save(os.getcwd() + f'\\pic_array1\\png_{int(time.time())}.png')
    else:
        tmp_pic.save(os.getcwd() + f'\\pic_array1\\png_{int(index):05.0f}.png')


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def main(random=False, just_save_objects=False, border=True):
    # Выбор режима работы
    # Либо расположить объекты рандомно, либо в соответствии с circle_map и polygon_map
    if random:
        ran_func = np.random.random
        circle_objects = [Circle((WIDTH*ran_func(), HEIGHT*ran_func()), 50*ran_func()) for _ in range(N_circles)]
        reg_polygon_objects = [Regular_polygon((WIDTH*ran_func(), HEIGHT*ran_func()), 50*ran_func(), int(6*ran_func()+3), 6*ran_func()) for _ in range(N_polygons)]
    else:
        circle_objects = [Circle(CIRCLE_MAP[i], CIRCLE_R) for i in range(len(CIRCLE_MAP))]
        reg_polygon_objects = []
    
    if border:
        circle_objects.append(Circle((WIDTH/2, HEIGHT/2), WIDTH+HEIGHT, special=True))

    # Создание матрицы размера WIDTH х HEIGHT 
    image_matrix = [['white' for j in range(HEIGHT)] for i in range(WIDTH)]

    # Отрисовка фигур
    for circle in circle_objects:
        image_matrix = draw_circle(circle, image_matrix)
    for polygon in reg_polygon_objects:
        image_matrix = draw_polygon(polygon, image_matrix)

    # Сохранить матрицу только с объектами
    image_with_shapes = copy.deepcopy(image_matrix)
    if just_save_objects:
        save_picture(image_matrix, 1)
        exit()

    # Цикл с поворотом начального вектора луча
    i_name = 0
    for i in range(2700, 7520):
        image_matrix = copy.deepcopy(image_with_shapes)

        # Задание нач параметров
        cur_k_vect, cur_p_start = INITIAL_k_vect, INITIAL_p_start
        cur_k = cur_k_vect[1]/cur_k_vect[0] + 1e-4*i
        cur_b = (cur_p_start[1] - cur_k*cur_p_start[0])

        # Основной цикл отрисовки 
        for _ in range(200):
        
            # Сбор точек пересечений с объектами
            points_list = []
            for circle in circle_objects:
                points = circle.calc_intersec_points((cur_k, cur_b))
                for point in points:
                    points_list.append(point)
            for polygon in reg_polygon_objects:
                points = polygon.calc_intersec_points((cur_k, cur_b))
                for point in points:
                    points_list.append(point)
            
            # Сортировка точек по расстоянию до начала луча и поиск тех, что удволетворяют следующим условиям:
            # Точка не находится там же, где начало луча; проекция вектора от начала луча до (предпологаемого) конца на вектор направления луча положительна
            points_list.sort(key=lambda point_info: (point_info[0][0]-cur_p_start[0])**2 + (point_info[0][1]-cur_p_start[1])**2)
            found = False
            if points_list:
                for j in range(len(points_list)):
                    check_intersec_point_info = points_list[j]
                    check_p_end, _ = check_intersec_point_info
                    if abs(cur_p_start[0]-check_p_end[0]) < PREC or abs(cur_p_start[1]-check_p_end[1]) < PREC:
                        continue
                    if cur_k_vect[0] * (check_p_end[0]-cur_p_start[0]) < 0:
                        continue
                    if cur_k_vect[1] * (check_p_end[1]-cur_p_start[1]) < 0:
                        continue
                    found = True
                    break
                if found:
                    cur_intersec_point_info = check_intersec_point_info
                else:
                    break
            else:
                break
            
            # Сохранить информацию о следующем луче
            next_p_start, cur_normal_vect = cur_intersec_point_info

            # Отрисовка линии на матрицк image_matrix
            image_matrix = draw_line(cur_p_start, next_p_start, image_matrix, color='red')
            
            # Вычисление вектора, отраженного от поверхности 
            cur_k_vect_np = np.array(cur_k_vect)
            cur_normal_vect_np = np.array(cur_normal_vect)
            next_k_vect = cur_k_vect_np - 2 * cur_normal_vect_np * np.dot(cur_k_vect_np, cur_normal_vect_np)/np.dot(cur_normal_vect_np, cur_normal_vect_np)

            # Новый луч
            cur_k_vect, cur_p_start = next_k_vect, next_p_start
            cur_k = cur_k_vect[1]/cur_k_vect[0]
            cur_b = (cur_p_start[1] - cur_k*cur_p_start[0])
        save_picture(image_matrix, i_name, time_name=False)
        i_name += 1

    


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


if __name__ == '__main__':
    WIDTH, HEIGHT  = 500, 500

    N_circles = 10
    N_polygons = 10
    # N_polygon_sides = 6

    INITIAL_k_vect = (1,1)
    INITIAL_p_start = (100,100)
    PREC = 1E-5

    # Задать центры окружностей
    CIRCLE_MAP = [(230, 60), (230, 60+140), (230, 60+2*140), (230, 60+3*140), 
                (315, -10), (315, -10+140), (315, -10+2*140), (315, -10+3*140), (315, -10+5*140),
                (400, 60), (400, 60+140), (400, 60+2*140), (400, 60+3*140)]
    CIRCLE_R = 40
    POLYGON_MAP = []


    main(random=False, just_save_objects=False)
