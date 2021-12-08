# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 09:34:59 2021

@author: Alex Akinin

Моделирование папоротника Барнсли 
Программа возможно не моя, я не помню
"""


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


import turtle
import random
import numpy as np


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


progress = 0
N_PIXEL = 100000


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


def fract_turt_run():
    pen = turtle.Turtle()
    pen.speed(0)
    pen.color("green")
    pen.penup()
    wn = turtle.Screen()
    wn.tracer(1000) 
    
    x = 0
    y = 0
    while True:
        pen.goto(100 * x, 70 * y - 350)
        pen.pendown()
        pen.dot(1.5)
        pen.penup()
        r = random.random()
        if r < 0.01:
            x, y =  0.00 * x + 0.00 * y,  0.00 * x + 0.16 * y + 0.00
        elif r < 0.86:
            x, y =  0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.60
        elif r < 0.93:
            x, y =  0.20 * x - 0.26 * y,  0.23 * x + 0.22 * y + 1.60
        else:
            x, y = -0.15 * x + 0.28 * y,  0.26 * x + 0.24 * y + 0.44
        # wn.update()
    wn.mainloop() 


# ----------------------------------------------------------------------


def fract_array(N=N_PIXEL):
    global progress
    x = 0
    y = 0
    arr = np.array([])
    for n in range(N):
    # while True:
        arr = np.append(arr, np.array([100 * x, 70 * y - 350]))
        r = random.random()
        if r < 0.01:
            x, y =  0.00 * x + 0.00 * y,  0.00 * x + 0.16 * y + 0.00
        elif r < 0.86:
            x, y =  0.85 * x + 0.04 * y, -0.04 * x + 0.85 * y + 1.60
        elif r < 0.93:
            x, y =  0.20 * x - 0.26 * y,  0.23 * x + 0.22 * y + 1.60
        else:
            x, y = -0.15 * x + 0.28 * y,  0.26 * x + 0.24 * y + 0.44
        progress += 1
    return arr


# ----------------------------------------------------------------------


def fract_from_arr(arr):
    tmp_a = arr.copy()
    pen = turtle.Turtle()
    pen.speed(0)
    pen.color("green")
    pen.penup()
    wn = turtle.Screen()
    wn.tracer(10000) 
    
    for i in range(round(len(tmp_a)/2)):
        pen.goto(tmp_a[2*i], tmp_a[2*i+1])
        pen.pendown()
        pen.dot(1.5)
        pen.penup()
    wn.mainloop() 


# ----------------------------------------------------------------------


fin = fract_array()
fract_from_arr(fin)
# fract_turt_run()