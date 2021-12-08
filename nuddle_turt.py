# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:06:22 2021

@author: Alex
"""

import turtle as tt
import numpy as np
import time

wn = tt.Screen()
wn.tracer(0)

aaa = tt.Turtle()
color = (0, 0, 0)
tt.colormode(255)
# aaa.color(())
aaa.speed(0)


def sq(turt, leng):
    for _ in range(4):
        turt.forward(leng)
        turt.right(90)


def rainbow_color():
    global color
    r, g, b = color
    for _ in range(3):
        if r >= 255:
            r = 0; g += 1
        if g >= 255:
            g = 0; b += 1
        if b >= 255:
            b = 0; r += 1
    r += 1
    color = r, g, b
    return color

def my_abs_sin(arg):
    pi = np.pi
    arg = arg % pi
    if arg < pi/2:
        return 2 * arg/pi
    else:
        return 2* (-arg + pi)/pi
     


t = 0
aaa.setpos((0, 0))
while True:
    # leng = 10 * np.random.rand()
    # angle = 180*(np.random.rand() - 1/2.)
    # tt.forward(leng)
    # tt.left(angle)
    tt = round((t/30)%1, 2)
    if tt == 0:
        aaa.left(90)
    aaa.forward(1)
    aaa.left(3*np.sin(t))
    sq(aaa, t)
    
    red_sin = round(255 *(my_abs_sin(2*t)))
    green_sin = round(255 *(my_abs_sin(3*t + 1/3 * np.pi)))
    blue_sin = round(255 * (my_abs_sin(5*t + 2/3 * np.pi)))
    
    aaa.color(red_sin, green_sin, blue_sin)
    wn.update()

    t += 0.01

aaa.done()
