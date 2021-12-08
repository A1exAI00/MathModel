# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:38:00 2021

@author: Alex Akinin

"""


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


import turtle as tt
import numpy as np
import time


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


wn = tt.Screen()
wn.tracer(0) 
turts = [tt.Turtle() for i in range(500)]

for turt in turts:
    rgb = (round(255*np.random.rand()) for i in range(3))
    # turt.color(rgb)
    turt.speed(0)

maxims = []
times = []

while True:
    st = time.perf_counter()
    for turt in turts:
        leng = 10 * np.random.rand()
        angle = 50*(np.random.rand() - 1/2.)/2
        turt.left(angle)
        turt.forward(leng)
     
    maxim = []
    for i in turts:
        maxim.append(abs(i.pos()))
        
        
    maxims.append(np.mean(maxim))
    wn.update()
    times.append(time.perf_counter() - st)
        

for turt in turts:
    turt.done()

wn.mainloop() 