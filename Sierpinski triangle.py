# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 11:37:04 2021

@author: Alex Akinin

Моделирование треугольника Серпинского
"""


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


import turtle as tt
import numpy as np


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


start = (-390,-330)
end = (390, -330)
aba = 1

wn = tt.Screen()
wn.screensize(2000,1500)
tt.penup()
tt.goto(start)
tt.pendown()
tt.speed(0)
wn.delay(1000)

PREC = 7

def three_lines(p1, p2):
    global aba
    x1, y1 = p1
    x2, y2 = p2
    leng = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    pt = []
    set_angle(p1, p2)
    
    if aba % 2 != 0:
        kk = 1
    else:
        kk = -1
    
    pt.append(tuple(tt.pos()))
    tt.left(kk*60)
    # tt.right(60)
    tt.forward(leng/2)
    pt.append(tuple(tt.pos()))
    tt.right(kk*60)
    tt.forward(leng/2)
    pt.append(tuple(tt.pos()))
    tt.right(kk*60)
    tt.forward(leng/2)
    pt.append(tuple(tt.pos()))
    # tt.left(kk*60)
    
    
    return pt


def set_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    if (y2-y1) < 0 and (x2-x1) < 0:
        tt.setheading((np.arctan((y2-y1)/(x2-x1)) * 180/np.pi)-180)
    elif (y2-y1) < 0 and (x2-x1) > 0:
        tt.setheading((np.arctan((y2-y1)/(x2-x1)) * 180/np.pi))
    elif (y2-y1) >= 0 and (x2-x1) < 0:
        tt.setheading((np.arctan((y2-y1)/(x2-x1)) * 180/np.pi)-180)
    else:
        tt.setheading((np.arctan((y2-y1)/(x2-x1)) * 180/np.pi))


def del_dupe(lii):
    tmp_a = lii.copy()
    fin = []
    rounded = []
    for i in range(len(tmp_a)):
        a,b = tmp_a[i]
        rounded.append((round(a, PREC), round(b, PREC)))
    tmp_a = rounded.copy()
    for i in tmp_a:
        if i not in fin:
            fin.append(i)
    return fin



start = (-390,-330)
end = (390, -330)
aba = 1
aa = [start, end]

wn = tt.Screen()
wn.screensize(2000,1500)
tt.penup()
tt.goto(start)
tt.pendown()
tt.speed(0)
wn.delay(1000)

PREC = 7

wn.delay(0)
for j in range(8):
    bb = []
    tt.clear()
    
    for i in range(0, len(aa)-1):
        tt.penup()
        tt.goto(aa[i])
        tt.pendown()
        
        cc = three_lines(aa[i], aa[i+1])
        for k in cc:
            bb.append(k)
        aba +=1
    
    wn.update()

    aa = del_dupe(bb.copy())
