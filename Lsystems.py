# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 10:15:49 2021

@author: Alex
"""

import turtle as tt
import numpy as np
import time

N = 5
tt.speed(0)


# ----------------------------------------------------------------------


def window_setup():
    global wn
    wn = tt.Screen()
    wn.screensize(3000,3000)
    wn.tracer(1000) 
    # tt.exitonclick()


# ----------------------------------------------------------------------


def gen_spier_triangle(nmax=5):
    axiom = 'A'
    string = axiom
    
    for _ in range(nmax):
        cur = []
        cur_srt = ''
        for j in string:
            if j == 'A':
                cur.append('B−A−B')
            elif j == 'B':
                cur.append('A+B+A')
            elif j == '+':
                cur.append('+')
            else: 
                cur.append('-')
        
        for j in cur:
            cur_srt += str(j)
        
        string = cur_srt
    
    return string


def turt_spier_triangle(L=500, n=5):
    path = gen_spier_triangle(n)
    angle = 60
    
    window_setup()
    
    for j in path:
        if j == 'A':
            tt.forward(L/(2**n))
        elif j == 'B':
            tt.forward(L/(2**n))
        elif j == '+':
            tt.left(angle)
        elif j == '-':
            tt.right(angle)
    wn.update()


# ----------------------------------------------------------------------


def gen_dragon_curve(nmax=5):
    axiom = 'F'
    string = axiom
    
    for _ in range(nmax):
        inv_str = string[::-1]
        inv_str_l = []
        
        for i in range(len(inv_str)):
            if inv_str[i] == '+':
                inv_str_l.append('-')
            elif inv_str[i] == '-':
                inv_str_l.append('+')
            else:
                inv_str_l.append(inv_str[i])
        
        inv_str = ''
        for i in inv_str_l:
            inv_str += i
        
        string += '+' + inv_str
    # print(string)
    return string


def turt_dragon_curve(L=5, n=5):
    path = gen_dragon_curve(n)
    angle = 90
    window_setup()
    
    for j in path:
        if j == 'F':
            tt.forward(L)
        elif j == '+':
            tt.right(angle)
        elif j == '-':
            tt.left(angle)
    wn.update()


# ----------------------------------------------------------------------


def gen_hilb_curve(nmax=5):
    axiom = 'A'
    string = axiom
    
    for _ in range(nmax):
        cur = []
        cur_srt = ''
        for j in string:
            if j == 'A':
                cur.append('−BF+AFA+FB−')
            elif j == 'B':
                cur.append('+AF−BFB−FA+')
            elif j == '+':
                cur.append('+')
            elif j == '-':
                cur.append('-')
            elif j == 'F':
                cur.append('FF')
        
        for j in cur:
            cur_srt += str(j)
        
        string = cur_srt
    
    return string


def turt_hilb_curve(L=50, n=5):
    path = gen_hilb_curve(n)
    angle = 90
    window_setup()
    
    for j in path:
        if j == 'F':
            tt.forward(L/(n))
        elif j == '+':
            tt.left(angle)
        elif j == '-':
            tt.right(angle)
    wn.update()


# ----------------------------------------------------------------------


def gen_snowflake(nmax=5):
    axiom = 'F--F--F'
    string = axiom
    
    for _ in range(nmax):
        cur = []
        cur_srt = ''
        
        for j in string:
            if j == 'F':
                cur.append('F+F--F+F')
            elif j == '+':
                cur.append('+')
            else: 
                cur.append('-')
        
        for j in cur:
            cur_srt += str(j)
        
        string = cur_srt
    return string


def turt_snowflake(L=50, n=5):
    path = gen_snowflake(n)
    angle = 60
    window_setup()
    
    for j in path:
        if j == 'F':
            tt.forward(L/(2.6**n))
        elif j == '+':
            tt.left(angle)
        elif j == '-':
            tt.right(angle)
    wn.update()


# ----------------------------------------------------------------------


# turt_spier_triangle(L=5000, n=10)

turt_dragon_curve(L=2, n=15) # не совсем L-система 

# turt_hilb_curve(50,2)

# turt_snowflake(L=50, n=7)

time.sleep(5)