import global_var as g
import cell

from time import time
from random import choice
from os import getcwd

import pygame as pg
from PIL import Image
import numpy as np


#########################################################################################
#########################################################################################


class Board():
    def __init__(self, size: int, start: tuple[int,int]) -> None:
        self.size = size
        self.cells = [[cell.Cell((i,j)) for j in range(self.size)] for i in range(self.size)]
        self.curr_pos = start
        self.mem_path = [start]

    def make_move1(self) -> None:
        i,j = self.curr_pos
        curr_cell = self.cells[i][j]
        curr_cell.set_visit()

        # Create a list of neighbours, that have not been visited yet
        neigh = []
        if i>0:
            if self.cells[i-1][j].get_status() == False:
                neigh.append(self.cells[i-1][j])
        if j>0:
            if self.cells[i][j-1].get_status() == False:
                neigh.append(self.cells[i][j-1])
        if i<self.size-1:
            if self.cells[i+1][j].get_status() == False:
                neigh.append(self.cells[i+1][j])
        if j<self.size-1:
            if self.cells[i][j+1].get_status() == False:
                neigh.append(self.cells[i][j+1])
        
        # If no available neighbours -> back up
        if len(neigh) == 0:
            self.curr_pos = self.mem_path.pop()
            return

        # Get random neighbour and it's position
        next_cell = choice(neigh)
        next_pos = next_cell.get_pos()
        d_j, d_i = next_pos[0]-self.curr_pos[0], next_pos[1]-self.curr_pos[1]

        # Destroy walls between current cell and next cell
        if d_i == 0:
            if d_j > 0:
                curr_cell.set_r_wall(False)
                next_cell.set_l_wall(False)
            else:
                curr_cell.set_l_wall(False)
                next_cell.set_r_wall(False)
        elif d_i > 0:
            curr_cell.set_b_wall(False)
            next_cell.set_t_wall(False)
        else:
            curr_cell.set_t_wall(False)
            next_cell.set_b_wall(False)

        # Append current position to a stack and change current position
        self.mem_path.append(self.curr_pos)
        self.curr_pos = next_pos
    
    def show(self, dspsurf: pg.Surface) -> None: 
        # Draw rectangles
        for i in range(self.size):
            for j in range(self.size):
                self.cells[i][j].show_fill(dspsurf)
        
        # Draw current cell a different color
        curr_rect = pg.Rect(self.curr_pos[0]*g.CELL_WIDTH,self.curr_pos[1]*g.CELL_WIDTH, g.CELL_WIDTH, g.CELL_WIDTH)
        pg.draw.rect(dspsurf, g.CURRENT_COLOR, curr_rect)
        
        # Draw  walls
        for i in range(self.size):
            for j in range(self.size):
                self.cells[i][j].show_walls(dspsurf)
    
    def save_maze_pic(self) -> None:
        tot_img_width = g.BOARD_SIZE*2+1
        tmp_pic = Image.new('RGB', (tot_img_width, tot_img_width), g.IMG_BG_COLOR)
        tmp_pic_colors = tmp_pic.load()

        # Draw Cells' corners
        for i in range(0,tot_img_width,2):
            for j in range(0,tot_img_width,2):
                tmp_pic_colors[i,j] = g.IMG_WALL_COLOR

        # Draw walls between Cells
        for i in range(1, tot_img_width, 2):
            for j in range(1, tot_img_width, 2):
                curr_cell_walls = self.cells[int(np.floor(i/2))][int(np.floor(j/2))].get_walls()
                if curr_cell_walls[0]:
                    tmp_pic_colors[i+1,j] = g.IMG_WALL_COLOR
                if curr_cell_walls[1]:
                    tmp_pic_colors[i,j-1] = g.IMG_WALL_COLOR
                if curr_cell_walls[2]:
                    tmp_pic_colors[i-1,j] = g.IMG_WALL_COLOR
                if curr_cell_walls[3]:
                    tmp_pic_colors[i,j+1] = g.IMG_WALL_COLOR

        tmp_pic.save(getcwd() + f'\\{g.PATH_TO_SAVE}\\png_{int(time())}.png')
                