import global_var as g
import pygame as pg


#########################################################################################
#########################################################################################


class Cell():
    def __init__(self, pos:tuple[int,int]) -> None:
        self.visited = False
        self.pos = pos
        self.r_wall, self.t_wall, self.l_wall, self.b_wall = True, True, True, True
    
    def get_pos(self) -> tuple[int,int]: return self.pos
    def set_visit(self) -> None: self.visited = True
    def get_status(self) -> bool: return self.visited

    def set_r_wall(self, boo: bool) -> None: self.r_wall = boo
    def set_t_wall(self, boo: bool) -> None: self.t_wall = boo
    def set_l_wall(self, boo: bool) -> None: self.l_wall = boo
    def set_b_wall(self, boo: bool) -> None: self.b_wall = boo

    def get_walls(self) -> tuple[bool,bool,bool,bool]: 
        return [self.r_wall, self.t_wall, self.l_wall, self.b_wall]
    
    def show_fill(self, dspsurf: pg.Surface) -> None:
        ''' Method to show Cell's fill color on dspsurf '''
        w = g.CELL_WIDTH
        i,j = self.pos
        color = g.VISITED_COLOR if self.visited else g.BG_COLOR
        pg.draw.rect(dspsurf, color, pg.Rect(i*w,j*w, w, w))
    
    def show_walls(self, dspsurf: pg.Surface) -> None:
        ''' Method to show Cell's walls on dspsurf '''
        w = g.CELL_WIDTH
        i,j = self.pos
        color = (255,255,255)
        if self.r_wall:
            pg.draw.line(dspsurf, color, ((i+1)*w, (j)*w), ((i+1)*w, (j+1)*w))
        if self.t_wall:
            pg.draw.line(dspsurf, color, ((i)*w, (j)*w), ((i+1)*w, (j)*w))
        if self.l_wall:
            pg.draw.line(dspsurf, color, ((i)*w, (j)*w), ((i)*w, (j+1)*w))
        if self.b_wall:
            pg.draw.line(dspsurf, color, ((i)*w, (j+1)*w), ((i+1)*w, (j+1)*w))


#########################################################################################
#########################################################################################


if __name__ == '__main__':
    test_obj = Cell((0,0))