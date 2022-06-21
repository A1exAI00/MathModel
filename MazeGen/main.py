'''
Maze generator 
Generates maze and shows a process of creation
'''


#########################################################################################
#########################################################################################


'''
DONE: PyGame version
DONE: Mark visited a different color
DONE: Walls removal
DONE: Saving pixelated maze as a pic
TODO: Saving nice looking maze as a pic
DONE: (?) refacror
DONE: type hinting
'''


#########################################################################################
#########################################################################################


import board as b
import global_var as g

from time import perf_counter

import pygame as pg


#########################################################################################
#########################################################################################


def main1():
    '''
    Main function, where the process of maze creation is shown
    '''
    # PyGame init business
    pg.init()
    DISPLAYSURF = pg.display.set_mode((g.WIDTH, g.HEIGHT))
    pg.display.set_caption(g.WINDOW_NAME)
    clock = pg.time.Clock()

    # Creating blank maze 
    board = b.Board(g.BOARD_SIZE, (0,0))
    maze_ready = False

    # Main gameloop
    gameLoop = True
    frames = 0
    while gameLoop:

        # PyGame loop business
        DISPLAYSURF.fill(g.BG_COLOR)
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                gameLoop = False
        
        # Raises an IndexError, when maze is ready
        try:
            board.make_move1()
        except IndexError:
            maze_ready = True
            gameLoop = False
        
        # Update display
        frames += 1
        if frames%g.MAX_FRAMES_SKIPPED == 0:
            board.show(DISPLAYSURF)
            pg.display.update()
        if g.APPLY_SLOW_DOWN:
            clock.tick(g.FPS)
    
    if maze_ready:
        print('MAZE FINISHED')
        if g.SAVE_PIS:
            board.save_maze_pic()
    
    # Close window
    pg.quit()


#########################################################################################
#########################################################################################


def main2():
    '''
    Main function that just generates a maze picture
    '''
    board = b.Board(g.BOARD_SIZE, (0,0))
    maze_ready = False

    gameLoop = True
    while gameLoop:
        # Raises an IndexError, when maze is ready
        try:
            board.make_move1()
        except IndexError:
            maze_ready = True
            gameLoop = False
    
    # Indicate that maze generation is ready
    if maze_ready:
        print('MAZE FINISHED')
        if g.SAVE_PIS:
            board.save_maze_pic()


#########################################################################################
#########################################################################################


if __name__ == '__main__':
    start_time = perf_counter()
    if g.PG_WINDOW:
        main1()
    else:
        main2()
    end_time = perf_counter()

    print(f'Eval: {end_time-start_time}')
    