import random
from enum import IntEnum

import numpy as np
from PIL import Image


class Cell(IntEnum):
    EMPTY = 0
    RIGHT = 1
    DOWN = 2
    BOTH = 3


class BML:
    def __init__(self, width, height, p, model):
        self.width = width
        self.height = height
        self.cells = np.zeros((height, width))
        self.model = model
        self.step = 0
        self.velocity = []
        
        num_of_cars = int(width * height * (p / 2))
        # Get a list of indices
        indices = list(np.ndindex(self.cells.shape))
        # Shuffle the indices in-place
        np.random.shuffle(indices)
        # Access array elements using the indices to do cool stuff
        for i in indices[:num_of_cars]:
            self.cells[i] = Cell.RIGHT
        for i in indices[num_of_cars:2 * num_of_cars]:
            self.cells[i] = Cell.DOWN
        
        self.total_number = 2 * num_of_cars
    
    def make_step(self):
        if self.model == 1:
            vel = self.step_right()
            vel += self.step_down()
            self.velocity.append(vel / (self.width * self.height))
        elif self.model == 2:
            self.velocity.append(self.step_all() / (self.width * self.height))
        elif self.model == 3:
            self.step_all_together()  # TODO
        self.step += 1
    
    def step_down(self):
        next_state = self.cells
        count = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.cells[self.upper(i, j)] == Cell.DOWN and self.cells[(i, j)] == Cell.EMPTY:
                    next_state[(i, j)] = Cell.DOWN
                    next_state[self.upper(i, j)] = Cell.EMPTY
                    count += 1
        self.cells = next_state
        return count
    
    def step_all_together(self):
        return -1
    
    def step_right(self):
        next_state = self.cells
        count = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.cells[self.left(i, j)] == Cell.RIGHT and self.cells[(i, j)] == Cell.EMPTY:
                    next_state[(i, j)] = Cell.RIGHT
                    next_state[self.left(i, j)] = Cell.EMPTY
                    count += 1
        self.cells = next_state
        return count
    
    def step_all(self):
        next_state = np.zeros((self.height, self.width))
        count = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.cells[(i, j)] == Cell.EMPTY:
                    if self.cells[self.left(i, j)] == Cell.RIGHT and self.cells[self.upper(i, j)] == Cell.DOWN:
                        if random.choice((True, False)):
                            next_state[(i, j)] = Cell.RIGHT
                            next_state[self.left(i, j)] = Cell.EMPTY
                        else:
                            next_state[(i, j)] = Cell.DOWN
                            next_state[self.upper(i, j)] = Cell.EMPTY
                        count += 1
                    elif self.cells[self.left(i, j)] == Cell.RIGHT:
                        next_state[(i, j)] = Cell.RIGHT
                        next_state[self.left(i, j)] = Cell.EMPTY
                        count += 1
                    elif self.cells[self.upper(i, j)] == Cell.DOWN:
                        next_state[(i, j)] = Cell.DOWN
                        next_state[self.upper(i, j)] = Cell.EMPTY
                        count += 1
        self.cells = next_state
        return count
    
    def upper(self, i, j):
        return self.height - 1 if i == 0 else i - 1, j
    
    def left(self, i, j):
        return i, self.width - 1 if j == 0 else j - 1
    
    def run(self, steps):
        for i in range(steps):
            self.make_step()
    
    def save(self):
        visual = np.zeros((self.height, self.width, 3)).astype('uint8')
        for i in range(self.height):
            for j in range(self.width):
                visual[i, j, 0] = 0 if self.cells[(i, j)] == Cell.DOWN else 255
                visual[i, j, 1] = 0 if self.cells[(i, j)] != Cell.EMPTY else 255
                visual[i, j, 2] = 0 if self.cells[(i, j)] == Cell.RIGHT else 255
        im = Image.fromarray(visual, mode='RGB')
        im.save("visual" + str(self.step) + ".png")


automat = BML(16, 16, 0.5, 1)
automat.save()
automat.run(1000)
automat.save()
