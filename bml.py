from enum import IntEnum
from random import choice

from PIL import Image
from pylab import *

num_of_colors = 2
colormap = plt.get_cmap('inferno')
colors = [colormap(k) for k in np.linspace(0.0, 0.8, num_of_colors)]
sec_colors = [colormap(k + 0.1) for k in np.linspace(0.0, 0.8, num_of_colors)]

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
        next_state = self.cells
        count = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.cells[(i, j)] == Cell.EMPTY:
                    if self.cells[self.left(i, j)] == Cell.RIGHT and self.cells[self.upper(i, j)] == Cell.DOWN:
                        if choice((True, False)):
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

    def save(self, pixel_size=10):
        """
        Saves current state as png image
        """
        visual = np.zeros((self.height, self.width, 3)).astype('uint8')
        for i in range(self.height):
            for j in range(self.width):
                visual[i, j, 0] = 0 if self.cells[(i, j)] == Cell.DOWN else 255
                visual[i, j, 1] = 0 if self.cells[(i, j)] != Cell.EMPTY else 255
                visual[i, j, 2] = 0 if self.cells[(i, j)] == Cell.RIGHT else 255
        im = Image.fromarray(visual, mode='RGB')
        im = im.resize((self.height * pixel_size, self.width * pixel_size))
        im.save("visual" + str(self.step) + ".png")

    def plot_velocity(self):
        """
        Builds plot of velocity as a function of step number until now
        """
        params = {'axes.labelsize': 8, 'font.size': 8, 'legend.fontsize': 10, 'xtick.labelsize': 10,
                  'ytick.labelsize': 10, 'text.usetex': False, 'figure.figsize': [4.5, 4.5]}
        rcParams.update(params)
        fig = figure()  # no frame
        ax = fig.add_subplot(111)
        ax.plot(self.velocity, linewidth=1, color=colors[0])
        ax.set_ylim(0, 1)
    
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    
        ax.set_xlabel('Step')
        ax.set_ylabel('Velocity')
    
        ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
        ax.set_axisbelow(True)
    
        plt.tight_layout()
    
        fig.savefig('velocity.png', dpi=600)


automat = BML(16, 16, 0.5, 2)
automat.save()
automat.run(100)
automat.save()
automat.plot_velocity()
