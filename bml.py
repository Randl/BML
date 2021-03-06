import time
from enum import IntEnum

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
    def __init__(self, width, height, p, model, move_probability=1, animation_ratio=10):
        self.width = width
        self.height = height
        self.cells = np.zeros((height, width))
        self.model = model  # probabilistic, klein bottle, chain
        self.step = 0
        self.velocity = []
        self.animation_ration = animation_ratio
        self.move_probability = move_probability
        
        num_of_cars = int(width * height * (p / 2))
        # Get a list of indices
        inds = list(np.ndindex(self.cells.shape))
        # Shuffle the indices in-place
        np.random.shuffle(inds)
        # Access array elements using the indices to do cool stuff
        for i in inds[:num_of_cars]:
            self.cells[i] = Cell.RIGHT

        for i in inds[num_of_cars:2 * num_of_cars]:
            self.cells[i] = Cell.DOWN
        
        self.total_number = 2 * num_of_cars

        self.animation = [self.get_image()]
    
    def make_step(self):
        if self.model == 1:
            vel = self.step_right()
            vel += self.step_down()
            self.velocity.append(vel / self.total_number)
        elif self.model == 2:
            self.velocity.append(self.step_all() / self.total_number)
        elif self.model == 3:
            self.velocity.append(self.step_all_together() / self.total_number)
        self.step += 1
        if self.step % self.animation_ration == 0:
            self.animation.append(self.get_image())
    
    def step_down(self):
        d = np.diff(np.r_[self.cells, [self.cells[0, :]]], axis=0)
        down_indexes = np.where(np.logical_and(self.cells == Cell.DOWN, d == Cell.EMPTY - Cell.DOWN))
        if self.move_probability < 1:
            tmp = np.transpose(down_indexes)
            np.random.shuffle(tmp)
            down_indexes = tuple(np.transpose(tmp[:int(self.move_probability * tmp.shape[0]), :]))
        empty_indexes = self.down_set(down_indexes)
        self.cells[down_indexes] = Cell.EMPTY
        self.cells[empty_indexes] = Cell.DOWN
        return len(down_indexes[0])
    
    def step_right(self):
        d = np.diff(np.c_[self.cells, self.cells[:, 0]])
        right_indexes = np.where(np.logical_and(self.cells == Cell.RIGHT, d == Cell.EMPTY - Cell.RIGHT))
        empty_indexes = self.right_set(right_indexes)
        self.cells[right_indexes] = Cell.EMPTY
        self.cells[empty_indexes] = Cell.RIGHT
        return len(right_indexes[0])
    
    def step_all(self):
        dr = np.c_[self.cells[:, -1], self.cells[:, :-1]] - self.cells
        dd = np.r_[[self.cells[-1, :]], self.cells[:-1, :]] - self.cells

        empty_cells = self.cells == Cell.EMPTY

        empty_indexes_both = np.transpose(np.where(np.logical_and(empty_cells,
                                                                  np.logical_and(dd == Cell.DOWN - Cell.EMPTY,
                                                                                 dr == Cell.RIGHT - Cell.EMPTY))))

        if empty_indexes_both.shape[0] > 0:
            np.random.shuffle(empty_indexes_both)
            right_nums = int(empty_indexes_both.shape[0] / 2)
            # @formatter:off
            empty_indexes_right = tuple(np.transpose(np.r_[
                                                      np.transpose(np.where(
                                                                   np.logical_and(empty_cells,
                                                                   np.logical_and(dr == Cell.RIGHT - Cell.EMPTY,
                                                                                  dd != Cell.DOWN - Cell.EMPTY)))),
                                                      empty_indexes_both[:right_nums]]))
            empty_indexes_down = tuple(np.transpose(np.r_[
                                                      np.transpose(np.where(
                                                              np.logical_and(empty_cells,
                                                              np.logical_and(dd == Cell.DOWN - Cell.EMPTY,
                                                                             dr != Cell.RIGHT - Cell.EMPTY)))),
                                                      empty_indexes_both[right_nums:]]))
            # @formatter:on
        else:
            empty_indexes_right = np.where(np.logical_and(empty_cells, dr == Cell.RIGHT - Cell.EMPTY))
            empty_indexes_down = np.where(np.logical_and(empty_cells, dd == Cell.DOWN - Cell.EMPTY))

        right_indexes = self.left_set(empty_indexes_right)
        down_indexes = self.up_set(empty_indexes_down)

        self.cells[right_indexes] = Cell.EMPTY
        self.cells[down_indexes] = Cell.EMPTY
        self.cells[empty_indexes_right] = Cell.RIGHT
        self.cells[empty_indexes_down] = Cell.DOWN

        # if self.step % 100 == 1:
        #    self.save()
        return len(right_indexes[0]) + len(down_indexes[0])
    
    def step_all_together(self):
        dr = np.c_[self.cells[:, -1], self.cells[:, :-1]] - self.cells
        dd = np.r_[[self.cells[-1, :]], self.cells[:-1, :]] - self.cells
    
        down_cond = np.logical_or(dd == Cell.DOWN - Cell.EMPTY, dd == Cell.BOTH - Cell.EMPTY)
        right_cond = np.logical_or(dr == Cell.RIGHT - Cell.EMPTY, dr == Cell.BOTH - Cell.EMPTY)
        empty_cells = self.cells == Cell.EMPTY
        empty_indexes_both = np.where(np.logical_and(empty_cells, np.logical_and(down_cond, right_cond)))

        empty_indexes_right = np.where(
            np.logical_and(empty_cells, np.logical_and(right_cond, np.logical_not(down_cond))))
        empty_indexes_down = np.where(
            np.logical_and(empty_cells, np.logical_and(down_cond, np.logical_not(right_cond))))
    
        right_indexes = self.left_set(empty_indexes_right)
        down_indexes = self.up_set(empty_indexes_down)
    
        self.cells[right_indexes] -= 1
        self.cells[down_indexes] -= 2
        self.cells[empty_indexes_right] = Cell.RIGHT
        self.cells[empty_indexes_down] = Cell.DOWN
    
        if len(empty_indexes_both[0]) > 0:
            both_right_indexes = self.left_set(empty_indexes_both)
            both_down_indexes = self.up_set(empty_indexes_both)
            self.cells[both_right_indexes] -= 1
            self.cells[both_down_indexes] -= 2
            self.cells[empty_indexes_both] = Cell.BOTH

        return len(right_indexes[0]) + len(down_indexes[0]) + len(empty_indexes_both[0]) * 2
        


    def down_set(self, inds):
        return tuple([[(x + 1) % self.height for x in inds[0]], inds[1]])
    
    def right_set(self, inds):
        return tuple([inds[0], [(x + 1) % self.width for x in inds[1]]])

    def up_set(self, inds):
        return tuple([[(x - 1) % self.height for x in inds[0]], inds[1]])

    def left_set(self, inds):
        return tuple([inds[0], [(x - 1) % self.width for x in inds[1]]])
    
    def run(self, steps):
        start = time.perf_counter()
        for i in range(steps):
            self.make_step()
        total_time = time.perf_counter() - start
        print('{} steps for {}x{} map run in {}s. Average of {}s per step'.format(steps, self.height, self.width,
                                                                                  total_time, total_time / steps))

    def get_image(self, pixel_size=2):
        """
        Returns image
        """
        visual = np.zeros((self.height, self.width, 3)).astype('uint8')
        down_indexes = np.where(self.cells == Cell.DOWN)
        right_indexes = np.where(self.cells == Cell.RIGHT)
        both_indexes = np.where(self.cells == Cell.BOTH)
        visual[:, :, :] = 255
        visual[:, :, 0][down_indexes] = 0  # down is blue
        
        visual[:, :, 1][down_indexes] = 0
        visual[:, :, 1][right_indexes] = 0
        visual[:, :, 1][both_indexes] = 0  # both is magenta
        
        visual[:, :, 2][right_indexes] = 0  # right are red
        im = Image.fromarray(visual, mode='RGB')
        im = im.resize((self.height * pixel_size, self.width * pixel_size))
        return im
        
        
    def save(self, pixel_size=10):
        """
        Saves current state as png image
        """
        visual = np.zeros((self.height, self.width, 3)).astype('uint8')
        down_indexes = np.where(self.cells == Cell.DOWN)
        right_indexes = np.where(self.cells == Cell.RIGHT)
        both_indexes = np.where(self.cells == Cell.BOTH)
        visual[:, :, :] = 255
        visual[:, :, 0][down_indexes] = 0  # down is blue

        visual[:, :, 1][down_indexes] = 0
        visual[:, :, 1][right_indexes] = 0
        visual[:, :, 1][both_indexes] = 0  # both is magenta

        visual[:, :, 2][right_indexes] = 0  # right are red
        im = Image.fromarray(visual, mode='RGB')
        im = im.resize((self.height * pixel_size, self.width * pixel_size))
        im.save("visual" + str(self.step) + ".png")
    
    def save_animation(self):
        self.animation[0].save("progress.gif", append_images=self.animation[1:], save_all=True, optimize=True,
                               duration=10)
        
    def plot_velocity(self):
        """
        Builds plot of velocity as a function of step number until now
        """
        params = {'axes.labelsize': 8, 'font.size': 8, 'legend.fontsize': 10, 'xtick.labelsize': 10,
                  'ytick.labelsize': 10, 'text.usetex': False, 'figure.figsize': [4.5, 4.5]}
        rcParams.update(params)
        fig = figure()  # no frame
        ax = fig.add_subplot(111)
        ax.plot(self.velocity, linewidth=1, color=colors[1])
        ax.set_ylim(0, 1)
        ax.set_yticks(np.arange(0, 1, 0.2))
        
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

        ax.set_xlabel('Step')
        ax.set_ylabel('Velocity')

        ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
        ax.set_axisbelow(True)

        plt.tight_layout()

        fig.savefig('velocity.png', dpi=600)
        # print(self.velocity)


automat = BML(width=256, height=256, p=0.2, model=1, move_probability=0.9, animation_ratio=10)
automat.save()
automat.run(5000)
automat.save()
automat.save_animation()
automat.plot_velocity()
