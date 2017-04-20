import functools
import time
from operator import mul

from mpl_toolkits.mplot3d import Axes3D
from pylab import *

num_of_colors = 3
colormap = plt.get_cmap('inferno')
colors = [colormap(k) for k in np.linspace(0.0, 0.8, num_of_colors)]
sec_colors = [colormap(k + 0.1) for k in np.linspace(0.0, 0.8, num_of_colors)]


class nd_BML:
    """
    Multidimensional BML
    """
    
    def __init__(self, shape, p, move_probability=1):
        self.shape = tuple(shape)
        self.dims = len(shape)
        self.cells = np.zeros(self.shape, dtype=np.int8)
        self.step = 0
        self.velocity = []
        self.move_probability = move_probability
        
        num_of_cars = int(functools.reduce(mul, self.shape) * (p / self.dims))
        # Get a list of indices
        inds = list(np.ndindex(self.cells.shape))
        # Shuffle the indices in-place
        np.random.shuffle(inds)
        # Access array elements using the indices to do cool stuff
        for n in range(self.dims):
            for i in inds[n * num_of_cars:(n + 1) * num_of_cars]:
                self.cells[i] = n + 1
        
        self.total_number = self.dims * num_of_cars
    
    def make_step(self):
        vel = self.step_nd()
        self.velocity.append(vel / self.total_number)
        self.step += 1
    
    def step_nd(self):
        moved = 0
        for n in range(self.dims):
            axis = np.arange(0, self.dims)
            axis[0] = n
            axis[n] = 0
            
            self.cells = self.cells.transpose(tuple(axis))
            
            CURR = n + 1
            d = np.diff(np.r_[self.cells, [self.cells[0, ...]]], axis=0)
            curr_indexes = np.where(np.logical_and(self.cells == CURR, d == -CURR))
            empty_indexes = self.down_set(curr_indexes, self.cells.shape[0])
            self.cells[curr_indexes] = 0
            self.cells[empty_indexes] = CURR
            
            self.cells = self.cells.transpose(tuple(axis))  # get back
            moved += len(curr_indexes[0])
        
        return moved
    
    def down_set(self, inds, dim):
        return tuple([[(x + 1) % dim for x in inds[0]]] + list(inds[1:]))
    
    def run(self, steps):
        start = time.perf_counter()
        for i in range(steps):
            self.make_step()
        total_time = time.perf_counter() - start
        print('{} steps for {} map run in {}s. Average of {}s per step'.format(steps, self.shape_string(), total_time,
                                                                               total_time / steps))
    
    def shape_string(self):
        shape_str = ''
        for x in self.cells.shape:
            shape_str += str(x) + 'x'
        return shape_str[:-1]

    def save(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(0, self.shape[0])
        ax.set_ylim(0, self.shape[1])
        ax.set_zlim(0, self.shape[2])
        cl = ['r', 'g', 'b']
        for n in range(self.dims):
            c = np.where(self.cells == n + 1)
            # print(c)
            # print(len(c[0]))
            ax.scatter(c[0], c[1], c[2], c=cl[n], marker=matplotlib.markers.MarkerStyle('s', 'none'))
    
        fig.savefig("visual3d_" + str(self.step) + ".png", dpi=600)
        
    
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


automat = nd_BML((32, 32, 32), 0.1)
automat.save()
automat.run(1000)
automat.save()
automat.plot_velocity()
