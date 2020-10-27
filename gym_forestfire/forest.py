"""
Forest-fire simulation based on the rossel and
Schwabl (1992) model. The vectorized implementation
of cellular automaton was adapted from:
http://drsfenner.org/blog/2015/08/game-of-life-in-numpy-2/

Author: Sahand Rezaei-Shoshtari
Copyright protected.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import colors
from numpy.lib.stride_tricks import as_strided


def get_neighborhoud(arr):
    """
    Computes the neighborhood for each element in the grid.
    """
    assert all(_len > 2 for _len in arr.shape)

    nDims = len(arr.shape)
    newShape = [_len - 2 for _len in arr.shape]
    newShape.extend([3] * nDims)

    newStrides = arr.strides + arr.strides
    return as_strided(arr, shape=newShape, strides=newStrides)


class Forest:

    def __init__(self, world_size=(100, 100), p_fire=0.0005, p_tree=0.0005):
        self.EMPTY_CELL = 0
        self.TREE_CELL = 1
        self.FIRE_CELL = 10

        self.p_fire = p_fire
        self.p_tree = p_tree

        full_size = tuple(i + 2 for i in world_size)
        self.full = np.zeros(full_size, dtype=np.uint16)
        nd_slice = (slice(1, -1),) * len(world_size)

        self.world = self.full[nd_slice]
        self.n_dims = len(self.world.shape)
        self.sum_over = tuple(-(i + 1) for i in range(self.n_dims))

        # a tree will burn if at least one neighbor is burning
        self.fire_rule_tree = np.ones(9 * self.FIRE_CELL, np.uint16)
        self.fire_rule_tree[9:] = self.FIRE_CELL

        # create a discrete colormap for rendering
        self.cmap = colors.ListedColormap(['white', 'green', 'red'])
        self.norm = colors.BoundaryNorm([0, 1, 2, 3], self.cmap.N)

    def step(self):
        neighborhoods = get_neighborhoud(self.full)
        neighbor_ct = np.sum(neighborhoods, self.sum_over) - self.world

        fire = self.world == self.FIRE_CELL
        tree = self.world == self.TREE_CELL
        empty = self.world == self.EMPTY_CELL

        # Apply the update rules:
        # 1. A burning cell turns into an empty cell
        self.world[fire] = self.EMPTY_CELL

        # 2. A tree will burn if at least one neighbor is burning
        self.world[tree] = self.fire_rule_tree[neighbor_ct][tree]

        # 3. A tree ignites with probability f even if no neighbor is burning
        ignition_cells = np.random.random(self.world.shape) < self.p_fire
        self.world[np.logical_and(tree, ignition_cells)] = self.FIRE_CELL

        # 4. An empty space fills with a tree with probability p
        grow_cells = np.random.random(self.world.shape) < self.p_tree
        self.world[np.logical_and(empty, grow_cells)] = self.TREE_CELL

    def render(self, pause_time=0.0001):
        plt.imshow(self.world, cmap=self.cmap, norm=self.norm)
        plt.xticks([])
        plt.yticks([])
        plt.ion()
        plt.draw()
        plt.pause(pause_time)


if __name__ == '__main__':
    forest = Forest(world_size=(100, 100))
    forest.world[10:30, 10:30] = 1
    n_steps = 100
    start = time.time()

    for t in range(n_steps):
        forest.step()
        forest.render()

    print("Time elapsed for {}: {} seconds".format(n_steps, time.time() - start))
