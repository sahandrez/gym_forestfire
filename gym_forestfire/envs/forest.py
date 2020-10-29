"""
Forest-fire simulation based on the rossel and
Schwabl (1992) model. The vectorized implementation
of cellular automaton was adapted from:
http://drsfenner.org/blog/2015/08/game-of-life-in-numpy-2/

Author: Sahand Rezaei-Shoshtari
Copyright protected.
"""

import numpy as np
import cv2
import time
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

    def __init__(self, world_size=(128, 128), p_fire=0.4,
                 p_ignition=0.00001, p_tree=0.00001,
                 init_tree=0.8, extinguisher_ratio=0.05):
        """
        :param world_size: Size of the world.
        :param p_fire: Probability of fire spreading from a burning tree to the neighboring trees.
        :param p_ignition: Probability of catching fire even if no neighbor is burning.
        :param p_tree: Probability of a tree growing in an empty space.
        :param init_tree: Initial probability of tree distribution.
        :param extinguisher_ratio: Ratio of the size of the fire extinguisher wrt to world size.
        """
        self.EMPTY_CELL = 0
        self.TREE_CELL = 1
        self.FIRE_CELL = 10

        self.p_fire = p_fire
        self.p_ignition = p_ignition
        self.p_tree = p_tree
        self.p_init_tree = init_tree
        self.extinguisher_ratio = extinguisher_ratio

        full_size = tuple(i + 2 for i in world_size)
        self.full = np.zeros(full_size, dtype=np.uint8)
        nd_slice = (slice(1, -1),) * len(world_size)

        self.world = self.full[nd_slice]
        self.n_dims = len(self.world.shape)
        self.sum_over = tuple(-(i + 1) for i in range(self.n_dims))

        # a tree will burn if at least one neighbor is burning
        self.fire_rule_tree = np.zeros(9 * self.FIRE_CELL, np.uint16)
        self.fire_rule_tree[9:] = self.FIRE_CELL

    def step(self, action=None):
        # if the action has been aimed at fire, returns true
        aimed_fire = False

        neighborhoods = get_neighborhoud(self.full)
        neighbor_ct = np.sum(neighborhoods, self.sum_over) - self.world

        self.fire = self.world == self.FIRE_CELL
        self.tree = self.world == self.TREE_CELL
        self.empty = self.world == self.EMPTY_CELL
        is_fire = np.any(self.fire)

        # action is a normalized 2D vector with [x. y] as the center
        # of the square of applying fire extinguishers
        if action is not None:
            x, y = int(self.world.shape[1] * action[1]), int(self.world.shape[0] * action[0])
            w, h = int(self.world.shape[1] * self.extinguisher_ratio), int(self.world.shape[0] * self.extinguisher_ratio)
            aimed_fire = np.any(self.fire[x - w:x + w, y - h:y + h])
            self.fire[x-w:x+w, y-h:y+h] = False

        # Apply the update rules:
        # 1. A burning cell turns into an empty cell
        self.world[self.fire] = self.EMPTY_CELL

        # 2. A tree will burn if at least one neighbor is burning
        fire_propagation_prop = np.random.random(self.world.shape) < self.p_fire
        fire_cells = np.logical_and(self.fire_rule_tree[neighbor_ct], fire_propagation_prop)
        self.world[np.logical_and(self.tree, fire_cells)] = self.FIRE_CELL

        # 3. A tree ignites with probability f even if no neighbor is burning
        ignition_cells = np.random.random(self.world.shape) < self.p_ignition
        self.world[np.logical_and(self.tree, ignition_cells)] = self.FIRE_CELL

        # 4. An empty space fills with a tree with probability p
        grow_cells = np.random.random(self.world.shape) < self.p_tree
        self.world[np.logical_and(self.empty, grow_cells)] = self.TREE_CELL

        return aimed_fire, is_fire

    def reset(self):
        tree_cells = np.random.random(self.world.shape) < self.p_init_tree
        self.world[tree_cells] = self.TREE_CELL
        self.world[np.logical_not(tree_cells)] = self.EMPTY_CELL

    def render(self):
        im = cv2.cvtColor(self.world, cv2.COLOR_GRAY2BGR)
        im[self.tree, 1] = 255
        im[self.fire, 2] = 255
        im = cv2.resize(im, (640, 640))
        cv2.imshow("Forest", im)
        cv2.waitKey(50)


if __name__ == '__main__':
    forest = Forest(world_size=(128, 128))
    forest.reset()
    n_steps = 20
    start = time.time()

    for t in range(n_steps):
        forest.step()
        forest.render()

    print("Time elapsed for {} steps: {} seconds".format(n_steps, time.time() - start))
