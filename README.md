# Gym Forest Fire
Forest fire simulation and [Gym](https://github.com/openai/gym) environment
for tackling wildfires with reinforcement learning. The simulation largely follows the
[forest-fire model](https://en.wikipedia.org/wiki/Forest-fire_model) of 
[Drossel and Schwabl (1992)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.69.1629)
which defines the forest fire model as a cellular automaton on a grid with L<sup>d</sup> 
cells, where L is the sidelength of the grid and d is its dimension. 

A cell can have three states: empty, occupied by a tree or burning. The Drossel and 
Schwabl (1992) model is then defined by four rules executed simultaneously:
1. A burning cell turns into an empty cell.
1. A tree will burn with probability q if at least one neighbor is burning.
1. A tree ignites with probability f even if no neighbor is burning.
1. An empty space fills with a tree with probability p.

## Dependencies 
This code has very few dependencies and the simulation is fully vectorized for faster
computation.  

Requirements:
* Python 3.5+
* OpenAI Gym
* Numpy 
* OpenCV (for rendering the environment)

## Instructions
Clone the package and install it using `pip`. This install the package and all its 
requirements.
```
From ~/gym_forestfire/
pip install -e .
```

## Refrences 
1. [Drossel B, Schwabl F. Self-organized critical forest-fire model. Physical review letters. 1992 Sep 14;69(11):1629.](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.69.1629)
2. [Brockman G, Cheung V, Pettersson L, Schneider J, Schulman J, Tang J, Zaremba W. Openai gym. arXiv preprint arXiv:1606.01540. 2016 Jun 5.](https://arxiv.org/abs/1606.01540)
3. [Rolnick D, Donti PL, Kaack LH, Kochanski K, Lacoste A, Sankaran K, Ross AS, Milojevic-Dupont N, Jaques N, Waldman-Brown A, Luccioni A. Tackling climate change with machine learning. arXiv preprint arXiv:1906.05433. 2019 Jun 10.
](https://arxiv.org/abs/1906.05433)
4. [Blog post: Game of Life in NumPy. URL: http://drsfenner.org/blog/2015/08/game-of-life-in-numpy-2/](http://drsfenner.org/blog/2015/08/game-of-life-in-numpy-2/)
