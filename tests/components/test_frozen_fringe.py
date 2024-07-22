"""Test the FrozenFringe component."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import jax.numpy as jnp
import equinox as eqx
from landlab import RasterModelGrid
from glacierbento import Field
from glacierbento.utils import freeze_grid
from glacierbento.components import FrozenFringe
import matplotlib.pyplot as plt
from landlab.plot import imshow_grid

def test_import():
    assert True

@pytest.fixture
def grid():
    rmg = RasterModelGrid((10, 10), xy_spacing = 100)
    return rmg

@pytest.fixture
def fields(grid):
    effective_pressure = (500 * grid.node_x + 1)**(1/2) * 0.8 * 9.81 * 917
    basal_melt = (np.max(grid.node_x) - grid.node_x) / 1000 / 31556926 * 0.5
    fringe_thickness = np.zeros(grid.number_of_nodes) + 1e-3

    return {
        'effective_pressure': Field(effective_pressure, 'Pa', 'node'),
        'basal_melt_rate': Field(basal_melt, 'm/s', 'node'),
        'till_thickness': Field(np.zeros(grid.number_of_nodes) + 1.0, 'm', 'node'),
        'fringe_thickness': Field(fringe_thickness, 'm', 'node')
    }

@pytest.fixture
def model(grid):
    static = freeze_grid(grid)
    return FrozenFringe(static)

def test_run_one_step(model, grid, fields):

    @eqx.filter_jit
    def update(fields):
        output = model.run_one_step(60.0, fields)
        return output
    
    for i in range(10):
        output = update(fields)
        
        fields = eqx.tree_at(
            lambda t: (t['fringe_thickness'], t['till_thickness']), 
            fields, 
            (output['fringe_thickness'], output['till_thickness'])
        )
    
    hf = output['fringe_thickness'].value
    S = output['fringe_saturation'].value
    theta = output['fringe_undercooling'].value
    ht = output['till_thickness'].value

    imshow_grid(grid, hf)
    plt.show()

    imshow_grid(grid, S)
    plt.show()

    imshow_grid(grid, theta)
    plt.show()

    imshow_grid(grid, ht)
    plt.show()