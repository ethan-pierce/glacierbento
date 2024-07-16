"""Test the DispersedLayer component."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import jax.numpy as jnp
import equinox as eqx
from landlab import RasterModelGrid
from glacierbento import Field
from glacierbento.utils import freeze_grid
from glacierbento.components import DispersedLayer, FrozenFringe
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
        'fringe_thickness': Field(fringe_thickness, 'm', 'node')
    }

@pytest.fixture
def ff(grid):
    static = freeze_grid(grid)
    return FrozenFringe(static)

@pytest.fixture
def all_fields(grid, ff, fields):
    for i in range(5):
        output = ff.run_one_step(60.0, fields)
        fields = eqx.tree_at(lambda t: t['fringe_thickness'], fields, output['fringe_thickness'])
    
    theta = output['fringe_undercooling'].value
    base_temperature = ff.params['base_temperature']
    dispersed_thickness = np.zeros(grid.number_of_nodes) + 1e-3

    return {
        'theta': Field(theta, '', 'node'),
        'base_temperature': Field(base_temperature, 'K', 'node'),
        'dispersed_thickness': Field(dispersed_thickness, 'm', 'node')
    }

@pytest.fixture
def model(grid):
    static = freeze_grid(grid)
    return DispersedLayer(static)

def test_run_one_step(model, grid, all_fields):
    fields = all_fields

    @eqx.filter_jit
    def update(fields):
        output = model.run_one_step(60.0, fields)
        return output
    
    for i in range(10):
        output = update(fields)
        
        fields = eqx.tree_at(
            lambda t: t['dispersed_thickness'], fields, output['dispersed_thickness']
        )
    
    hd = output['dispersed_thickness'].value
    G = output['temperature_gradient'].value
    R = output['regelation_rate'].value

    imshow_grid(grid, hd)
    plt.show()

    imshow_grid(grid, G)
    plt.show()

    imshow_grid(grid, R)
    plt.show()
