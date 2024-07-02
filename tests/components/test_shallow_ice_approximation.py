"""Test the ShallowIceApproximation component."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import jax.numpy as jnp
import equinox as eqx
from landlab import RasterModelGrid
from glacierbento import Field
from glacierbento.utils import freeze_grid
from glacierbento.components import ShallowIceApproximation
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
    ice_thickness = (500 * grid.node_x + 1)**(1/2)
    surface_elevation = ice_thickness + grid.node_x * 1e-2
    return {
        'ice_thickness': Field(ice_thickness, 'm', 'node'),
        'surface_elevation': Field(surface_elevation, 'm', 'node')
    }

@pytest.fixture
def sia(grid):
    static = freeze_grid(grid)
    return ShallowIceApproximation(static)

def test_run_one_step(sia, grid, fields):

    @eqx.filter_jit
    def update(fields):
        output = sia.run_one_step(0.0, fields)
        return output
    
    output = update(fields)
    velocity = grid.map_mean_of_links_to_node(output['deformation_velocity'].value)

    imshow_grid(grid, velocity)
    plt.show()