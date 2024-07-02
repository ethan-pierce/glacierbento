"""Test the SimpleGlacialEroder component."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import jax.numpy as jnp
import equinox as eqx
from landlab import RasterModelGrid
from glacierbento import Field
from glacierbento.utils import freeze_grid
from glacierbento.components import SimpleGlacialEroder
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
    sliding = jnp.ones(grid.number_of_nodes) * grid.node_x / 20 / 31556926
    return {
        'sliding_velocity': Field(sliding, 'm/s', 'node'),
        'till_thickness': Field(jnp.zeros(grid.number_of_nodes), 'm', 'node')
    }

@pytest.fixture
def eroder(grid):
    static = freeze_grid(grid)
    return SimpleGlacialEroder(static)

def test_initialize(eroder):
    assert 'rate_coefficient' in eroder.params.keys()
    assert 'sliding_exponent' in eroder.params.keys()

def test_run_one_step(eroder, grid, fields):

    @eqx.filter_jit
    def update(fields):
        output = eroder.run_one_step(31556926, fields)
        return output

    for i in range(10):
        output = update(fields)
        fields = eqx.tree_at(
            lambda t: t['till_thickness'].value,
            fields,
            output['till_thickness'].value
        )

    assert 'erosion_rate' in output
    assert 'till_thickness' in output
