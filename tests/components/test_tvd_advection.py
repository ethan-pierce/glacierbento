"""Test the TVDAdvector component."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import jax.numpy as jnp
import equinox as eqx
from landlab import RasterModelGrid
from glacierbento import Field
from glacierbento.utils import freeze_grid
from glacierbento.components import TVDAdvector
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
    tracer = np.zeros(grid.number_of_nodes)
    tracer[22] = 1

    return {
        'velocity_x': Field(np.ones(grid.number_of_nodes) * 20, 'm/s', 'node'),
        'velocity_y': Field(np.ones(grid.number_of_nodes) * 10, 'm/s', 'node'),
        'tracer': Field(tracer, 'kg/m^3', 'node')
    }

@pytest.fixture
def tvd(grid):
    static = freeze_grid(grid)
    return TVDAdvector(static, fields_to_advect = ['tracer'])

def test_initialize(tvd, fields):
    tvd = tvd.initialize(fields)

    assert tvd._link_velocity.shape == (tvd._grid.number_of_links,)
    assert tvd._upwind_ghost_nodes.shape == (tvd._grid.number_of_links, 2)
    assert tvd._real_indices.shape == (tvd._grid.number_of_links,)
    assert tvd._real_xy.shape == (tvd._grid.number_of_links, 2)

    assert np.mean(tvd._link_velocity) == 15

def test_run_one_step(tvd, fields, grid):
    tvd = tvd.initialize(fields)

    @eqx.filter_jit
    def update(fields):
        output = tvd.run_one_step(1.0, fields)
        return output

    for i in range(10):
        output = update(fields)
        fields = eqx.tree_at(
            lambda t: t['tracer'].value,
            fields,
            output['tracer_advected'].value
        )

    assert 'tracer_advected' in output
    
    imshow_grid(grid, output['tracer_advected'].value)
    plt.show()