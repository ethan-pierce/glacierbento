"""Test the DistributedDrainageSystem Component."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from landlab import RasterModelGrid
from glacierbento.utils import freeze_grid
from glacierbento.components import DistributedDrainageSystem

def test_import():
    assert True

@pytest.fixture
def grid():
    rmg = RasterModelGrid((3, 10), xy_spacing = 100)
    static = freeze_grid(rmg)
    return static

def test_init(grid):
    ice_thickness = 5 * grid.node_x**(1/2) # from 150 m to 0 m
    bed_elevation = np.zeros(grid.number_of_nodes) # 0 m
    sliding_velocity = np.ones(grid.number_of_nodes) / 31556926 # 1 m / a
    melt_input = np.ones(grid.number_of_nodes) * 0.05 / 31556926 # 0.05 m / a

    dds = DistributedDrainageSystem(
        grid, 
        {
            'ice_thickness': ice_thickness,
            'bed_elevation': bed_elevation,
            'sliding_velocity': sliding_velocity,
            'melt_input': melt_input
        }
    )

    assert dds.grid == grid
    
