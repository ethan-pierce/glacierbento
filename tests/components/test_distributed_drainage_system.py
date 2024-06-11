"""Test the DistributedDrainageSystem Component."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
import jax.numpy as jnp
import equinox as eqx
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

@pytest.fixture
def dds(grid):
    ice_thickness = 5 * grid.node_x**(1/2) # from 150 m to 0 m
    bed_elevation = np.zeros(grid.number_of_nodes) + 1 # 1 m
    sliding_velocity = np.ones(grid.number_of_nodes) / 31556926 # 1 m / a
    melt_input = np.ones(grid.number_of_nodes) * 0.05 / 31556926 # 0.05 m / a

    return DistributedDrainageSystem(
        grid, 
        {
            'ice_thickness': ice_thickness,
            'bed_elevation': bed_elevation,
            'sliding_velocity': sliding_velocity,
            'melt_input': melt_input
        }
    )

def test_init(dds):
    assert_array_equal(
        dds._fields['hydraulic_potential'].value, 
        917 * 9.81 * dds._fields['ice_thickness'].value * 0.8
    )
    assert_array_equal(
        dds._fields['sheet_thickness'].value, 
        np.zeros(dds._grid.number_of_nodes)
    )

def test_calc_pressure(dds):
    N = dds.calc_effective_pressure(dds._fields['hydraulic_potential'].value)

    Pi = 917 * 9.81 * dds._fields['ice_thickness'].value
    Pw = dds._fields['hydraulic_potential'].value - 1000 * 9.81 * dds._fields['bed_elevation'].value

    assert_array_equal(N, Pi - Pw)

def test_calc_dhdt(dds):
    dhdt = dds._calc_dhdt(dds._fields['hydraulic_potential'].value, dds._fields['sheet_thickness'].value)

    assert_array_almost_equal(dhdt, np.full(dds._grid.number_of_nodes, 1.584e-9), decimal = 12)


