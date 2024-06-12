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
    init_potential = 917 * 9.81 * ice_thickness * 0.2 + 1000 * 9.81 * bed_elevation
    init_sheets = np.zeros(grid.number_of_nodes)

    return DistributedDrainageSystem(
        grid, 
        {
            'ice_thickness': ice_thickness,
            'bed_elevation': bed_elevation,
            'sliding_velocity': sliding_velocity,
            'melt_input': melt_input,
            'hydraulic_potential': init_potential,
            'sheet_flow_height': init_sheets
        }
    )

def test_calc_pressure(dds):
    N = dds.calc_effective_pressure(dds._fields['hydraulic_potential'].value)

    Pi = 917 * 9.81 * dds._fields['ice_thickness'].value
    Pw = dds._fields['hydraulic_potential'].value - 1000 * 9.81 * dds._fields['bed_elevation'].value

    assert_array_equal(N, Pi - Pw)

def test_calc_discharge(dds):
    Q = dds.calc_discharge(
        dds._fields['hydraulic_potential'].value,
        jnp.ones(dds._grid.number_of_nodes) * 1e-3
    )

    assert_array_almost_equal(Q[:3], [-0.0364, -0.0026, -0.0012], decimal = 4)

def test_run_one_step(dds):
    dds = eqx.tree_at(
        lambda t: t._fields['bed_elevation'].value,
        dds,
        dds._fields['bed_elevation'].value.at[10].set(0)
    )

    for i in range(4):
        dds = dds.run_one_step(1)

    print(dds.get_output())
