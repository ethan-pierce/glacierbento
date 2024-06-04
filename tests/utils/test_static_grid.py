"""Test the StaticGrid utility."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from landlab import RasterModelGrid, HexModelGrid
from glacierbento.utils.static_grid import freeze_grid, StaticGrid

@pytest.fixture
def grid():
    return RasterModelGrid((3, 3))

def test_freeze_grid(grid):
    """Test that freeze_grid creates a StaticGrid object."""
    static = freeze_grid(grid)
    
    assert isinstance(static, StaticGrid)

@pytest.fixture
def static_grid(grid):
    return freeze_grid(grid)

def test_map_mean_of_links_to_node(grid, static_grid):
    """Test the map_mean_of_links_to_node method."""
    array = np.arange(grid.number_of_links)
    result = static_grid.map_mean_of_links_to_node(array)
    expected_result = grid.map_mean_of_links_to_node(array)
    assert_array_equal(result, expected_result)

def test_map_mean_of_link_nodes_to_link(grid, static_grid):
    """Test the map_mean_of_link_nodes_to_link method."""
    array = np.arange(grid.number_of_nodes)
    result = static_grid.map_mean_of_link_nodes_to_link(array)
    expected_result = grid.map_mean_of_link_nodes_to_link(array)
    assert_array_equal(result, expected_result)

def test_map_vectors_to_links(grid, static_grid):
    """Test the map_vectors_to_links method."""
    x_component = np.arange(grid.number_of_nodes)
    y_component = np.arange(grid.number_of_nodes) * -1
    
    result = static_grid.map_vectors_to_links(x_component, y_component)

    x_component = grid.map_mean_of_link_nodes_to_link(x_component)
    y_component = grid.map_mean_of_link_nodes_to_link(y_component)

    expected_result = grid.map_vectors_to_links(x_component, y_component)
    
    assert_array_equal(result, expected_result)

def test_map_value_at_max_node_to_link(grid, static_grid):
    """Test the map_value_at_max_node_to_link method."""
    controls = np.arange(grid.number_of_nodes)
    values = np.arange(grid.number_of_nodes) + 10

    result = static_grid.map_value_at_max_node_to_link(controls, values)
    expected_result = grid.map_value_at_max_node_to_link(controls, values)

    assert_array_equal(result, expected_result)

    constant_controls = np.ones(grid.number_of_nodes)
    result = static_grid.map_value_at_max_node_to_link(constant_controls, values)
    expected_result = grid.map_value_at_max_node_to_link(constant_controls, values)

    assert_array_equal(result, expected_result)

def test_map_mean_of_patch_nodes_to_patch(grid, static_grid):
    """Test the map_mean_of_patch_nodes_to_patch method."""
    array = np.arange(grid.number_of_nodes)
    result = static_grid.map_mean_of_patch_nodes_to_patch(array)
    expected_result = grid.map_mean_of_patch_nodes_to_patch(array)
    assert_array_equal(result, expected_result)

def test_sum_at_nodes(grid, static_grid):
    """Test the sum_at_nodes method."""
    array = np.arange(grid.number_of_links)
    result = static_grid.sum_at_nodes(array)

    # Take the difference, because outlinks are negative fluxes by convention
    expected_result = grid.map_sum_of_inlinks_to_node(array) - grid.map_sum_of_outlinks_to_node(array)
    assert_array_equal(result, expected_result)

def test_calc_unit_normal_at_patch():
    """Test the calc_unit_normal_at_patch method."""
    hexgrid = HexModelGrid((3, 3))
    static_hexgrid = freeze_grid(hexgrid)

    array = np.arange(hexgrid.number_of_nodes)
    result = static_hexgrid.calc_unit_normal_at_patch(array)
    expected_result = hexgrid.calc_unit_normal_at_patch(array)
    assert_array_almost_equal(result, expected_result)

def test_calc_grad_at_patch():
    """Test the calc_grad_at_patch method."""
    hexgrid = HexModelGrid((3, 3))
    static_hexgrid = freeze_grid(hexgrid)

    array = np.arange(hexgrid.number_of_nodes)
    slope, comps = static_hexgrid.calc_grad_at_patch(array)
    x_comps, y_comps = hexgrid.calc_grad_at_patch(array)
    expected_comps = np.stack([x_comps, y_comps])

    assert_array_almost_equal(comps, expected_comps)

    expected_slope = hexgrid.calc_slope_at_patch(array)

    assert_array_almost_equal(slope, expected_slope)

def test_calc_grad_at_link(grid, static_grid):
    """Test the calc_grad_at_link method."""
    array = np.arange(grid.number_of_nodes)
    result = static_grid.calc_grad_at_link(array)
    expected_result = grid.calc_grad_at_link(array)
    assert_array_almost_equal(result, expected_result)

def test_calc_flux_div_at_node(grid, static_grid):
    """Test the calc_flux_div_at_node method."""
    array = np.arange(grid.number_of_links)
    result = static_grid.calc_flux_div_at_node(array)
    expected_result = grid.calc_flux_div_at_node(array)
    assert_array_almost_equal(result, expected_result)

def test_calc_gradient_vector_at_node(grid, static_grid):
    """Test the calc_gradient_vector_at_node method."""
    array = np.arange(grid.number_of_nodes)
    magnitude, components = static_grid.calc_gradient_vector_at_node(array)

    expected_magnitude, expected_comps = grid.calc_slope_at_node(array, return_components = True)
    expected_comps_transpose = np.stack(expected_comps).T

    assert_array_almost_equal(magnitude, expected_magnitude)
    assert_array_almost_equal(components, expected_comps_transpose)