"""Test the MatrixAssembler utility."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from landlab import RasterModelGrid
from glacierbento.utils import freeze_grid, MatrixAssembler

@pytest.fixture
def grid():
    rmg = RasterModelGrid((5, 5), xy_spacing = (10, 10))
    return freeze_grid(rmg)

@pytest.fixture
def assembler(grid):
    coeffs = np.full(grid.number_of_faces, -1.0)
    boundary_values = np.full(grid.number_of_nodes, 1.0)
    is_fixed_value = np.zeros(grid.number_of_nodes, dtype=bool)
    is_fixed_gradient = np.ones(grid.number_of_nodes, dtype=bool)
    is_fixed_value[:5] = 1
    is_fixed_gradient[:5] = 0
    forcing = np.zeros(grid.number_of_cells)
    return MatrixAssembler(grid, coeffs, forcing, boundary_values, is_fixed_value)

def test_get_adjacent_node(grid, assembler):
    assert assembler.get_adjacent_node(0, 5) == 1
    assert assembler.get_adjacent_node(0, 9) == 5
    assert assembler.get_adjacent_node(0, 10) == 7
    assert assembler.get_adjacent_node(0, 14) == 11

def test_assemble_matrix(grid, assembler):
    forcing, matrix = assembler.assemble_matrix()
    print(forcing)
    print(matrix)

    assert forcing.shape == (grid.number_of_cells,)
    expected_forcing = np.zeros(grid.number_of_cells)
    expected_forcing[:3] = 1
    # assert_array_equal(forcing, expected_forcing)

    assert matrix.shape == (grid.number_of_cells, grid.number_of_cells)
    assert_array_equal(
        (matrix != 0).astype(int),
        np.array([
            [1, 1, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 1, 0, 1, 1, 1, 0, 1, 0],
            [0, 0, 1, 0, 1, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 1, 1]
        ])
    )
