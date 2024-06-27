"""Provides utility routines for assembling finite volume matrices."""

import jax
import jax.numpy as jnp
import equinox as eqx
from glacierbento.utils import StaticGrid

class MatrixAssembler:
    """Assembles a matrix of finite volume coefficients for an elliptic problem.

    Can be extended in the future to support different discretizations and
    problem types, but for now this is the only problem we need it for. The only
    supported boundary conditions are fixed-value (Dirichlet) and no-flux (Neumann).
    If a boundary node is not specified positively as a fixed-value boundary, it
    is assumed to be a no-flux boundary.

    Attributes:
        grid: A StaticGrid object representing the computational mesh.
        coeffs: An array of flux coefficients defined on grid faces.
        forcing: An vector of forcing terms defined on grid cells.
        boundary_values: An array of values for fixed-value boundary nodes.
        is_fixed_value: A boolean array indicating fixed-value boundary nodes.

    Methods:
        assemble_matrix: Assembles the finite volume matrix. Returns a (n_cells, n_cells) array.
    """

    def __init__(
        self,
        grid: StaticGrid,
        coeffs: jnp.ndarray,
        forcing: jnp.ndarray,
        boundary_values: jnp.ndarray,
        is_fixed_value: jnp.ndarray
    ):
        """Initialize the MatrixAssembler with grid and boundary conditions."""
        self.grid = grid
        self.coeffs = jnp.asarray(coeffs)
        self.forcing = jnp.asarray(forcing)
        self.boundary_values = jnp.asarray(boundary_values)
        self.is_fixed_value = jnp.asarray(is_fixed_value)

    def get_adjacent_node(self, cell, link):
        """Get the node adjacent to a given node along a given link."""
        return jnp.where(
            self.grid.node_at_link_head[link] == self.grid.node_at_cell[cell],
            self.grid.node_at_link_tail[link],
            self.grid.node_at_link_head[link]
        )

    def add_dirichlet(self, args):
        """Add coefficients from a link that extends to a fixed-value node."""
        cell, link, forcing, row = args
        adj_node = self.get_adjacent_node(cell, link)
        face = self.grid.face_at_link[link]
        row = row.at[cell].add(-self.coeffs[face])
        forcing = forcing.at[cell].add(-self.coeffs[face] * self.boundary_values[adj_node])
        return forcing, row

    def add_interior(self, args):
        """Add coefficients from a link that extends to an interior node."""
        cell, link, forcing, row = args
        adj_node = self.get_adjacent_node(cell, link)
        adj_cell = self.grid.cell_at_node[adj_node]
        face = self.grid.face_at_link[link]
        row = row.at[cell].add(-self.coeffs[face])
        row = row.at[adj_cell].add(self.coeffs[face])
        return forcing, row

    def add_valid(self, args):
        """Given a link, dispatch the correct row modification."""
        cell, link, forcing, row = args
        forcing, row = jax.lax.cond(
            self.grid.status_at_node[self.get_adjacent_node(cell, link)] == 0,
            lambda args: self.add_interior(args),
            lambda args: jax.lax.cond(
                self.is_fixed_value[self.get_adjacent_node(cell, link)] == 1,
                lambda args: self.add_dirichlet(args),
                lambda args: (forcing, row),
                args
            ),
            args
        )
        return forcing, row

    def assemble_row(self, forcing, cell):
        """Assemble one row of the matrix."""
        row = jnp.zeros(self.grid.number_of_cells)
        node = self.grid.node_at_cell[cell]
        for link in self.grid.links_at_node[node]:
            args = (cell, link, forcing, row)
            forcing, row = jax.lax.cond(
                link == -1,
                lambda args: (forcing, row),
                lambda args: self.add_valid(args),
                args
            )

        return forcing, row

    def assemble_matrix(self) -> jnp.ndarray:
        """Assembles the finite volume matrix."""
        forcing, matrix = jax.lax.scan(
            self.assemble_row, 
            self.forcing, 
            jnp.arange(self.grid.number_of_cells)
        )

        return forcing, matrix