"""Immutable grid object that represents a computational mesh.

The StaticGrid is designed to have a similar interface as standard Landlab grid
types (e.g., RasterModelGrid), but to only encapsulate the functionality of a
Landlab Graph object (e.g., DualGraph). The StaticGrid is immutable, so that it
can be used by Components within JIT-compiled update functions. Fields, instead 
of being stored in the Grid, as in Landlab, are instead handled by Model objects.

The StaticGrid should never be instantiated directly. Instead, use the utility
function, 'freeze_grid', to convert a previously instantiated Landlab grid into
a StaticGrid, which will ensure that the grid connectivity is correct.

Typical usage example:
    from landlab import RasterModelGrid

    rmg = RasterModelGrid((3, 3), xy_spacing = 1.0)
    static_grid = freeze_grid(rmg)
"""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import dataclasses
from landlab import ModelGrid


def freeze_grid(grid: ModelGrid):
    """Convert an existing Landlab grid to a new StaticGrid.
    
    Args:
        grid: A Landlab grid object.
    
    Returns:
        A new StaticGrid object, with the same geometry as the input grid.
    """
    fields = {
        field.name: getattr(grid, field.name) for field in dataclasses.fields(StaticGrid)
    }
    return StaticGrid(**fields)


class StaticGrid(eqx.Module):
    """Stores a computational mesh and provides methods to map between elements.
    
    See also: 
        freeze_grid(ModelGrid): a utility that converts a Landlab grid to a StaticGrid.
    """

    # Density of grid elements
    number_of_nodes: int = eqx.field(converter = int)
    number_of_links: int = eqx.field(converter = int)
    number_of_patches: int = eqx.field(converter = int)
    number_of_corners: int = eqx.field(converter = int)
    number_of_faces: int = eqx.field(converter = int)
    number_of_cells: int = eqx.field(converter = int)

    # Locations and sizes of grid elements
    node_x: jax.Array = eqx.field(converter = jnp.asarray)
    node_y: jax.Array = eqx.field(converter = jnp.asarray)
    midpoint_of_link: jax.Array = eqx.field(converter = jnp.asarray)
    length_of_link: jax.Array = eqx.field(converter = jnp.asarray)
    angle_of_link: jax.Array = eqx.field(converter = jnp.asarray)
    xy_of_patch: jax.Array = eqx.field(converter = jnp.asarray)
    area_of_patch: jax.Array = eqx.field(converter = jnp.asarray)
    x_of_corner: jax.Array = eqx.field(converter = jnp.asarray)
    y_of_corner: jax.Array = eqx.field(converter = jnp.asarray)
    length_of_face: jax.Array = eqx.field(converter = jnp.asarray)
    cell_area_at_node: jax.Array = eqx.field(converter = jnp.asarray)

    # Information about boundary conditions
    status_at_node: jax.Array = eqx.field(converter = jnp.asarray)
    core_nodes: jax.Array = eqx.field(converter = jnp.asarray)
    boundary_nodes: jax.Array = eqx.field(converter = jnp.asarray)
    status_at_link: jax.Array = eqx.field(converter = jnp.asarray)
    active_links: jax.Array = eqx.field(converter = jnp.asarray)

    # Mapping between grid elements
    adjacent_nodes_at_node: jax.Array = eqx.field(converter = jnp.asarray)
    node_at_link_head: jax.Array = eqx.field(converter = jnp.asarray)
    node_at_link_tail: jax.Array = eqx.field(converter = jnp.asarray)
    links_at_node: jax.Array = eqx.field(converter = jnp.asarray)
    link_dirs_at_node: jax.Array = eqx.field(converter = jnp.asarray)
    patches_at_node: jax.Array = eqx.field(converter = jnp.asarray)
    nodes_at_patch: jax.Array = eqx.field(converter = jnp.asarray)
    cell_at_node: jax.Array = eqx.field(converter = jnp.asarray)
    node_at_cell: jax.Array = eqx.field(converter = jnp.asarray)
    links_at_patch: jax.Array = eqx.field(converter = jnp.asarray)
    patches_at_link: jax.Array = eqx.field(converter = jnp.asarray)
    face_at_link: jax.Array = eqx.field(converter = jnp.asarray)
    link_at_face: jax.Array = eqx.field(converter = jnp.asarray)
    corners_at_face: jax.Array = eqx.field(converter = jnp.asarray)

    def mask_links_at_node_array(self, array):
        """Map an array of values at links to nodes, masking out invalid values."""
        return jnp.where(self.links_at_node != -1, array[self.links_at_node], jnp.nan)

    def map_mean_of_links_to_node(self, array):
        """Map the mean of values at links to nodes."""
        masked_values = self.mask_links_at_node_array(array)
        return jnp.nanmean(masked_values, axis = 1)

    def map_mean_of_link_nodes_to_link(self, array):
        """Map the mean of values at nodes to links."""
        return 0.5 * (array[self.node_at_link_head] + array[self.node_at_link_tail])

    def map_vectors_to_links(self, x_component, y_component):
        """Map magnitude and sign of vectors with components (xcomp, ycomp) onto grid links."""
        ux_at_link = self.map_mean_of_link_nodes_to_link(x_component)
        uy_at_link = self.map_mean_of_link_nodes_to_link(y_component)

        magnitude = jnp.sqrt(ux_at_link**2 + uy_at_link**2)
        vector_angle = jnp.arctan2(uy_at_link, ux_at_link)
        mapped_angle = vector_angle - self.angle_of_link
        
        return magnitude * jnp.cos(mapped_angle)

    def resolve_values_on_links(self, array):
        """Resolve an array defined on links to its x-and-y components."""
        x_vals = (
            (self.node_x[self.node_at_link_head] - self.node_x[self.node_at_link_tail])
            / self.length_of_link 
            * array
        )

        y_vals = (
            (self.node_y[self.node_at_link_head] - self.node_y[self.node_at_link_tail])
            / self.length_of_link 
            * array
        )
        
        return (x_vals, y_vals)

    def map_value_at_max_node_to_link(self, controls, values):
        """Map values field to links based on the maximum value of the controls field."""
        ctrl_heads = controls[self.node_at_link_head]
        ctrl_tails = controls[self.node_at_link_tail]
        val_heads = values[self.node_at_link_head]
        val_tails = values[self.node_at_link_tail]

        return jnp.where(
            ctrl_tails > ctrl_heads, val_tails, val_heads
        )

    def map_mean_of_patch_nodes_to_patch(self, array):
        """Map the mean of values at nodes to patches."""
        return jnp.mean(array[self.nodes_at_patch], axis = 1)

    def sum_at_nodes(self, array):
        """Sum values at links to nodes, considering the direction of each link."""
        masked_values = self.mask_links_at_node_array(array)
        return jnp.nansum(self.link_dirs_at_node * masked_values, axis = 1)

    def calc_unit_normal_at_patch(self, array):
        """Calculate the three-dimensional unit normal vector to each patch."""
        
        # Each patch is defined by three nodes, PQR
        vector_PQ = jnp.column_stack(
            [
                self.node_x[self.nodes_at_patch[:, 1]] - self.node_x[self.nodes_at_patch[:, 0]],
                self.node_y[self.nodes_at_patch[:, 1]] - self.node_y[self.nodes_at_patch[:, 0]],
                array[self.nodes_at_patch[:, 1]] - array[self.nodes_at_patch[:, 0]]
            ]
        )
        vector_PR = jnp.column_stack(
            [
                self.node_x[self.nodes_at_patch[:, 2]] - self.node_x[self.nodes_at_patch[:, 0]],
                self.node_y[self.nodes_at_patch[:, 2]] - self.node_y[self.nodes_at_patch[:, 0]],
                array[self.nodes_at_patch[:, 2]] - array[self.nodes_at_patch[:, 0]]
            ]
        )

        normal_hat = jnp.cross(vector_PQ, vector_PR)
        normal_mag = jnp.sqrt(jnp.square(normal_hat).sum(axis = 1))

        return jnp.divide(normal_hat, normal_mag[:, None])

    def calc_grad_at_patch(self, array):
        """Calculate the gradient of an array at patches."""
        unit_normal = self.calc_unit_normal_at_patch(array)
        theta = jnp.arctan2(-unit_normal[:, 1], -unit_normal[:, 0])
        slope_at_patch = jnp.arccos(unit_normal[:, 2])

        components = (
            jnp.cos(theta) * slope_at_patch,
            jnp.sin(theta) * slope_at_patch
        )

        return slope_at_patch, components

    def calc_grad_at_link(self, array):
        """At each link, calculate the gradient of an array defined on nodes."""
        return jnp.divide(
            array[self.node_at_link_head] - array[self.node_at_link_tail],
            self.length_of_link
        )

    def calc_flux_div_at_node(self, array):
        """At each node, calculate the divergence of an array of fluxes defined on links."""
        face_flux = array * self.length_of_face[self.face_at_link]
        return jnp.where(
            self.cell_area_at_node != 0,
            -self.sum_at_nodes(face_flux) / self.cell_area_at_node,
            0.0
        )

    def calc_gradient_vector_at_node(self, array):
        """At each node, calculate the component-wise gradient of an array defined on nodes."""
        slope_at_patch, components = self.calc_grad_at_patch(array)
        x_slope_unmasked, y_slope_unmasked = components

        magnitude_at_node = jnp.nanmean(
            jnp.where(
                self.patches_at_node != -1,
                slope_at_patch[self.patches_at_node],
                jnp.nan
            ),
            axis = 1
        )

        x_component = jnp.nanmean(
            jnp.where(
                self.patches_at_node != -1,
                x_slope_unmasked[self.patches_at_node],
                jnp.nan
            ),
            axis = 1
        )

        y_component = jnp.nanmean(
            jnp.where(
                self.patches_at_node != -1,
                y_slope_unmasked[self.patches_at_node],
                jnp.nan
            ),
            axis = 1
        )

        return magnitude_at_node, jnp.asarray([x_component, y_component]).T
        