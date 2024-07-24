"""Models advection using a total-variation-diminishing (TVD) scheme.

The TVDAdvector models advective transport of a scalar field using a 
total-variation-diminishing (TVD) scheme. Our approach here follows
Hou et al. (2013), where a standard TVD scheme is extended to handle
an irregular mesh by use of a shift vector. We use a Van Leer flux
limiter here, but the TVDAdvector can be extended to use other limiters
as needed.

Hou, J., Simons, F., & Hinkelmann, R. (2013). 
A new TVD method for advection simulation on 2D unstructured grids. 
International Journal for Numerical Methods in Fluids, 71(10), 1260-1281.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array
from scipy.spatial import KDTree
from glacierbento import Field, Component


class TVDAdvector(Component):
    """Models advection using a total-variation-diminishing (TVD) scheme.
    
    Arguments:
        grid: A StaticGrid object.
        fields_to_advect: A list with names of fields to advect.

    Input fields:
        velocity_x: x-component of velocity field
        velocity_y: y-component of velocity field

    Output fields:
        For each field in fields_to_advect, a field with the same name

    Plus, for every name in fields_to_advect, a field with that name must
    be defined on grid nodes.
    
    Methods:
        initialize: perform setup operations outside of JIT wrapper
        run_one_step: return the advected fields after one time step
    """

    _link_velocity: Array
    _upwind_ghost_nodes: Array
    _real_indices: Array
    _real_xy: Array
    initialized: bool

    def __init__(
        self,
        grid,
        fields_to_advect = ['field'],
        params = {}
    ):
        """Initialize the Component and perform setup operations."""
        self.input_fields = {
            'velocity_x': 'node',
            'velocity_y': 'node'
        }
        
        self.output_fields = {}

        for field in fields_to_advect:
            self.input_fields[field] = 'node'
            self.output_fields[field] = 'node'

        super().__init__(grid, params)

        self._link_velocity = jnp.zeros(self._grid.number_of_links)
        self._upwind_ghost_nodes = jnp.zeros((self._grid.number_of_links, 2))
        self._real_indices = jnp.zeros(self._grid.number_of_links, dtype = int)
        self._real_xy = jnp.zeros((self._grid.number_of_links, 2))

        self.initialized = False

    def _map_velocity_to_links(self, fields: dict[str, Field]) -> Array:
        """Map components of velocity to grid links."""
        return self._grid.map_vectors_to_links(
            fields['velocity_x'].value, fields['velocity_y'].value
        )

    def _set_upwind_ghost_nodes(self, velocity: Array) -> Array:
        """Extend links upwind to set the location of ghost nodes."""
        head_x = self._grid.node_x[self._grid.node_at_link_head]
        head_y = self._grid.node_y[self._grid.node_at_link_head]
        tail_x = self._grid.node_x[self._grid.node_at_link_tail]
        tail_y = self._grid.node_y[self._grid.node_at_link_tail]

        ghost_if_head_upwind = jnp.asarray([
            tail_x - (head_x - tail_x),
            tail_y - (head_y - tail_y)
        ])

        ghost_if_tail_upwind = jnp.asarray([
            head_x - (tail_x - head_x),
            head_y - (tail_y - head_y)
        ])

        # By convention, zero velocity is considered positive
        return jnp.where(
            velocity >= 0,
            ghost_if_tail_upwind,
            ghost_if_head_upwind
        ).T

    def _get_nearest_upwind_real(self, upwind_ghost_nodes: Array) -> tuple[Array, Array]:
        """At each link, identify the nearest real node to the link's upwind ghost node."""
        points = jnp.asarray([self._grid.node_x, self._grid.node_y]).T

        tree = KDTree(points)
        _, indices = tree.query(upwind_ghost_nodes)

        return indices, points[indices]

    def _interpolate_upwind_values(self, field: Array) -> Array:
        """Interpolate field values to upwind ghost nodes."""
        magnitude, components = self._grid.calc_gradient_vector_at_node(field)
        shift_vector = self._real_xy - self._upwind_ghost_nodes

        return (
            field[self._real_indices]
            + jnp.sum(shift_vector * components[self._real_indices], axis = 1)
        )

    def _van_leer(self, r: Array) -> Array:
        """Van Leer flux limiter."""
        return (r + jnp.abs(r)) / (1 + jnp.abs(r))

    def _calc_flux(self, field: Array) -> Array:
        """Calculate the advective flux of a field across faces."""
        center = jnp.where(
            self._link_velocity >= 0,
            field[self._grid.node_at_link_tail],
            field[self._grid.node_at_link_head]
        )

        downwind = jnp.where(
            self._link_velocity >= 0,
            field[self._grid.node_at_link_head],
            field[self._grid.node_at_link_tail]
        )

        upwind = self._interpolate_upwind_values(field)

        r_factor = jnp.where(
            downwind != center,
            (center - upwind) / (downwind - center),
            0.0
        )

        face_flux = center + 0.5 * self._van_leer(r_factor) * (downwind - center)

        return self._link_velocity * face_flux

    def initialize(self, fields: dict[str, Field]):
        """Runs all setup operations that rely only on velocity. Not JIT-compatible."""
        link_velocity = self._map_velocity_to_links(fields)
        upwind_ghost_nodes = self._set_upwind_ghost_nodes(link_velocity)
        real_indices, real_xy = self._get_nearest_upwind_real(upwind_ghost_nodes)
        
        fields_to_init = lambda t: (
            t._link_velocity, t._upwind_ghost_nodes, t._real_indices, t._real_xy, t.initialized
        )

        return eqx.tree_at(
            fields_to_init,
            self,
            (link_velocity, upwind_ghost_nodes, real_indices, real_xy, True)
        )

    def run_one_step(self, dt: float, fields: [str, Field]) -> dict[str, Field]:
        """Advect every field in targets by one step of size dt."""
        output = {key: [] for key, _ in self.output_fields.items()}

        for field_name in self.output_fields.keys():
            field = fields[field_name].value
            flux_div = -self._grid.calc_flux_div_at_node(self._calc_flux(field))
            advected = field + dt * flux_div

            output = eqx.tree_at(
                lambda t: t[field_name],
                output,
                Field(advected, fields[field_name].units, 'node'),
            )

        return output

    def calc_stable_time_step(self, cfl: Float) -> Float:
        """Calculate the maximum stable time step for the current grid."""
        return cfl * jnp.min(self._grid.length_of_link / jnp.abs(self._link_velocity))