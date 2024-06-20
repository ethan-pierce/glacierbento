"""This Component models the drainage system beneath a glacier.

The DistributedDrainageSystem models the evolution of hydraulic potential
(i.e., water pressure) and subglacial flow thickness beneath an individual 
glacier or a catchment of an ice sheet. Unlike more sophisticated models, this
Component ignores the channelized elements of the system, instead viewing the
entire drainage system as a network of linked cavities. Mass conservation and
the rate balance of gap opening and closure are handled similarly to GlaDS
(see Werder et al., 2013). Unlike GlaDS, however, we assume that the flow is
always laminar, following the results from Hill et al. (2023).

Werder, M. A., Hewitt, I. J., Schoof, C. G., & Flowers, G. E. (2013). 
Modeling channelized and distributed subglacial drainage in two dimensions. 
Journal of Geophysical Research: Earth Surface, 118(4), 2140-2158.

Hill, T., Flowers, G. E., Hoffman, M. J., Bingham, D., & Werder, M. A. (2023). 
Improved representation of laminar and turbulent sheet flow in subglacial drainage models.
Journal of Glaciology, 1-14.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import lineax as lx
import optimistix as optx
from glacierbento.utils import Field, MatrixAssembler
from glacierbento.components import Component


class DistributedDrainageSystem(Component):
    """Model a distributed subglacial drainage system."""

    boundary_tags: jax.Array = eqx.field(converter = jnp.asarray, init = False)

    def __init__(
        self,
        grid,
        fields,
        params = {},
        required_fields = {
            'ice_thickness': 'node',
            'bed_elevation': 'node',
            'sliding_velocity': 'node',
            'melt_input': 'node',
            'potential': 'node',
            'sheet_flow_height': 'node'
        },
        default_parameters = {
            'gravity': 9.81,
            'ice_density': 917,
            'water_density': 1000,
            'cavity_spacing': 10,
            'bed_bump_height': 0.5,
            'ice_flow_n': 3,
            'ice_flow_coeff': 2.4e-24,
            'sheet_conductivity': 0.05,
            'flow_exp_a': 3,
            'flow_exp_b': 2
        }
    ):
        """Initialize the DistributedDrainageSystem."""
        super().__init__(grid, fields, params, required_fields, default_parameters)

        self.boundary_tags = self._set_inflow_outflow(
            self._fields['bed_elevation'].value
            * self._params['water_density']
            * self._params['gravity']
        )

    def _set_inflow_outflow(self, potential: jnp.array) -> jnp.array:
        """Determine whether boundary nodes expect inflow or outflow."""
        min_adj_potential = jnp.min(
            jnp.where(
                self._grid.adjacent_nodes_at_node != -1,
                potential[self._grid.adjacent_nodes_at_node],
                jnp.inf
            ),
            axis = 1
        )

        # 1 denotes inflow, -1 denotes outflow
        inflow_outflow = jnp.where(
            potential <= min_adj_potential,
            -1 * (self._grid.status_at_node > 0),
            1 * (self._grid.status_at_node > 0)
        )

        return inflow_outflow

    def _calc_dhdt(self, potential: jnp.ndarray, sheet_flow_height: jnp.ndarray):
        """Calculate the rate of change of the thickness of sheet flow."""
        N = self.calc_effective_pressure(potential)
        ub = self._fields['sliding_velocity'].value

        opening = (
            jnp.abs(ub) 
            / self._params['cavity_spacing']
            * jnp.where(
                sheet_flow_height >= self._params['bed_bump_height'],
                jnp.zeros_like(sheet_flow_height),
                self._params['bed_bump_height'] - sheet_flow_height
            )
        )

        closure = (
            (2 / self._params['ice_flow_n']**self._params['ice_flow_n'])
            * self._params['ice_flow_coeff']
            * sheet_flow_height
            * jnp.abs(N)**(self._params['ice_flow_n'] - 1)
            * N
        )

        return opening - closure

    def _calc_coeffs(self, sheet_flow_height: jnp.ndarray):
        """Calculate the coefficients for the finite volume matrix."""
        k = self._params['sheet_conductivity']
        a = self._params['flow_exp_a']
        h_at_links = self.grid.map_mean_of_link_nodes_to_link(sheet_flow_height)
        h_at_faces = h_at_links[self.grid.link_at_face]

        return (
            self.grid.length_of_face / self.grid.length_of_link[self.grid.link_at_face] 
            * -k * h_at_faces**a
        )

    def _build_forcing_vector(self, potential: jnp.ndarray, sheet_flow_height: jnp.ndarray):
        """Build the forcing vector for the potential field."""
        dhdt = self._calc_dhdt(potential, sheet_flow_height)
        return (
            (self._fields['melt_input'].value - dhdt) * self._grid.cell_area_at_node
        )

    def _assemble_linear_system(self, potential: jnp.ndarray, sheet_flow_height: jnp.ndarray):
        """Assemble the linear system for the potential field."""
        coeffs = self._calc_coeffs(sheet_flow_height)
        forcing = self._build_forcing_vector(potential, sheet_flow_height)
        is_fixed_value = jnp.where(self.boundary_tags == 1, 1, 0)

        assembler = MatrixAssembler(
            self.grid, coeffs, forcing, jnp.zeros(self.grid.number_of_nodes), is_fixed_value
        )

        forcing, matrix = assembler.assemble_matrix()
        return forcing, matrix

    def _solve_for_potential(self, potential: jnp.ndarray, sheet_flow_height: jnp.ndarray):
        """Solve for potential from the previous step's potential and sheet thickness."""
        b, A = self._assemble_linear_system(potential, sheet_flow_height)
        operator = lx.MatrixLinearOperator(A)
        solution = lx.linear_solve(operator, b)

        base_potential = (
            self._fields['bed_elevation'].value
            * self._params['water_density']
            * self._params['gravity']
        )

        return jnp.where(
            self._grid.cell_at_node != -1,
            solution.value[self._grid.cell_at_node],
            0.0
        )

    def _update_sheet_flow_height(
        self, dt: float, potential: jnp.ndarray, sheet_flow_height: jnp.ndarray
    ):
        """Update the thickness of the sheet flow."""
        residual = lambda h, _: h - sheet_flow_height - dt * self._calc_dhdt(potential, h)
        solver = optx.Newton(rtol = 1e-6, atol = 1e-6)
        solution = optx.root_find(residual, solver, sheet_flow_height, args = None)

        return jnp.where(
            self.boundary_tags == 0,
            solution.value,
            0.0
        )

    def calc_effective_pressure(self, potential: jnp.ndarray):
        """Calculate effective pressure from the hydraulic potential."""
        H = self._fields['ice_thickness'].value
        b = self._fields['bed_elevation'].value
        overburden = self._params['ice_density'] * self._params['gravity'] * H
        water_pressure = (
            potential - self._params['water_density'] * self._params['gravity'] * b
        )
        return overburden - water_pressure

    def calc_discharge(self, potential: jnp.ndarray, sheet_flow_height: jnp.ndarray):
        """Calculate the discharge through the sheet flow on grid links."""
        k = self._params['sheet_conductivity']
        a = self._params['flow_exp_a']
        b = self._params['flow_exp_b']
        grad_phi = self._grid.calc_grad_at_link(potential)
        h_at_links = self._grid.map_mean_of_link_nodes_to_link(sheet_flow_height)

        return (
            k
            * h_at_links**a
            * jnp.abs(grad_phi)**b
            * grad_phi
        )

    def run_one_step(self, dt: float):
        """Advance the model by one time step."""
        updated_thickness = self._update_sheet_flow_height(
            dt, self._fields['potential'].value, self._fields['sheet_flow_height'].value
        )

        updated_potential = self._solve_for_potential(
            self._fields['potential'].value, updated_thickness
        )

        return eqx.tree_at(
            lambda t: (t._fields['potential'].value, t._fields['sheet_flow_height'].value),
            self,
            (updated_potential, updated_thickness)
        )
        