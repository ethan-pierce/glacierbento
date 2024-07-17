"""Models the inefficient component of the drainage system beneath a glacier.

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
from jaxtyping import Array
from glacierbento import Field, Component
from glacierbento.utils import MatrixAssembler


class DistributedDrainageSystem(Component):
    """Model a distributed subglacial drainage system."""

    def __init__(
        self,
        grid,
        params = {
            'gravity': 9.81,
            'ice_density': 917,
            'water_density': 1000,
            'cavity_spacing': 10,
            'bed_bump_height': 1.0,
            'ice_flow_n': 3,
            'ice_flow_coeff': 2.4e-24,
            'sheet_conductivity': 0.05,
            'flow_exp_a': 3,
            'flow_exp_b': 2
        }
    ):
        """Initialize the DistributedDrainageSystem."""
        self.input_fields = {
            'ice_thickness': 'node',
            'bed_elevation': 'node',
            'sliding_velocity': 'node',
            'basal_melt_rate': 'node',
            'potential': 'node',
            'sheet_flow_height': 'node'
        }

        self.output_fields = {
            'potential': 'node',
            'sheet_flow_height': 'node'
        }

        super().__init__(grid, params)

    def _set_inflow_outflow(self, fields: dict[str, Field]) -> Array:
        """Determine whether boundary nodes expect inflow or outflow."""
        bed_elevation = fields['bed_elevation'].value
        base_potential = (self.params['water_density'] * self.params['gravity'] * bed_elevation)

        min_adj_potential = jnp.min(
            jnp.where(
                self._grid.adjacent_nodes_at_node != -1,
                base_potential[self._grid.adjacent_nodes_at_node],
                jnp.inf
            ),
            axis = 1
        )

        # 1 denotes inflow, -1 denotes outflow
        inflow_outflow = jnp.where(
            (base_potential <= min_adj_potential),
            -1 * (self._grid.status_at_node > 0),
            1 * (self._grid.status_at_node > 0)
        )

        return inflow_outflow

    def _calc_dhdt(self, sheet_flow_height: Array, fields: dict[str, Field]) -> Array:
        """Calculate the rate of change of the thickness of sheet flow."""
        sliding_velocity = fields['sliding_velocity'].value
        N = self.calc_effective_pressure(fields)

        opening = (
            jnp.abs(sliding_velocity) 
            / self.params['cavity_spacing']
            * jnp.where(
                sheet_flow_height >= self.params['bed_bump_height'],
                jnp.zeros_like(sheet_flow_height),
                self.params['bed_bump_height'] - sheet_flow_height
            )
        )

        closure = (
            (2 / self.params['ice_flow_n']**self.params['ice_flow_n'])
            * self.params['ice_flow_coeff']
            * sheet_flow_height
            * jnp.abs(N)**(self.params['ice_flow_n'] - 1)
            * N
        )

        return opening - closure

    def _calc_coeffs(self, fields: dict[str, Field]) -> Array:
        """Calculate the coefficients for the finite volume matrix."""
        sheet_flow_height = fields['sheet_flow_height'].value
        k = self.params['sheet_conductivity']
        a = self.params['flow_exp_a']

        h_at_links = self._grid.map_mean_of_link_nodes_to_link(sheet_flow_height)
        h_at_faces = h_at_links[self._grid.link_at_face]

        return (
            self._grid.length_of_face / self._grid.length_of_link[self._grid.link_at_face] 
            * -k * h_at_faces**a
        )

    def _build_forcing_vector(self, fields: dict[str, Field]) -> Array:
        """Build the forcing vector for the potential field."""
        basal_melt_rate = fields['basal_melt_rate'].value
        sheet_flow_height = fields['sheet_flow_height'].value
        dhdt = self._calc_dhdt(sheet_flow_height, fields)
        forcing_at_nodes = basal_melt_rate - dhdt

        return forcing_at_nodes[self._grid.node_at_cell]

    def _assemble_linear_system(self, fields) -> tuple[Array, Array]:
        """Assemble the linear system for the potential field."""
        coeffs = self._calc_coeffs(fields)
        forcing = self._build_forcing_vector(fields)

        bed_elevation = fields['bed_elevation'].value
        boundary_values = self.params['water_density'] * self.params['gravity'] * bed_elevation

        boundary_tags = self._set_inflow_outflow(fields)
        is_fixed_value = jnp.where(boundary_tags == 1, 1, 0)

        assembler = MatrixAssembler(self._grid, coeffs, forcing, boundary_values, is_fixed_value)

        forcing, matrix = assembler.assemble_matrix()
        return forcing, matrix

    def _solve_for_potential(self, fields: dict[str, Field]) -> Array:
        """Solve for potential from the previous step's potential and sheet thickness."""
        b, A = self._assemble_linear_system(fields)

        operator = lx.MatrixLinearOperator(A)
        solution = lx.linear_solve(operator, b)

        bed_elevation = fields['bed_elevation'].value
        base_potential = (bed_elevation * self.params['water_density'] * self.params['gravity'])

        return jnp.where(
            self._grid.cell_at_node != -1,
            solution.value[self._grid.cell_at_node],
            base_potential
        )

    def _update_sheet_flow_height(self, dt: float, fields: dict[str, Field]) -> Array:
        """Update the thickness of the sheet flow."""
        sheet_flow_height = fields['sheet_flow_height'].value

        residual = lambda h, _: h - sheet_flow_height - dt * self._calc_dhdt(h, fields)
        solver = optx.Newton(rtol = 1e-6, atol = 1e-6)
        solution = optx.root_find(residual, solver, sheet_flow_height, args = None)

        updated_sheet_flow = jnp.where(
            solution.value >= 0,
            solution.value,
            0.0
        )

        return jnp.where(
            self._grid.cell_at_node != -1,
            solution.value,
            0.0
        )

    def calc_effective_pressure(self, fields: dict[str, Field]) -> Array:
        """Calculate effective pressure from the hydraulic potential."""
        potential = fields['potential'].value
        bed_elevation = fields['bed_elevation'].value
        ice_thickness = fields['ice_thickness'].value

        overburden = self.params['ice_density'] * self.params['gravity'] * ice_thickness
        water_pressure = (
            potential - self.params['water_density'] * self.params['gravity'] * bed_elevation
        )
        return overburden - water_pressure

    def calc_discharge(self, fields: dict[str, Field]) -> Array:
        """Calculate the discharge through the sheet flow on grid links."""
        potential = fields['potential'].value
        sheet_flow_height = fields['sheet_flow_height'].value
        k = self.params['sheet_conductivity']
        a = self.params['flow_exp_a']
        b = self.params['flow_exp_b']
        grad_phi = self._grid.calc_grad_at_link(potential)
        h_at_links = self._grid.map_mean_of_link_nodes_to_link(sheet_flow_height)

        return (
            -k
            * h_at_links**a
            * jnp.abs(grad_phi)**(b - 2)
            * grad_phi
        )

    def run_one_step(self, dt: float, fields: dict[str, Field]) -> dict[str, Field]:
        """Advance the model by one time step."""
        updated_potential = self._solve_for_potential(fields)

        updated_sheet_flow_height = self._update_sheet_flow_height(dt, fields)
        
        return {
            'potential': Field(updated_potential, 'Pa', 'node'),
            'sheet_flow_height': Field(updated_sheet_flow_height, 'm', 'node')
        }
        