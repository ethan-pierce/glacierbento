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
from glacierbento.utils import Field
from glacierbento.components import Component


class DistributedDrainageSystem(Component):
    """Model a distributed subglacial drainage system."""

    def __init__(
        self,
        grid,
        fields,
        params = {},
        required_fields = {
            'ice_thickness': 'node',
            'bed_elevation': 'node',
            'sliding_velocity': 'node',
            'melt_input': 'node'
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

        initial_potential = (
            self._params['ice_density'] 
            * self._params['gravity'] 
            * self._fields['ice_thickness'].value
            * 0.8
        )

        self._fields['hydraulic_potential'] = Field(
            'hydraulic_potential', initial_potential, 'Pa', 'node'
        )

        self._fields['sheet_thickness'] = Field(
            'sheet_thickness', jnp.zeros(grid.number_of_nodes), 'm', 'node'
        )

    def _calc_dhdt(self, hydraulic_potential: jnp.ndarray, sheet_thickness: jnp.ndarray):
        """Calculate the rate of change of the thickness of sheet flow."""
        N = self.calc_effective_pressure(hydraulic_potential)
        ub = self._fields['sliding_velocity'].value

        opening = (
            jnp.abs(ub) 
            / self._params['cavity_spacing']
            * jnp.where(
                sheet_thickness >= self._params['bed_bump_height'],
                jnp.zeros_like(sheet_thickness),
                self._params['bed_bump_height'] - sheet_thickness
            )
        )

        closure = (
            (2 / self._params['ice_flow_n']**self._params['ice_flow_n'])
            * self._params['ice_flow_coeff']
            * sheet_thickness
            * jnp.abs(N)**(self._params['ice_flow_n'] - 1)
            * N
        )

        return opening - closure

    def _solve_for_potential(self, hydraulic_potential: jnp.ndarray, sheet_thickness: jnp.ndarray):
        """Solve for the hydraulic potential given the sheet thickness."""
        pass

    def calc_effective_pressure(self, hydraulic_potential: jnp.ndarray):
        """Calculate effective pressure from the hydraulic potential."""
        H = self._fields['ice_thickness'].value
        b = self._fields['bed_elevation'].value
        overburden = self._params['ice_density'] * self._params['gravity'] * H
        water_pressure = (
            hydraulic_potential - self._params['water_density'] * self._params['gravity'] * b
        )
        return overburden - water_pressure

    def calc_discharge(self, hydraulic_potential: jnp.ndarray, sheet_thickness: jnp.ndarray):
        """Calculate the discharge through the sheet flow on grid links."""
        k = self._params['sheet_conductivity']
        a = self._params['flow_exp_a']
        b = self._params['flow_exp_b']
        grad_phi = self._grid.calc_grad_at_link(hydraulic_potential)
        h_at_links = self._grid.map_mean_of_link_nodes_to_link(sheet_thickness)

        return (
            -k
            * h_at_links**a
            * jnp.abs(grad_phi)**b
            * grad_phi
        )

    def run_one_step(self, dt: float):
        """Advance the model by one time step."""
        updated_potential = self._fields['hydraulic_potential']
        updated_thickness = self._fields['sheet_thickness']

        return eqx.tree_at(
            lambda t: (t._fields['hydraulic_potential'], t._fields['sheet_thickness']),
            self,
            (updated_potential, updated_thickness)
        )
        
    def get_output(self) -> dict:
        """Return the output field(s) of the model."""
        return {
            'hydraulic_potential': self._fields['hydraulic_potential'], 
            'sheet_thickness': self._fields['sheet_thickness']
        }