"""Models ice velocity using a shallow ice approximation.

The shallow ice approximation models ice as a viscous fluid following a Glen-Nye
power-law rheology. It is most appropriate for large ice masses with minimal
topographic influence, low sliding velocities, small surface slopes (e.g., the
interior regions of large ice sheets). It can be combined with observations of
surface velocity to estimate sliding speeds, assuming that surface velocity 
is a combination of slip and deformation (where the SIA models deformation).
The default rate coefficient is appropriate for ice near the pressure-melting 
point, and should be adjusted accordingly for colder ice (see Cuffey and Paterson,
2010). Additionally, users should be careful to average surface slopes over very
large areas (at least 5-10 times ice thickness). This can be done quickly in
Python using scipy.ndimage.gaussian_filter or similar operations.

Cuffey, K. M., & Paterson, W. S. B. (2010). The physics of glaciers. Academic Press.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array
from glacierbento import Field, Component


class ShallowIceApproximation(Component):
    """Models ice velocity using a shallow ice approximation.

    Arguments:
        grid: A StaticGrid object.

    Input fields:
        ice_thickness: thickness of glacier ice at nodes
        surface_elevation: elevation of the ice surface at nodes

    Output fields:
        deformation_velocity: velocity due to deformation of the ice

    Methods:
        calc_velocity: calculates the velocity of the ice
        run_one_step: updates the model state and returns the output fields

    The argument dt is not used for this component.
    """

    def __init__(
        self,
        grid,
        params = {
            'ice_flow_coefficient': 2.4e-24, # Pa^-n s^-1
            'glens_n': 3,
            'ice_density': 917,
            'gravity': 9.81
        }
    ):
        """Initialize the component."""
        self.input_fields = {
            'ice_thickness': 'node',
            'surface_elevation': 'node'
        }

        self.output_fields = {
            'deformation_velocity': 'link'
        }

        super().__init__(grid, params)

    def _calc_velocity(self, fields: dict[str, Field]) -> Array:
        """Calculate the velocity of the ice."""
        flow_coeff = (
            2 * self.params['ice_flow_coefficient']
            / (self.params['glens_n'] + 2)
            * (self.params['ice_density'] * self.params['gravity'])**self.params['glens_n']
        )

        surface_elevation = fields['surface_elevation'].value
        surface_slope = self._grid.calc_grad_at_link(surface_elevation)

        ice_thickness = fields['ice_thickness'].value
        thickness_at_links = self._grid.map_mean_of_link_nodes_to_link(ice_thickness)

        return (
            flow_coeff
            * surface_slope**self.params['glens_n']
            * thickness_at_links**(self.params['glens_n'] + 1)
        )

    def run_one_step(self, dt: Float, fields: dict[str, Field]) -> dict[str, Field]:
        """Update the ice velocity field."""
        deformation_velocity = self._calc_velocity(fields)

        return {
            'deformation_velocity': Field(deformation_velocity, 'm/s', 'link')
        }