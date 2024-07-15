"""Models the dispersed layer of sediment beneath a glacier.

The DispersedLayer component models the migration of sediment particles within
a dispersed basal ice layer, typically overlying till or frozen fringe. This model
is based on thermal regelation driven by the supercooling of ice at the top of
the frozen fringe, and as such, should usually be coupled with the FrozenFringe 
component. The physics are described in Pierce et al. (in press).
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from jaxtyping import Array
from glacierbento import Field, Component


class DispersedLayer(Component):
    """Models particle migration in the dispersed layer.

    Arguments:
        grid: A StaticGrid object.

    Input fields:
        temperature_gradient: temperature gradient through the dispersed layer
        dispersed_thickness: thickness of the dispersed layer at nodes

    Output fields:
        dispersed_thickness: thickness of the dispersed layer at nodes
        regelation_rate: maximum velocity of grains in the dispersed layer

    Methods:
        run_one_step: advance the model by one time step of size dt
    """

    def __init__(
        self,
        grid,
        params = {
            'ice_density': 917,
            'ice_conductivity': 2.0,
            'latent_heat': 3.34e5
        }
    ):
        """Initialize the DispersedLayer."""
        self.input_fields = {
            'temperature_gradient': 'node',
            'dispersed_thickness': 'node'
        }

        self.output_fields = {
            'dispersed_thickness': 'node',
            'regelation_rate': 'node'
        }

        super().__init__(grid, params)

    def _calc_regelation_rate(self, fields: dict[str, Field]) -> Array:
        """Calculate the maximum rate of regelation for particles or clusters."""
        return (
            3 * self.params['ice_conductivity'] * fields['temperature_gradient'].value
            / (self.params['ice_density'] * self.params['latent_heat'])
        )

    def run_one_step(self, dt: float, fields: dict[str, Field]) -> dict[str, Field]:
        """Advance the model by one step."""
        regelaton_rate = self._calc_regelation_rate(fields)

        residual = lambda h, _: h - fields['dispersed_thickness'].value - regelaton_rate * dt
        solver = optx.Newton(rtol = 1e-6, atol = 1e-6)
        solution = optx.root_find(residual, solver, fields['dispersed_thickness'].value, args = None)

        updated_dispersed = jnp.maximum(solution, 0.0)

        return {
            'dispersed_thickness': Field(updated_dispersed, 'm', 'node'),
            'regelation_rate': Field(regelaton_rate, 'm/s', 'node')
        }