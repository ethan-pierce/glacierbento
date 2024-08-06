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
        theta: non-dimensional supercooling at the top of the frozen fringe
        base_temperature: temperature at the base of the frozen fringe
        dispersed_thickness: thickness of the dispersed layer at nodes

    Output fields:
        dispersed_thickness: thickness of the dispersed layer at nodes
        temperature_gradient: temperature gradient through the dispersed layer
        regelation_rate: maximum velocity of grains in the dispersed layer

    Methods:
        run_one_step: advance the model by one time step of size dt
    """

    def __init__(
        self,
        grid,
        params = {
            'melt_temperature': 273.15,
            'ice_density': 917,
            'ice_conductivity': 2.0,
            'latent_heat': 3.34e5,
            'critical_depth': 10.0
        }
    ):
        """Initialize the DispersedLayer."""
        self.input_fields = {
            'theta': 'node',
            'base_temperature': 'node',
            'dispersed_thickness': 'node'
        }

        self.output_fields = {
            'dispersed_thickness': 'node',
            'temperature_gradient': 'node',
            'regelation_rate': 'node'
        }

        super().__init__(grid, params)

    def _calc_top_temperature(self, fields: dict[str, Field]) -> Array:
        """Calculate the temperature at the top of the frozen fringe."""
        theta = fields['theta'].value
        Tf = fields['base_temperature'].value
        Tm = self.params['melt_temperature']

        return Tm - theta * (Tm - Tf)

    def _calc_temperature_gradient(self, fields: dict[str, Field]) -> Array:
        """Calculate the temperature gradient through the dispersed layer."""
        top_temperature = self._calc_top_temperature(fields)
        Tm = self.params['melt_temperature']
        crit_depth = jnp.maximum(fields['dispersed_thickness'].value, self.params['critical_depth'])

        return (top_temperature - Tm) / crit_depth

    def _calc_regelation_rate(self, dispersed_thickness: Array, fields: dict[str, Field]) -> Array:
        """Calculate the maximum rate of regelation for particles or clusters."""
        fields = eqx.tree_at(lambda t: t['dispersed_thickness'].value, fields, dispersed_thickness)
        G = self._calc_temperature_gradient(fields)
        ki = self.params['ice_conductivity']
        rho_i = self.params['ice_density']
        L = self.params['latent_heat']

        return -3 * ki * G / (rho_i * L)

    def run_one_step(self, dt: float, fields: dict[str, Field]) -> dict[str, Field]:
        """Advance the model by one step."""
        dispersed_thickness = fields['dispersed_thickness'].value

        # residual = lambda h, _: h - dispersed_thickness - dt * self._calc_regelation_rate(h, fields)
        # solver = optx.Newton(rtol = 1e-3, atol = 1e-3)
        # solution = optx.root_find(residual, solver, dispersed_thickness, args = None)
        # updated_dispersed = jnp.maximum(solution.value, 1e-3)

        updated_dispersed = dispersed_thickness + dt * self._calc_regelation_rate(dispersed_thickness, fields)

        fields = eqx.tree_at(lambda t: t['dispersed_thickness'].value, fields, updated_dispersed)
        gradient = self._calc_temperature_gradient(fields)
        regelaton_rate = self._calc_regelation_rate(updated_dispersed, fields)

        return {
            'dispersed_thickness': Field(updated_dispersed, 'm', 'node'),
            'temperature_gradient': Field(gradient, 'K/m', 'node'),
            'regelation_rate': Field(regelaton_rate, 'm/s', 'node')
        }