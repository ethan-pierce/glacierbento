"""Models the frozen fringe beneath a glacier.

The FrozenFringe models the force balance beneath a glacier that gives rise to
a layer of debris-rich ice near the bed. Frozen fringe growth is governed by the
balance between effective pressure, which drives infiltration of ice into the
underlying till, and basal melt, which removes entrained material from the sole
of the glacier. This version of the model is adopted from Meyer et al. (2019).

Meyer, C. R., Robel, A. A., & Rempel, A. W. (2019). 
Frozen fringe explains sediment freeze-on during Heinrich events. 
Earth and Planetary Science Letters, 524, 115725.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
import lineax as lx
from jaxtyping import Array
from glacierbento import Field, Component


class FrozenFringe(Component):
    """Models the frozen fringe.

    Arguments:
        grid: A StaticGrid object.
    
    Input fields:
        ice_thickness: thickness of glacier ice at nodes
        basal_melt_rate: rate of subglacial melt at nodes
        fringe_thickness: thickness of the frozen fringe at nodes

    Output fields:
        fringe_thickness: thickness of the frozen fringe at nodes
        fringe_saturation: ratio of the volume occupied by ice
        fringe_undercooling: non-dimensional undercooling at the top of the fringe

    Methods:
        run_one_step: advance the model by one time step of size dt
    """

    def __init__(
        self,
        grid,
        params = {
            'surface_energy': 0.034,
            'pore_throat_radius': 1e-6,
            'melt_temperature': 273,
            'ice_density': 917,
            'water_density': 1000,
            'water_viscosity': 1.8e-3,
            'latent_heat': 3.34e5,
            'ice_conductivity': 2.0,
            'till_porosity': 0.35,
            'till_permeability': 4.1e-17,
            'till_grain_radius': 1.5e-4, # 0.15 mm, "medium sand"
            'film_thickness': 1e-8,
            'alpha': 3.1,
            'beta': 0.53,
            'min_fringe': 1e-3
        }
    ):
        """Initialize the FrozenFringe."""
        self.input_fields = {
            'ice_thickness': 'node',
            'basal_melt_rate': 'node',
            'till_thickness': 'node',
            'fringe_thickness': 'node'
        }

        self.output_fields = {
            'fringe_thickness': 'node',
            'fringe_saturation': 'node',
            'fringe_undercooling': 'node',
            'till_thickness': 'node'
        }

        self.params = params

        self.params['threshold_pressure'] = (
            2 * self.params['surface_energy'] / self.params['pore_throat_radius']
        )

        self.params['base_temperature'] = self._calc_base_temperature()

        super().__init__(grid, self.params)

    def _calc_base_temperature(self) -> Array:
        """Calculate the temperature at the base of the frozen fringe."""
        pf = self.params['threshold_pressure']
        Tm = self.params['melt_temperature']
        rho_i = self.params['ice_density']
        L = self.params['latent_heat']

        return Tm - ((pf * Tm) / (rho_i * L))

    def _calc_temperature_gradient(self, fields: dict[str, Field]) -> Array:
        """Calculate the temperature gradient at the base of the frozen fringe."""
        m = fields['basal_melt_rate'].value
        rho_i = self.params['ice_density']
        L = self.params['latent_heat']
        ki = self.params['ice_conductivity']

        return -m * rho_i * L / ki

    def _calc_undercooling(self, fields: dict[str, Field]) -> Array:
        """Calculate the undercooling at the top of the frozen fringe."""
        G = self._calc_temperature_gradient(fields)
        h = fields['fringe_thickness'].value
        Tm = self.params['melt_temperature']
        Tf = self.params['base_temperature']

        theta = 1 - ((G * h) / (Tm - Tf))
        return jnp.where(theta < 1, 1, theta)

    def _calc_fringe_saturation(self, fields: dict[str, Field]) -> Array:
        """Calculate the saturation of the frozen fringe."""
        theta = self._calc_undercooling(fields)

        return 1 - theta**(-self.params['beta'])

    def _calc_nominal_heave_rate(self, fields: dict[str, Field]) -> Array:
        """Calculate the nominal rate of heave at the base of the frozen fringe."""
        G = self._calc_temperature_gradient(fields)
        rho_w = self.params['water_density']
        L = self.params['latent_heat']
        k0 = self.params['till_permeability']
        rho_i = self.params['ice_density']
        Tm = self.params['melt_temperature']
        eta = self.params['water_viscosity']

        return -(rho_w**2 * L * G * k0) / (rho_i * Tm * eta)
    
    def _calc_resistance(self, fields: dict[str, Field]) -> Array:
        """Calculate the flow resistance within the frozen fringe."""
        G = self._calc_temperature_gradient(fields)
        rho_w = self.params['water_density']
        k0 = self.params['till_permeability']
        R = self.params['till_grain_radius']
        rho_i = self.params['ice_density']
        Tm = self.params['melt_temperature']
        Tf = self.params['base_temperature']
        d = self.params['film_thickness']

        return -(rho_w**2 * k0 * G * R**2) / (rho_i**2 * (Tm - Tf) * d**3)

    def _calc_heave_rate(self, fields: dict[str, Field]) -> Array:
        """Calculate the rate of heave at the base of the frozen fringe."""
        N = fields['effective_pressure'].value
        Vn = self._calc_nominal_heave_rate(fields)
        Pi = self._calc_resistance(fields)
        theta = self._calc_undercooling(fields)
        phi = self.params['till_porosity']
        pf = self.params['threshold_pressure']
        a = self.params['alpha']
        b = self.params['beta']

        numerator = (
            theta + phi * (1 - theta + (1 / (1 - b) * (theta**(1 - b) - 1))) - N / pf
        )

        d_first = ((1 - phi)**2 / (a + 1)) * (theta**(a + 1) - 1)
        d_second = ((2 * (1 - phi) * phi) / (a - b + 1)) * (theta**(a - b + 1) - 1)
        d_third = (phi**2 / (a - 2 * b + 1)) * (theta**(a - 2 * b + 1) - 1)
        denominator = (d_first + d_second + d_third + Pi)

        return jnp.where(denominator != 0.0, Vn * numerator / denominator, 0.0)

    def _calc_dhdt(self, fringe_thickness: Array, fields: dict[str, Field]) -> Array:
        """Calculate the rate of change of fringe thickness."""
        fields = eqx.tree_at(lambda t: t['fringe_thickness'].value, fields, fringe_thickness)
        m = fields['basal_melt_rate'].value
        heave = self._calc_heave_rate(fields)
        S = self._calc_fringe_saturation(fields)
        phi = self.params['till_porosity']

        return jnp.where(S != 0, (-m - heave) / (phi * S), 0.0)
        
    def run_one_step(self, dt: float, fields: dict[str, Field]) -> dict[str, Field]:
        """Advance the model one step."""
        fringe_thickness = fields['fringe_thickness'].value

        # residual = lambda h, _: h - fringe_thickness - self._calc_dhdt(h, fields) * dt
        # solver = optx.Newton(rtol = 1e-3, atol = 1e-3)
        # solution = optx.root_find(residual, solver, fringe_thickness, args = None, max_steps = 10)

        solution = fringe_thickness + self._calc_dhdt(fringe_thickness, fields) * dt

        fields = eqx.tree_at(lambda t: t['fringe_thickness'].value, fields, solution)
        fringe_saturation = self._calc_fringe_saturation(fields)
        fringe_undercooling = self._calc_undercooling(fields)

        return {
            'fringe_thickness': Field(solution, 'm', 'node'),
            'fringe_saturation': Field(fringe_saturation, 'None', 'node'),
            'fringe_undercooling': Field(fringe_undercooling, 'None', 'node')
        }
