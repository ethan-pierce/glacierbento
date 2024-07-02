"""Process-agnostic model of glacial erosion.

The SimpleGlacialEroder uses a nonlinear, sliding-based formulation to estimate
the rate of subglacial erosion in a catchment. The model is based on a synthesis
from Herman et al. (2021) that compared velocities and erosion rates from a wide
variety of regions and geologic settings. There are two key parameters: a rate
coefficient and a sliding exponent. Users should be cautious when applying this
model to small areas (e.g., sub-kilometer grid cells) and over short time scales 
(e.g., seasons or months), as process-based models of abrasion and quarrying may
be more appropriate for those investigations.

Herman, F., De Doncker, F., Delaney, I., Prasicek, G., & Koppes, M. (2021). 
The impact of glaciers on mountain erosion. 
Nature Reviews Earth & Environment, 2(6), 422-435.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Float, Array
from glacierbento import Field, Component


class SimpleGlacialEroder(Component):
    """Models erosion beneath a glacier.

    Arguments:
        grid: A StaticGrid object.
    
    Input fields:
        sliding_velocity: the magnitude of sliding velocity at grid nodes.
        till_thickness: the thickness of till at grid nodes.
    
    Output fields:
        erosion_rate: the rate of subglacial erosion at grid nodes.
        till_thickness: the updated thickness of till at grid nodes.
    
    Methods:
        run_one_step: return the updated erosion rate after one time step.
    """

    def __init__(
        self, 
        grid,
        params = {
            'rate_coefficient': 2.7e-7 * 31556926, # 1 / (ms)
            'sliding_exponent': 2
        }
    ):
        """Initialize the component."""
        self.input_fields = {
            'sliding_velocity': 'node',
            'till_thickness': 'node'
        }

        self.output_fields = {
            'erosion_rate': 'node',
            'till_thickness': 'node'
        }

        super().__init__(grid, params)

    def run_one_step(self, dt: Float, fields: dict[str, Field]) -> dict[str, Field]:
        """Advance the model by one time step of size dt."""
        sliding_velocity = jnp.abs(fields['sliding_velocity'].value)

        erosion_rate = (
            self.params['rate_coefficient'] 
            * jnp.power(sliding_velocity, self.params['sliding_exponent'])
        )

        till_thickness = fields['till_thickness'].value + erosion_rate * dt

        return {
            'erosion_rate': Field(erosion_rate, fields['sliding_velocity'].units, 'node'),
            'till_thickness': Field(till_thickness, fields['till_thickness'].units, 'node')
        }