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
from scipy.spatial import KDTree
from glacierbento import Component, StaticGrid

class TVDAdvector(Component):
    """Models advection using a total-variation-diminishing (TVD) scheme."""

    grid: StaticGrid