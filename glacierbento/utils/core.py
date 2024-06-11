"""Core utilities for GlacierBento.

Includes:
    Field: A class for storing data fields.
"""

import jax
import jax.numpy as jnp
import equinox as eqx


class Field(eqx.Module):
    """Stores spatially variable data.
    
    Attributes:
        name: A human-readable string identifier for the field.
        value: An array-like object containing the data.
        units: A string indicating the units of the data.
        location: A string indicating the location of the data.
    """
    name: str = eqx.field(converter = str)
    value: jnp.ndarray = eqx.field(converter = jnp.asarray)
    units: str = eqx.field(converter = str)
    location: str = eqx.field(converter = str)

    def __post_init__(self):
        if self.location not in ["node", "link", "patch", "corner", "face", "cell"]:
            raise ValueError("Invalid location for field.")
