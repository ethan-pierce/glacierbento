"""Core utilities for GlacierBento.

Includes:
    Field: A class for storing data fields.
"""

import jax
import jax.numpy as jnp
import equinox as eqx


class Field(eqx.Module):
    """Stores spatially variable data."""
    name: str = eqx.field(converter = str)
    value: jnp.ndarray = eqx.field(converter = jnp.asarray)
    units: str = eqx.field(converter = str)
    location: str = eqx.field(converter = str)


