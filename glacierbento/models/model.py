"""Define a common interface for GlacierBento Models."""

import equinox as eqx
from abc import abstractmethod
from glacierbento.utils import Field, StaticGrid

class Model(eqx.Module):
    """Models mediate between Components, Fields, and the Grid.

    The primary responsibility of a Model is to run each Component in the
    correct order and to update the Fields as necessary.

    Attributes:
        grid: The StaticGrid object that represents the computational mesh.
        fields: A dictionary of Field objects or Arrays.
        components: A dictionary of Component objects.

    Methods:
        update: Update the model by one time step.
    """