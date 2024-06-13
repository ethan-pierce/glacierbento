"""Define a common interface for GlacierBento Models.

The Model class is the top-level object that coordinates between Components,
Fields, and the Grid. It is responsible for running each Component in the
correct order and updating the Fields as necessary. Specific Models should
extend the base class here with the necessary logic for the process they model.
"""

import equinox as eqx
from abc import abstractmethod
from jaxtyping import Float, Array
from glacierbento.utils import Field, StaticGrid


def check_location(grid, array) -> str:
    """Check which set of grid elements matches the size of an array-like."""
    if len(array) == grid.number_of_nodes:
        return "node"
    elif len(array) == grid.number_of_links:
        return "link"
    elif len(array) == grid.number_of_patches:
        return "patch"
    elif len(array) == grid.number_of_corners:
        return "corner"
    elif len(array) == grid.number_of_faces:
        return "face"
    elif len(array) == grid.number_of_cells:
        return "cell"
    else:
        raise ValueError("Array size does not match any grid elements.")


class Model(eqx.Module):
    """Models mediate between Components, Fields, and the Grid.

    The primary responsibility of a Model is to run each Component in the
    correct order and to update the Fields as necessary. For best performance,
    the entire Model.update() method should be wrapped in a @jax.jit decorator.

    Attributes:
        grid: The StaticGrid object that represents the computational mesh.
        fields: A dictionary of Field objects or Arrays.
        components: A dictionary of Component objects.

    Methods:
        add_field(): Add a Field object to the Model.
        update_field(): Update the value of a Field.
        get_value(): Get the value of a Field.
        add_component(): Add a Component to the Model.
        check_fields(): Check that all required fields are present.
        initialize(): Perform any setup operations.
        update(): Update the model by one time step.
        finalize(): Perform any cleanup operations.
    """

    grid: StaticGrid
    fields: dict[str, Field]
    components: dict[str, eqx.Module]

    def __init__(self, fields, components):
        """Initialize the Model with fields and components."""
        self.fields = fields
        self.components = components

    def add_field(self, field: Field) -> None:
        """Add a Field object to the Model."""
        self.fields[field.name] = field

    def update_field(self, field: str, new_value: Array) -> None:
        """Update the value of a Field."""
        self.fields[field].value = new_value

    def get_field(self, field: str) -> Array:
        """Get the value of a Field."""
        return self.fields[field].value

    def add_component(self, name: str, component: eqx.Module) -> None:
        """Add a Component to the Model."""
        self.components[name] = component

    def check_fields(self):
        """Check that all required fields are present in the correct locations."""
        for component in self.components.values():
            for field, location in component.required_fields.items():
                if field not in self.fields.keys():
                    raise ValueError(f"Field {field} is required but not provided.")
                if check_location(self.grid, self.fields[field].value) != location:
                    raise ValueError(f"Field {field} must be defined on grid element: {location}.")

    @abstractmethod
    def initialize(self):
        """Perform any setup operations."""
        pass

    @abstractmethod
    @jax.jit
    def update(self, dt: Float):
        """Update the model by one time step."""
        pass

    @abstractmethod
    def finalize(self):
        """Perform any cleanup operations."""
        pass