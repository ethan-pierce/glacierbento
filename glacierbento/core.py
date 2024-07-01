"""Core utilities for GlacierBento.

Includes:
    check_location: Infer an array's location based on its size.
    Field: A class for storing data fields.
    Component: A base class for process models.
    Model: A class for managing multiple linked Components.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod
from jaxtyping import Array, Float
from glacierbento.utils import StaticGrid


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


class Field(eqx.Module):
    """Stores spatially variable data.
    
    Attributes:
        name: A human-readable string identifier for the field.
        value: An array-like object containing the data.
        units: A string indicating the units of the data.
        location: A string indicating the location of the data.
    """
    value: jnp.ndarray = eqx.field(converter = jnp.asarray)
    units: str = eqx.field(converter = str)
    location: str = eqx.field(converter = str)

    def __post_init__(self):
        if self.location not in ["node", "link", "patch", "corner", "face", "cell"]:
            raise ValueError("Invalid location for field.")


class Component(eqx.Module):
    """Components are individual process models with a common interface.
    
    Every Component has to provide a run_one_step() method that accepts all
    required input fields and returns all output fields. By convention, the
    first argument to run_one_step() should be the time step, dt. Additionally, 
    each Component should provide reasonable default values for parameters in
    its __init__() method.

    Attributes:
        grid: The StaticGrid object that the Component operates on.
        input_fields: A dictionary with {name: location} of all input fields.
        output_fields: A dictionary with {name: location} of all output fields.
        params: A dictionary of parameters.
    
    Methods:
        update_param: Update the value of a parameter.
        run_one_step: Advance the model by one time step.
    """

    _grid: StaticGrid
    input_fields: dict[str, str]
    output_fields: dict[str, str]
    params: dict[str, float]

    @abstractmethod
    def __init__(self, grid, params = {}):
        """Components should be able to be instantiated with only a grid."""
        self._grid = grid
        self.params = params

    def update_param(self, param: str, new_value: float) -> None:
        """Update the value of a parameter."""
        self.params[param] = new_value

    @abstractmethod
    def run_one_step(self, dt: float, input_fields: dict) -> dict[str, Field]:
        """Advance the model by one time step."""
        pass


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