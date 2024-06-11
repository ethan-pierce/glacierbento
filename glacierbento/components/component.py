"""Define a common interface for model components."""

import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod
from glacierbento.utils.core import Field
from glacierbento.utils.static_grid import StaticGrid


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


class Component(eqx.Module):
    """Components are individual process models with a common interface.
    
    Every Component has to fulfill the following contract: given a grid and a
    specific set of fields defined on elements of that grid, the Component will
    provide an run_one_step() method that calculates some output field. For time-
    independent processes, the run_one_step() method should not accept any 
    arguments; for time-dependent processes, the run_one_step() method should 
    accept a float indicating the size of the desired time step.

    Specific Components that inherit from this class *must* write default values 
    for required fields and default parameters in the class definition. Where
    appropriate, Components should also cite a source paper in their docstring.
    
    Components will be created and destroyed many times during the typical
    lifetime of a Model, so it is important to keep any required setup as
    minimal as possible. If there are expensive operations, consider creating
    another Component to handle those processes and e.g., saving the result as
    a Field that the Model can pass to the main Component on initialization.

    Attributes:
        grid: The StaticGrid object that the Component operates on.
        fields: A dictionary of Field objects or Arrays.
        required_fields: A dictionary with {names: locations} of required fields.
        params: A dictionary of parameters.
    
    Methods:
        run_one_step: Advance the model by one time step.
        get_output: Return the output field(s) of the model.
    """

    _grid: StaticGrid
    _fields: dict
    _params: dict
    required_fields: dict

    def __init__(self, grid, fields, params = {}, required_fields = {}, default_parameters = {}):
        """Initialize the Component."""
        self._grid = grid
        self.required_fields = required_fields

        self._fields = {}
        for var, field in fields.items():
            try:
                loc = field.location
            except:
                loc = check_location(self._grid, field)
                field = Field(var, jnp.asarray(field), units = '', location = loc)
                
            self._fields[var] = field

        for field, loc in self.required_fields.items():
            if field not in self._fields.keys():
                raise ValueError(f"Field {field} is required but not provided.")
            if self._fields[field].location != loc:
                raise ValueError(f"Field {field} must be defined on grid element: {loc}.")

        self._params = default_parameters
        for key, value in params.items():
            self._params[key] = value


    @abstractmethod
    def run_one_step(self):
        """Run one time step of the Component."""
        pass

    @abstractmethod
    def get_output(self):
        """Return the output field(s) of the Component."""
        pass