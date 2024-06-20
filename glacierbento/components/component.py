"""Define a common interface for Components.

Components are self-contained process models that return output fields when the
run_one_step() method is called. It may help to think of each Component as being
responsible for one equation in a paper or textbook. Specific Components should
extend the base class here with the necessary logic for the process they model.
Consider adding citations to the docstrings of Components to properly attribute
the source of the equation(s) being implemented.
"""

import equinox as eqx
from abc import abstractmethod
from glacierbento.utils import Field, StaticGrid

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

    def update_param(self, param: str, new_value: float) -> None:
        """Update the value of a parameter."""
        self.params[param] = new_value

    @abstractmethod
    def run_one_step(self, dt: float, input_fields: dict) -> dict[str, Field]:
        """Advance the model by one time step."""
        pass
