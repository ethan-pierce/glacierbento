"""Define a common interface for model components.

In order to function properly as a Component, an object must be tied to a Model
instance. This is necessary because the Model object mediates between all
Components, the shared Grid, Fields, and Parameters. Components should not access 
these objects directly, but instead should pass information through the Model. By
default, Components will not
"""

from typing import Protocol
from abc import abstractmethod

class Component(Protocol):
    """Components are individual process models with a common interface."""

    def __init__(
        self, 
        model, 
        input_fields: Dict, 
        output_fields: Dict, 
        default_parameters: Dict
    ):
        """Initialize the Component."""
        self._model = model
        self._input_fields = input_fields
        self._output_fields = output_fields
        self._default_parameters = default_parameters

    def add_parameters(self):
        """Add parameters to the Model object if they do not already exist."""
        for key, val in self._default_parameters.items():
            if key not in self._model.parameters:
                self._model.add_parameter(key, val)

    def verify_required_fields(self):
        """Verify that all required Fields are present."""
        for key, val in self._input_fields.items():
            if key not in self._model.fields:
                raise ValueError(f"Field {key} is required but not present.")

        for key, val in self._output_fields.items():
            if key not in self._model.fields:
                self._model.add_field(key, val)

    @abstractmethod
    def add_output_fields(self):
        """Add output Fields to the Model object."""
        pass

    @abstractmethod
    def run_one_step(self):
        """Run one time step of the Component."""
        pass

