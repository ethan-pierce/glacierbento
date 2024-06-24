"""Test the DistributedDrainageSystem using the GlaDS test case."""

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_traceback_filtering", "off")

import jax.numpy as jnp
import equinox as eqx
import lineax as lx
import optimistix as optx
import optax
import numpy as np
import matplotlib.pyplot as plt

from glacierbento.components import DistributedDrainageSystem
from glacierbento.utils import freeze_grid
from glacierbento import Field
from landlab import RasterModelGrid

rmg = RasterModelGrid((50, 150), xy_spacing = (400, 400))
grid = freeze_grid(rmg)

surface_elevation = np.sqrt(grid.node_x + 10) * (1500 / np.sqrt(np.max(grid.node_x)))
bed_elevation = grid.node_x * 1e-3
ice_thickness = surface_elevation - bed_elevation
sliding_velocity = np.full(grid.number_of_nodes, 50 / 31556926)
melt_input = np.maximum(1.62e-6 - surface_elevation * 1e-3 * 1.16e-6, 0.0)
phi0 = np.full(grid.number_of_nodes, 917 * 9.81 * ice_thickness)
h0 = np.full(grid.number_of_nodes, 1e-3)

g = 9.81
rhoi = 917
rhow = 1000
lc = 10
hc = 0.5
n = 3
A = 2.4e-24
k = 0.05
a = 3

model = DistributedDrainageSystem(grid)
fields = {
    'ice_thickness': Field('ice_thickness', ice_thickness, units = 'm', location = 'node'),
    'bed_elevation': Field('bed_elevation', bed_elevation, units = 'm', location = 'node'),
    'sliding_velocity': Field('sliding_velocity', sliding_velocity, units = 'm/s', location = 'node'),
    'melt_input': Field('melt_input', melt_input, units = 'm/s', location = 'node'),
    'potential': Field('potential', phi0, units = 'Pa', location = 'node'),
    'sheet_flow_height': Field('sheet_flow_height', h0, units = 'm', location = 'node')
}

@eqx.filter_jit
def update(dt, fields):
    output = model.run_one_step(dt, fields)

    updated_fields = eqx.tree_at(
        lambda t: (t['potential'].value, t['sheet_flow_height'].value),
        fields,
        (output['potential'].value, output['sheet_flow_height'].value)
    )

    return updated_fields

fields = update(1.0, fields)

import time
start = time.time()
fields = update(60.0, fields)
end = time.time()
print(f'Time elapsed: {end - start}')

for i in range(10):
    fields = update(60.0, fields)

plt.imshow(fields['potential'].value.reshape(grid.shape))
plt.colorbar()
plt.title('Hydraulic Potential (Pa)')
plt.show()

plt.imshow(fields['sheet_flow_height'].value.reshape(grid.shape))
plt.colorbar()
plt.title('Sheet Flow Height (m)')
plt.show()
