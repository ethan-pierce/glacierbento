"""Test the DistributedDrainageSystem using the GlaDS test case."""

import jax
import jax.numpy as jnp
import lineax as lx
import optimistix as optx
import optax
import numpy as np
import matplotlib.pyplot as plt

from glacierbento.components import DistributedDrainageSystem
from glacierbento.utils import freeze_grid, Field
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

def calc_dhdt(phi, h):
    opening = jnp.abs(sliding_velocity) / lc * jnp.where(h > hc, 0.0, hc - h)
    N = rhoi * g * ice_thickness + rhow * g * bed_elevation - phi
    closure = 2 * A / n**n * h * jnp.abs(N)**(n - 1) * N
    return opening - closure

def build_forcing_vector(phi, h):
    dhdt = calc_dhdt(phi, h)
    return (melt_input - dhdt) * grid.cell_area_at_node

def calc_coeffs(h):
    h_at_faces = grid.map_mean_of_link_nodes_to_link(h)[grid.link_at_face]
    return (
        -grid.length_of_face / grid.length_of_link[grid.link_at_face] * k * h_at_faces**a
    )
coeffs = calc_coeffs(h0)
is_dirichlet = jnp.where(grid.node_x == 0, 1, 0)
is_neumann = jnp.where((grid.status_at_node != 0) & (grid.node_x != 0), 1, 0)

def assemble_row(cell):
    row = jnp.zeros(grid.number_of_cells)
    node = grid.node_at_cell[cell]
    for link in grid.links_at_node[node]:
        jax.lax.cond(
            link == -1,
            lambda link: None,
            add_valid,
            link
        )

        def add_valid(link):
            adj = jnp.where(
                grid.node_at_link_head[link] == node, 
                grid.node_at_link_tail[link], 
                grid.node_at_link_head[link]
            )

            condition = jnp.argwhere(
                jnp.array([
                    link == -1, 
                    is_dirichlet[adj], 
                    is_neumann[adj], 
                    grid.status_at_node[adj] == 0
                ], size = 1)
            ).squeeze()

            jax.lax.switch(condition, [lambda: None, add_dirichlet, add_neumann, add_interior])

            def add_dirichlet(link):
                row.at[adj].set()

def assemble_matrix():
    return jax.vmap(assemble_row)(jnp.arange(grid.number_of_cells))

def residual(phi, h):
    h_at_links = grid.map_mean_of_link_nodes_to_link(h)
    grad_phi = grid.calc_grad_at_link(phi)
    q = -k * h_at_links**a * grad_phi
    flux_div = grid.calc_flux_div_at_node(q)
    forcing = build_forcing_vector(phi, h)
    return flux_div - forcing

solution = optx.least_squares(
    residual,
    solver = optx.OptaxMinimiser(optax.adabelief(learning_rate = 1e-3), rtol = 1e-8, atol = 1e-8,),
    y0 = jnp.where(grid.node_x == 0, 0.0, phi0),
    args = h0,
    max_steps = 10000
)
new_phi = solution.value
res = residual(new_phi, h0)

plt.imshow(res.reshape(grid.shape))
plt.colorbar()
plt.show()
