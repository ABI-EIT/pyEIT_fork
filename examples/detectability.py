# coding: utf-8
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import pyeit.eit.jac as jac
import pyeit.mesh as mesh
from pyeit.eit.fem import EITForward
import pyeit.eit.protocol as protocol
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.mesh.external import place_electrodes_equal_spacing
from pyeit.eit.render import render_2d_mesh
from pyeit.visual.plot import (
    create_image_plot,
    create_layered_image_plot,
    create_mesh_plot,
)
from pyeit.quality.merit import calc_greit_figures_of_merit, calc_fractional_amplitude_set
from pyeit.quality.eit_system import calc_detectability
from numpy.random import default_rng
from tqdm import tqdm
import scipy.stats as st

"""
Trying to experimentally validate the detectability calculation as a z statistic. This is not working. I probably did 
it wrong.
"""


def main():
    n_el = 16
    render_resolution = (64, 64)
    background = 1
    anomaly = 1.1
    conductive_target = True if anomaly - background > 0 else False
    noise_magnitude = 7e-5
    num_tries_target = 1000
    num_tries_background = 1000

    # Problem setup
    sim_mesh = mesh.create(n_el, h0=0.05)
    electrode_nodes = place_electrodes_equal_spacing(sim_mesh, n_electrodes=16)
    sim_mesh.el_pos = np.array(electrode_nodes)
    anomaly = PyEITAnomaly_Circle(center=[0.5, 0], r=0.05, perm=anomaly)
    sim_mesh = mesh.set_perm(sim_mesh, anomaly=anomaly, background=background)

    # Simulation
    protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")
    fwd = EITForward(sim_mesh, protocol_obj)
    v0 = fwd.solve_eit(perm=background)
    v1 = fwd.solve_eit(perm=sim_mesh.perm)

    recon_mesh = mesh.create(n_el, h0=0.1)
    electrode_nodes = place_electrodes_equal_spacing(recon_mesh, n_electrodes=16)
    recon_mesh.el_pos = np.array(electrode_nodes)
    eit = jac.JAC(recon_mesh, protocol_obj)
    eit.setup(p=0.5, lamb=0.03, method="kotre", perm=1, jac_normalized=True)

    rng = default_rng(0)

    recon_renders = solve_noisy_eit(v0, v1, noise_magnitude, rng, eit, recon_mesh, render_resolution, num_tries_target)

    mean_recon = np.mean(recon_renders, axis=0)
    roi = calc_fractional_amplitude_set(mean_recon, conductive_target=conductive_target)

    detectabilities = [calc_detectability(recon_render) for recon_render in recon_renders]
    recon_means = [np.mean(recon_render[roi == 1]) for recon_render in recon_renders]

    mean_detectability = np.mean(detectabilities)
    mean_recon_mean = np.mean(recon_means)

    p_value = st.norm.sf(mean_detectability)

    print(f"Mean recon mean:{mean_recon_mean}\nmean detectability z statistic:{mean_detectability}\nprobability of a mean"
          f" at least as high in this ROI by chance:{p_value}")

    # Render results
    sim_render = render_2d_mesh(
        sim_mesh, sim_mesh.perm, resolution=render_resolution
    )
    recon_render = mean_recon

    background_renders = solve_noisy_eit(v0, v0, noise_magnitude, rng, eit, recon_mesh, render_resolution, num_tries_background)

    background_means = [np.mean(render[roi == 1]) for render in background_renders]
    number_above_mean = np.count_nonzero(np.array(background_means) >= mean_recon_mean)
    print(f"Max background mean in same ROI is:{max(background_means)}\nNumber above recon mean in {num_tries_background} tries is:{number_above_mean}\n"
          f"proportion:{number_above_mean/num_tries_background}")

    # Create mesh plots
    fig, axs = plt.subplots(1, 2)
    create_mesh_plot(axs[0], sim_mesh, ax_kwargs={"title": "Sim mesh"})
    create_mesh_plot(axs[1], recon_mesh, ax_kwargs={"title": "Recon mesh"})
    fig.set_size_inches(10, 4)

    fig, axs = plt.subplots(1, 2)
    im_simulation = create_image_plot(axs[0], sim_render, title="Target image")
    im_recon = create_image_plot(axs[1], recon_render, title="Reconstruction image")
    fig.set_size_inches(10, 4)

    fig, ax = plt.subplots(constrained_layout=True)
    create_layered_image_plot(
        ax,
        (
            recon_render,
            roi,
        ),
        labels=["Background", "ROI"],
        title="ROI",
        margin=10,
    )

    plt.show()


def solve_noisy_eit(v0, v1, noise_magnitude, rng, eit, recon_mesh, render_resolution, num_tries):

    renders = []
    for _ in tqdm(range(num_tries)):
        n_0 = noise_magnitude * rng.standard_normal(len(v0))
        n_1 = noise_magnitude * rng.standard_normal(len(v1))
        v0_n = v0 + n_0
        v1_n = v1 + n_1

        ds = eit.solve(v1_n, v0_n, normalize=True)
        solution = np.real(ds)
        recon_render = render_2d_mesh(recon_mesh, solution, resolution=render_resolution)
        renders.append(recon_render)

    return np.array(renders)



if __name__ == "__main__":
    main()
