import pyvista as pv
from vtkmodules.util.vtkConstants import VTK_TETRA
import numpy as np
from pyeit.mesh.wrapper import PyEITMesh
import pyeit.eit.protocol as protocol
from pyeit.eit.jac import JAC
from pyeit.eit.fem import EITForward
from pyeit.mesh.external import place_electrodes_3d
from pyeit.visual.plot import create_mesh_plot_3d, create_3d_plot_with_slice

mesh_filename = "example_data/mean_meshes_tetrahedralized_coarse.vtu"


def main():
    dataset_title = "Lung Mesh with Torso"
    long_axis = "z"
    slice_ratio = 0.5
    n_electrodes = 8
    lamb = 5.5e-17

    # Read mesh
    pv_mesh = pv.read(mesh_filename)
    mesh = PyEITMesh(node=pv_mesh.points,
                     element=pv_mesh.cells_dict[VTK_TETRA],
                     perm=np.array(
                         [1. if s == 0 else 10. for s in pv_mesh["Scalar"]]) if "Scalar" in pv_mesh.cell_data else 1)

    pv_mesh["Impedance"] = mesh.perm

    electrode_nodes = place_electrodes_3d(pv_mesh, n_electrodes, long_axis, slice_ratio)
    mesh.el_pos = np.array(electrode_nodes)

    protocol_obj = protocol.create(
        n_electrodes, dist_exc=int(n_electrodes / 2), step_meas=1, parser_meas="std"
    )
    fwd = EITForward(mesh, protocol_obj)
    vh = fwd.solve_eit(perm=1)
    vi = fwd.solve_eit(perm=mesh.perm)

    # Recon
    # Set up eit object
    pyeit_obj = JAC(mesh, protocol_obj) 
    pyeit_obj.setup(p=0.5, lamb=lamb, method="kotre", perm=1)

    # # Dynamic solve simulated data
    ds_sim = pyeit_obj.solve(vi, vh, normalize=False)
    solution = np.real(ds_sim)

    pv_mesh["Reconstructed Impedance"] = solution

    # pv_mesh = pv.read("example_data/reconstructed_lung_mesh.vtk")  # If we don't want to wait, just read the pre calculated result
    # electrode_nodes = [1178, 1147, 1129, 1098, 1049, 1272, 1257, 1227]
    plot_with_pyvista(pv_mesh, electrode_nodes, dataset_title)


def plot_with_pyvista(pv_mesh, electrode_nodes, dataset_title):
    p = pv.Plotter()
    pv_mesh.set_active_scalars("Impedance")
    create_mesh_plot_3d(p, pv_mesh, electrode_nodes, title=f"Mesh Plot: {dataset_title}")
    p.camera_position = [0, -1, 0.75]
    p.camera.focal_point = np.add(p.camera.focal_point, [0, 0, 25])
    p.show(interactive_update=True)

    p = pv.Plotter()
    pv_mesh.set_active_scalars("Reconstructed Impedance")
    create_mesh_plot_3d(p, pv_mesh, electrode_nodes, title=f"EIT Plot: {dataset_title}")
    p.camera_position = [0, -1,  0.75]
    p.camera.focal_point = np.add(p.camera.focal_point, [0, 0, 25])
    p.show(interactive_update=True)

    p = pv.Plotter()
    create_3d_plot_with_slice(p, pv_mesh, title=f"EIT Plot: {dataset_title}", electrode_nodes=electrode_nodes)
    p.camera_position = [0, -1,  0.75]
    p.camera.focal_point = np.add(p.camera.focal_point, [0, 0, 25])
    p.show()


if __name__ == "__main__":
    main()
