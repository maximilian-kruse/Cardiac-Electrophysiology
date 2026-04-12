import numpy as np
import pyvista as pv


# ==================================================================================================
def visualize_full_vector_field(
    mesh: pv.PolyData,
    vector_field: np.ndarray[tuple[int, int], np.dtype[np.float64]],
    scaling_factor: float,
    vector_color: str = "blue",
    mesh_color: str = "lightgray",
) -> None:
    cell_centers = mesh.cell_centers()
    cell_centers.point_data["vector_field_to_plot"] = vector_field
    glyphs = cell_centers.glyph(
        orient="vector_field_to_plot",
        scale=False,
        factor=scaling_factor,
        geom=pv.Arrow(),
    )
    plotter = pv.Plotter()
    plotter.add_mesh(glyphs, color=vector_color)
    plotter.add_mesh(mesh, color=mesh_color)
    plotter.show()


# --------------------------------------------------------------------------------------------------
def visualize_full_scalar_field(
    mesh: pv.PolyData,
    scalar_field: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> None:
    copied_mesh = mesh.copy()
    copied_mesh.cell_data["scalar_field_to_plot"] = scalar_field
    plotter = pv.Plotter()
    plotter.add_mesh(copied_mesh, scalars="scalar_field_to_plot", cmap="viridis")
    plotter.show()


# --------------------------------------------------------------------------------------------------
def visualize_partial_scalar_field(
    mesh: pv.PolyData,
    observation_inds: np.ndarray[tuple[int], np.dtype[np.float64]],
    point_size: int = 10,
    mesh_color: str = "lightgray",
) -> None:
    centers_subset = mesh.extract_cells(observation_inds).cell_centers()
    plotter = pv.Plotter()
    plotter.add_mesh(
        centers_subset, color="red", point_size=point_size, render_points_as_spheres=True
    )
    plotter.add_mesh(mesh, color=mesh_color)
    plotter.show()