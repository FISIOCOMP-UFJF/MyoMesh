import os
import time
import argparse
import meshio
import numpy as np
import vtk
import matplotlib.pyplot as plt

def mark_fibrosis(path_msh, stl_dir, output_path="saida_com_fibrose.msh", plot_centroids=False):
    # Reads the original .msh
    mesh      = meshio.read(path_msh)
    points    = mesh.points  # (N_pts, 3)
    cells     = mesh.cells
    cell_data = mesh.cell_data

    # Locates the tetra elements and their original tags
    tetra_index = None
    for i, c in enumerate(cells):
        if c.type == "tetra":
            tetra_index = i
            break
    tetra = cells[tetra_index].data  # array shape = (n_tetra, 4)
    tags = cell_data["gmsh:physical"][tetra_index]  # shape = (n_tetra,)

    n_tetra = tetra.shape[0]

    # Builds a list of unique nodes used in the tetra elements
    flat_nodes = tetra.flatten()  # shape = (n_tetra*4,)
    unique_nodes, inverse_idx = np.unique(flat_nodes, return_inverse=True)
    
    # unique_nodes: unique indices of 'points' that actually appear in a tetra
    # inverse_idx: array (n_tetra*4,) of integers saying, for each position in flat_nodes,
    # which position corresponds inside unique_nodes.

    # Now coords of the unique nodes:
    coords_unique = points[unique_nodes]  # (N_unique, 3)

    # Creates vtkPolyData only with these unique nodes
    vtk_pts_nodes = vtk.vtkPoints()
    for p in coords_unique:
        vtk_pts_nodes.InsertNextPoint(p)
    input_poly_nodes = vtk.vtkPolyData()
    input_poly_nodes.SetPoints(vtk_pts_nodes)

    centroids = np.mean(points[tetra], axis=1)  # (n_tetra, 3)
    # For optional plotting if plot_centroids=True.

    # Prepares the tags array that will be updated
    tags_new = tags.copy()  # we will mark as “2” (fibrosis) the identified tetra

    # For each STL of fibrosis
    # Run vtkSelectEnclosedPoints on the set of N_unique nodes
    # Run vtkSelectEnclosedPoints on the set of N_tetra centroids
    # Combine: tetra is marked if (center is inside) or (some vertex is inside)

    # To avoid recreating the centroids polydata every time, we create it here:
    vtk_pts_cent = vtk.vtkPoints()
    for c in centroids:
        vtk_pts_cent.InsertNextPoint(c)
    input_poly_cent = vtk.vtkPolyData()
    input_poly_cent.SetPoints(vtk_pts_cent)

    # Reads and sorts the STL files
    for fname in sorted(os.listdir(stl_dir)):
        if not fname.lower().endswith(".stl"):
            continue

        full_path = os.path.join(stl_dir, fname)

        # "Adaptive wait" to ensure the STL is ready
        for _ in range(20):  # up to ~2 seconds
            try:
                with open(full_path, "r", errors="ignore") as f:
                    lines = [next(f) for _ in range(5)]
                if any("vertex" in line for line in lines):
                    break
            except Exception:
                pass
            time.sleep(0.1)
        else:
            print(f"[ERROR] STL {fname} not ready or malformed. Skipping.")
            continue

        print(f"Processing {fname} ...")
        reader = vtk.vtkSTLReader()
        reader.SetFileName(full_path)
        reader.Update()  # processes reading
        fib_surface = reader.GetOutput()

        # Creates a selector for the N_unique nodes
        selector_nodes = vtk.vtkSelectEnclosedPoints()
        selector_nodes.SetInputData(input_poly_nodes)
        selector_nodes.SetSurfaceData(fib_surface)
        selector_nodes.SetTolerance(1e-6)  # can adjust (1e-5, 1e-4) for "slack"
        selector_nodes.Update()

        # Retrieves booleans of which nodes are inside
        Nuniq = coords_unique.shape[0]  # number of tetra nodes
        inside_unique_nodes = np.zeros(Nuniq, dtype=bool)
        for i in range(Nuniq):
            inside_unique_nodes[i] = bool(selector_nodes.IsInside(i))

        # Selector for centroids
        # selector_cent tests points
        # tool to verify if the point is inside or outside the surface
        selector_cent = vtk.vtkSelectEnclosedPoints()
        selector_cent.SetInputData(input_poly_cent)
        selector_cent.SetSurfaceData(fib_surface)
        selector_cent.SetTolerance(1e-6)
        selector_cent.Update()
        
        inside_cent = np.zeros(n_tetra, dtype=bool)
        for i in range(n_tetra):
            inside_cent[i] = bool(selector_cent.IsInside(i))

        # Rebuilds an array (n_tetra, 4) indicating if each vertex of the tetra is inside
        inside_nodes_per_tet = inside_unique_nodes[inverse_idx].reshape((n_tetra, 4))

        # Rule of decision for each tetra:
        # If the centroid is inside or any vertex is inside
        tet_to_mark = (inside_cent) | (inside_nodes_per_tet.any(axis=1))

        # Updates the tags
        tags_new[tet_to_mark] = 2

    # Rewrites the mesh with updated tags
    new_cell_data = {}
    for key in cell_data:
        new_cell_data[key] = cell_data[key].copy()
        new_cell_data[key][tetra_index] = tags_new

    meshio.write(
        output_path,
        meshio.Mesh(
            points=points,
            cells=cells,
            cell_data=new_cell_data
        ),
        file_format="gmsh22",
        binary=False
    )

    # Optional plot of the centroids
    if plot_centroids:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=5, c='k')
        ax.set_title("Centroids of Tetrahedrons")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marks fibrosis in the mesh using STL files.")
    parser.add_argument("--msh", required=True, help="Path to the original .msh file")
    parser.add_argument("--stl_dir", required=True, help="Directory containing the STL files")
    parser.add_argument("--output_path", required=True, help="Where to save the new .msh with fibrosis")
    parser.add_argument("--plot", action="store_true", help="If set, plots the centroids")
    args = parser.parse_args()

    mark_fibrosis(args.msh, args.stl_dir, args.output_path, plot_centroids=args.plot)
