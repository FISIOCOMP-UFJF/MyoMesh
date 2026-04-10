import os
import time
import argparse
import meshio
import numpy as np
import vtk
import matplotlib.pyplot as plt


def add_greyzone_shell(tetra, tags, core_tag=2, grey_tag=3):
    """
    Garante que todo tetra core (tag=core_tag) tenha vizinhos imediatos
    (por face) marcados como greyzone (tag=grey_tag), sem sobrescrever core.

    tetra: array (n_tetra, 4) com índices de nós
    tags:  array (n_tetra,) com as tags gmsh:physical já marcadas
    """
    n_tets = tetra.shape[0]

    # faces locais do tetra: combinações de 3 vértices
    faces_local = np.array([[0, 1, 2],
                            [0, 1, 3],
                            [0, 2, 3],
                            [1, 2, 3]], dtype=int)

    # mapeia face (3 nós globais ordenados) -> lista de células que usam essa face
    face_to_cells = {}
    for ci in range(n_tets):
        nodes = tetra[ci]
        for fl in faces_local:
            face_nodes = tuple(sorted(nodes[fl]))
            if face_nodes in face_to_cells:
                face_to_cells[face_nodes].append(ci)
            else:
                face_to_cells[face_nodes] = [ci]

    grey_to_add = np.zeros(n_tets, dtype=bool)

    # para cada face compartilhada, se alguma das células é core,
    # marcamos as outras células da face como greyzone candidata
    for cells in face_to_cells.values():
        if len(cells) < 2:
            continue  # face de fronteira, só 1 célula

        has_core = any(tags[ci] == core_tag for ci in cells)
        if not has_core:
            continue

        for ci in cells:
            if tags[ci] != core_tag:
                grey_to_add[ci] = True

    new_tags = tags.copy()
    mask = grey_to_add & (new_tags != core_tag)
    new_tags[mask] = grey_tag

    return new_tags


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
    if tetra_index is None:
        raise RuntimeError("Nenhum elemento 'tetra' encontrado no .msh.")

    tetra = cells[tetra_index].data  # array shape = (n_tetra, 4)
    tags = cell_data["gmsh:physical"][tetra_index]  # shape = (n_tetra,)

    n_tetra = tetra.shape[0]

    # Builds a list of unique nodes used in the tetra elements
    flat_nodes = tetra.flatten()  # shape = (n_tetra*4,)
    unique_nodes, inverse_idx = np.unique(flat_nodes, return_inverse=True)
    coords_unique = points[unique_nodes]  # (N_unique, 3)

    # Creates vtkPolyData only with these unique nodes
    vtk_pts_nodes = vtk.vtkPoints()
    for p in coords_unique:
        vtk_pts_nodes.InsertNextPoint(p)
    input_poly_nodes = vtk.vtkPolyData()
    input_poly_nodes.SetPoints(vtk_pts_nodes)

    # Centroids of tetrahedra
    centroids = np.mean(points[tetra], axis=1)  # (n_tetra, 3)

    vtk_pts_cent = vtk.vtkPoints()
    for c in centroids:
        vtk_pts_cent.InsertNextPoint(c)
    input_poly_cent = vtk.vtkPolyData()
    input_poly_cent.SetPoints(vtk_pts_cent)

    # Prepares the tags array that will be updated
    # 2 -> core
    # 3 -> greyzone / border zone
    tags_new = tags.copy()

    # Reads and sorts the STL files
    for fname in sorted(os.listdir(stl_dir)):
        if not fname.lower().endswith(".stl"):
            continue

        full_path = os.path.join(stl_dir, fname)

        # Decide which tag to apply from filename
        lower_name = fname.lower()
        if "core" in lower_name:
            tag_value = 2
        elif "greyzone" in lower_name or "grayzone" in lower_name or "border" in lower_name or "gz" in lower_name:
            tag_value = 3
        else:
            tag_value = 2  # default core if not clear

        print(f"Processing {fname}  (tag {tag_value})...")

        # Small wait to ensure STL is ready
        for _ in range(20):
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

        reader = vtk.vtkSTLReader()
        reader.SetFileName(full_path)
        reader.Update()
        fib_surface = reader.GetOutput()

        # Selector for vertices
        selector_nodes = vtk.vtkSelectEnclosedPoints()
        selector_nodes.SetInputData(input_poly_nodes)
        selector_nodes.SetSurfaceData(fib_surface)
        selector_nodes.SetTolerance(1e-6)
        selector_nodes.Update()

        Nuniq = coords_unique.shape[0]
        inside_unique_nodes = np.zeros(Nuniq, dtype=bool)
        for i in range(Nuniq):
            inside_unique_nodes[i] = bool(selector_nodes.IsInside(i))

        # Selector for centroids
        selector_cent = vtk.vtkSelectEnclosedPoints()
        selector_cent.SetInputData(input_poly_cent)
        selector_cent.SetSurfaceData(fib_surface)
        selector_cent.SetTolerance(1e-6)
        selector_cent.Update()
        
        inside_cent = np.zeros(n_tetra, dtype=bool)
        for i in range(n_tetra):
            inside_cent[i] = bool(selector_cent.IsInside(i))

        # Node inclusion per tetra
        inside_nodes_per_tet = inside_unique_nodes[inverse_idx].reshape((n_tetra, 4))

        tet_to_mark = inside_cent | inside_nodes_per_tet.any(axis=1)

        if tag_value == 2:
            tags_new[tet_to_mark] = 2
        elif tag_value == 3:
            mask_border = tet_to_mark & (tags_new != 2)
            tags_new[mask_border] = 3

    # Pós passo: garante “casca” de greyzone em volta do core
    tags_new = add_greyzone_shell(tetra, tags_new, core_tag=2, grey_tag=3)

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

    if plot_centroids:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s=5, c='k')
        ax.set_title("Centroids of Tetrahedrons")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Marks fibrosis (core + border zone) in the mesh using STL files.")
    parser.add_argument("--msh", required=True, help="Path to the original .msh file")
    parser.add_argument("--stl_dir", required=True, help="Directory containing the STL files")
    parser.add_argument("--output_path", required=True, help="Where to save the new .msh with fibrosis tags")
    parser.add_argument("--plot", action="store_true", help="If set, plots the centroids")
    args = parser.parse_args()

    mark_fibrosis(args.msh, args.stl_dir, args.output_path, plot_centroids=args.plot)
