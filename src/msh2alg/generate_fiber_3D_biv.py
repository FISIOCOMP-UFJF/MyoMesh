import os

import dolfin as df
import ldrb
import meshio
import numpy as np
from .msh2carp import gmsh2carp


def solve_laplace(mesh, boundary_markers, boundary_values, ldrb_markers):
    """
    Solves the Laplace problem (∇²u = 0) with Dirichlet boundary conditions on
    the RV, LV, and epicardial surfaces. Used to generate the scalar fields that
    drive the cardiac fiber orientation (LDRB algorithm).
    """
    V = df.FunctionSpace(mesh, 'P', 1)

    u_rv, u_lv, u_epi = boundary_values

    bc1 = df.DirichletBC(V, u_rv, boundary_markers, ldrb_markers["rv"]) 
    bc2 = df.DirichletBC(V, u_lv, boundary_markers, ldrb_markers["lv"])
    bc3 = df.DirichletBC(V, u_epi, boundary_markers, ldrb_markers["epi"])

    bcs=[bc1, bc2 ,bc3]

    ds = df.Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    dx = df.Measure('dx', domain=mesh)

    # Define variational problem
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    f = df.Constant(0.0)   
    a = df.dot(df.grad(u), df.grad(v))*dx  
    L = f*v*dx

    # Compute solution
    u = df.Function(V)
    df.solve(a == L, u, bcs, solver_parameters=dict(linear_solver='gmres', preconditioner='hypre_amg')) 

    return u

def solve_laplace_greyzone(mesh, tecido_fn):
    """
    Solves Laplace in the grey-zone region to interpolate conductivity between
    core (u=0) and healthy tissue (u=1), producing a smooth weight field for the ALG.
    Boundary conditions are placed on faces shared between core and healthy/greyzone cells.
    """
    V  = df.FunctionSpace(mesh, "P",  1)
    V0 = df.FunctionSpace(mesh, "DG", 0)

    facet_mark  = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

    tecido_vals = tecido_fn.vector().get_local()

    dofmap0     = V0.dofmap()
    cell_to_dof = [dofmap0.cell_dofs(c.index())[0] for c in df.cells(mesh)]

    for facet in df.facets(mesh):

        cells = [c for c in df.cells(facet)]

        if len(cells) == 2:
            c1, c2 = cells

            t1 = tecido_vals[cell_to_dof[c1.index()]]
            t2 = tecido_vals[cell_to_dof[c2.index()]]

            if (t1 == 2 and t2 == 0) or (t1 == 0 and t2 == 2):
                facet_mark[facet] = 1

            if (t1 == 2 and t2 == 1) or (t1 == 1 and t2 == 2):
                facet_mark[facet] = 2

    bc1 = df.DirichletBC(V, df.Constant(1.0), facet_mark, 1)
    bc2 = df.DirichletBC(V, df.Constant(0.0), facet_mark, 2)

    u = df.TrialFunction(V)
    v = df.TestFunction(V)

    a = df.dot(df.grad(u), df.grad(v)) * df.dx
    L = df.Constant(0.0) * v * df.dx

    u_sol = df.Function(V)

    df.solve(a == L, u_sol, [bc1, bc2])

    return u_sol
    
def request_functions(pathMesh, meshname, carpOutput, aux_alpha_endo_lv, aux_alpha_epi_lv, aux_beta_endo_lv,
                    aux_beta_epi_lv, aux_alpha_endo_sept, aux_alpha_epi_sept,
                    aux_beta_endo_sept, aux_beta_epi_sept, aux_alpha_endo_rv,
                    aux_alpha_epi_rv, aux_beta_endo_rv, aux_beta_epi_rv):
    """
    Main FEniCS pipeline function: loads the XML mesh, solves the Laplace fields,
    computes fiber/sheet orientations with LDRB, reads fibrosis tags from the .msh,
    solves Laplace in the grey zone, and saves everything to XDMF/VTU.
    Also exports to CARP format when carpOutput is provided.
    """
    # mesh reading converted by dolfin-convert
    mesh = df.Mesh(meshname + '.xml')
    materials = df.MeshFunction("size_t", mesh, meshname + '_physical_region.xml')
    ffun = df.MeshFunction("size_t", mesh, meshname + '_facet_region.xml')

    ldrb_markers = {
        "base": 10,
        "lv": 30,
        "epi": 40,
        "rv": 20
    }

    # Choose space for the fiber fields
    fiber_space = "DG_0"

    #--------
    # Create field to define the action potential fenotype
    # Solve Laplace problems with different boundary conditions
    # u=1 on epicardium
    phi_epi = solve_laplace(mesh, ffun, [0, 0, 1], ldrb_markers)
    # u=1 on LV endocardium
    phi_lv = solve_laplace(mesh, ffun, [0, 1, 0], ldrb_markers)
    # u=1 on RV endocardium
    phi_rv = solve_laplace(mesh, ffun, [1, 0, 0], ldrb_markers)

    # Compute field with Laplace solutions
    V = df.FunctionSpace(mesh, 'Lagrange', 1)
    u = df.Function(V)
    u.interpolate(df.Expression('-(epi + 2*rv*lv/(rv+lv) ) + 1', epi=phi_epi, rv=phi_rv, lv=phi_lv, degree=1))

    bc3 = df.DirichletBC(V, 0, ffun, ldrb_markers["epi"])
    bc3.apply(u.vector())

    u.rename("fenotipo","fenotipo")
    #--------

    # Compute the microstructure
    fiber, sheet, sheet_normal = ldrb.dolfin_ldrb(
        mesh=mesh,
        fiber_space=fiber_space,
        ffun=ffun,
        markers=ldrb_markers,
        alpha_endo_lv=aux_alpha_endo_lv,  # Fiber angle on the LV endocardium
        alpha_epi_lv=aux_alpha_epi_lv,  # Fiber angle on the LV epicardium
        beta_endo_lv=aux_beta_endo_lv,  # Sheet angle on the LV endocardium
        beta_epi_lv=aux_beta_epi_lv,  # Sheet angle on the LV epicardium
        alpha_endo_sept=aux_alpha_endo_sept,  # Fiber angle on the Septum endocardium
        alpha_epi_sept=aux_alpha_epi_sept,  # Fiber angle on the Septum epicardium
        beta_endo_sept=aux_beta_endo_sept,  # Sheet angle on the Septum endocardium
        beta_epi_sept=aux_beta_epi_sept,  # Sheet angle on the Septum epicardium
        alpha_endo_rv=aux_alpha_endo_rv,  # Fiber angle on the RV endocardium
        alpha_epi_rv=aux_alpha_epi_rv,  # Fiber angle on the RV epicardium
        beta_endo_rv=aux_beta_endo_rv,  # Sheet angle on the RV endocardium
        beta_epi_rv=aux_beta_epi_rv,
    )

    # OpenCarp
    if carpOutput:  # flagCarp
        print(50*"=", flush = True)
        print("Converting to CARP...")
        print(50*"=", flush = True)
        gmsh_path = pathMesh
        gmsh2carp(
            gmsh_path,
            carpOutput,
            mesh_fenics=mesh,
            fiber_fn=fiber,
            sheet_fn=sheet,
            normal_fn=sheet_normal,
            round_dec=8,
        )
        
    fiber.rename("f_0","f_0")
    sheet.rename("s_0","s_0")
    sheet_normal.rename("n_0","n_0")
    
    print(50*'=')
    V_fiber = fiber.function_space()
    mesh_f  = V_fiber.mesh()
    dofmap  = V_fiber.dofmap()

    print("=== Fiber FunctionSpace ===")
    print("  Element     :", V_fiber.ufl_element())
    print("  Degree      :", V_fiber.ufl_element().degree())
    print("  Mesh dim    :", mesh_f.topology().dim())
    print("  # vertices  :", mesh_f.num_vertices())
    print("  # cells     :", mesh_f.num_cells())
    print("  # DOFs      :", V_fiber.dim())
    print("  DOFs/cell   :", dofmap.cell_dofs(0).shape[0])
    print("")

    print(50*'=')
    cell_dim  = mesh.topology().dim()
    cell_mark = df.MeshFunction("size_t", mesh, cell_dim, 0)

    for facet in df.facets(mesh):
        tag = ffun[facet]
        if tag != 0:
            for cell in df.cells(facet):
                cell_mark[cell] = tag
    cell_mark.rename("region_id", "region_id")

    # Convert to DG0 Function for region_id
    V0 = df.FunctionSpace(mesh, "DG", 0)
    region_id = df.Function(V0)
    region_id.vector()[:] = cell_mark.array()
    region_id.rename("region_id", "region_id")

    # Build a per-cell tissue field (0=healthy, 1=core, 2=greyzone) from the
    # gmsh:physical tags written by markFibroseFromMsh.py
    print("Reading fibrosis tags from .msh to build tissue field (0=healthy, 1=core, 2=greyzone)...")
    gmsh_mesh = meshio.read(pathMesh)

    tetra_index = None
    for i, c in enumerate(gmsh_mesh.cells):
        if c.type == "tetra":
            tetra_index = i
            break
    if tetra_index is None:
        raise RuntimeError("No 'tetra' elements found in the gmsh mesh.")

    tetra_cells = gmsh_mesh.cells[tetra_index].data
    phys_tags   = gmsh_mesh.cell_data["gmsh:physical"][tetra_index]

    n_gmsh_cells   = tetra_cells.shape[0]
    n_fenics_cells = mesh.num_cells()

    if n_gmsh_cells != n_fenics_cells:
        print("[WARN] n_gmsh_cells != n_fenics_cells. Assuming same cell ordering.")
    else:
        print(f"  n_cells gmsh = n_cells fenics = {n_gmsh_cells}")

    tecido_fn = df.Function(V0)
    tecido_arr = tecido_fn.vector().get_local()

    dofmap0 = V0.dofmap()
    cell_to_dof = [dofmap0.cell_dofs(cell.index())[0] for cell in df.cells(mesh)]

    for cell_idx, dof in enumerate(cell_to_dof):
        tag = phys_tags[cell_idx]
        if tag == 2:
            tecido_arr[dof] = 1.0  # core
        elif tag == 3:
            tecido_arr[dof] = 2.0  # greyzone
        else:
            tecido_arr[dof] = 0.0  # saudável

    tecido_fn.vector().set_local(tecido_arr)
    tecido_fn.vector().apply("insert")
    tecido_fn.rename("tecido", "tecido")
    # =====================================================

    # Solve Laplace in the grey zone to produce a smooth conductivity weight field
    V0 = tecido_fn.function_space()
    has_greyzone = np.any(tecido_fn.vector().get_local() == 2.0)

    if has_greyzone:
        peso_tecido = solve_laplace_greyzone(mesh, tecido_fn)
        peso_tecido = df.project(peso_tecido, V0)
        peso_arr = peso_tecido.vector().get_local()
        peso_arr[peso_arr > 1.0] = 1.0
        peso_arr[peso_arr < 0.0] = 0.0
    else:
        peso_tecido = df.Function(V0)
        peso_arr = np.ones(len(tecido_fn.vector().get_local()))

    tecido_arr = tecido_fn.vector().get_local()

    for i in range(len(peso_arr)):
        if tecido_arr[i] == 0:      # saudável
            peso_arr[i] = 1.0
        elif tecido_arr[i] == 1:    # fibrose/core
            peso_arr[i] = 0.0

    peso_tecido.vector().set_local(peso_arr)
    peso_tecido.vector().apply("insert")

    peso_tecido.rename("peso_tecido", "peso_tecido")
    # =====================================================

    print("Saving…")
    with df.XDMFFile(mesh.mpi_comm(), meshname + ".xdmf") as xdmf:
        xdmf.parameters.update(
        {
            "functions_share_mesh": True,
            "rewrite_function_mesh": False
        })
        xdmf.write(mesh)
        xdmf.write(fiber, 0)
        xdmf.write(sheet, 0)
        xdmf.write(sheet_normal, 0)
        xdmf.write(tecido_fn, 0)
        xdmf.write(u, 0)
        xdmf.write(region_id, 0)   # <- gravando região sem erro
        xdmf.write(peso_tecido, 0) # NOVO: Add greyzone resolvida com Laplace

    convert_xdmf_to_vtu(meshname)

    print("Done.")

def convert_xdmf_to_vtu(meshname):
    """Converte o arquivo XDMF gerado pelo FEniCS para VTU, formato requerido pelo HexaMeshFromVTK."""
    print(50*"=", flush = True)
    print("Converting .xdmf to .vtu")
    print(50*"=", flush = True)
    filename = meshname+".xdmf"
    t, point_data, cell_data = None, None, None

    with meshio.xdmf.TimeSeriesReader(filename) as reader:
        points, cells = reader.read_points_cells()
        t, point_data, cell_data = reader.read_data(0)

    mesh = meshio.Mesh(points, cells, point_data=point_data, cell_data=cell_data,)
    mesh.write(meshname+".vtu", file_format="vtu",  )

def mirror_msh_x(in_msh, out_msh, about='centroid', x0=None):
    """
    Mirrors the mesh about a plane perpendicular to the X axis.
    about='centroid' uses x0 = mean X; about='bbox' uses x0 = bbox centre.
    Pass x0 explicitly (float) to override.

    NOTE: the actual mirroring is currently disabled (commented out below).
    The function still re-writes the mesh to Gmsh v2 ASCII, which is required
    by dolfin-convert regardless of whether the flip is applied.
    """
    m = meshio.read(in_msh)
    P = m.points.copy()
    #if x0 is None:
    #    if about == 'centroid':
    #        x0 = P[:,0].mean()
    #    elif about == 'bbox':
    #        x0 = 0.5*(P[:,0].min() + P[:,0].max())
    #    else:
    #        raise ValueError("about deve ser 'centroid' ou 'bbox' se x0=None")
    #P[:,0] = 2.0*x0 - P[:,0]         # espelho no plano x = x0
    m.points = P
    meshio.write(out_msh, m, file_format='gmsh22', binary = False)  # <-- Gmsh v2 ASCII (compatível com dolfin-convert)


# request_functions(pathMesh = "Patient_7_marked_smooth_mirX.msh", meshname = "Patient_7_marked_smooth_mirX",
#     carpOutput = None, aux_alpha_endo_lv = -60, aux_alpha_epi_lv = 60, aux_beta_endo_lv = -20,
#     aux_beta_epi_lv = 20, aux_alpha_endo_sept = -60, aux_alpha_epi_sept = 60, aux_beta_endo_sept = -20,
#     aux_beta_epi_sept = 20, aux_alpha_endo_rv = -60, aux_alpha_epi_rv = 60, aux_beta_endo_rv = -20,
#     aux_beta_epi_rv = 20)


# ==============================================================================
# BASE SURFACE EXTRACTION
# ==============================================================================

def extract_base_stl(msh_path, output_stl_path):
    """
    Reads the Gmsh .msh file and extracts the BASE surface (tag 10),
    saving it as a separate STL file.
    """
    BASE_TAG = 10  # Physical Surface("BASE", 10) defined in biv_mesh.geo
    mesh = meshio.read(msh_path)

    base_cells = []
    phys_tags = mesh.cell_data.get("gmsh:physical", [])

    for cell_block, tags in zip(mesh.cells, phys_tags):
        if cell_block.type == "triangle":
            mask = tags == BASE_TAG
            if mask.any():
                base_cells.append(cell_block.data[mask])

    if not base_cells:
        print(f"[WARN] No BASE surface (tag {BASE_TAG}) found in {msh_path}.")
        return False

    base_mesh = meshio.Mesh(
        points=mesh.points,
        cells=[("triangle", np.vstack(base_cells))],
    )
    meshio.write(output_stl_path, base_mesh)
    print(f"[OK] BASE STL saved: {output_stl_path}")
    return True


def extract_all_surfaces_stl(msh_path, output_dir, prefix):
    """
    Reads a Gmsh .msh file and extracts every physical surface (triangle tag)
    as a separate STL file.

    Files are saved as: <output_dir>/<prefix>-<NAME_OR_TAG>.stl
    Names are resolved from the mesh field_data when available.
    """
    mesh = meshio.read(msh_path)

    # Build tag -> name mapping from field_data (dimension 2 = surface)
    # Fallback to known BIV tags when field_data is absent (e.g. after fibrosis marking)
    tag_to_name = {10: "BASE", 20: "RV", 30: "LV", 40: "EPI"}
    for name, info in mesh.field_data.items():
        tag_id, dim = info[0], info[1]
        if dim == 2:
            tag_to_name[int(tag_id)] = name

    phys_tags = mesh.cell_data.get("gmsh:physical", [])

    # Collect triangles per tag
    cells_per_tag = {}
    for cell_block, tags in zip(mesh.cells, phys_tags):
        if cell_block.type == "triangle":
            for tag in np.unique(tags):
                mask = tags == tag
                cells_per_tag.setdefault(int(tag), []).append(cell_block.data[mask])

    if not cells_per_tag:
        print(f"[WARN] No surface triangles found in {msh_path}.")
        return

    os.makedirs(output_dir, exist_ok=True)
    for tag, cell_list in cells_per_tag.items():
        label = tag_to_name.get(tag, str(tag))
        out_path = os.path.join(output_dir, f"{prefix}-{label}.stl")
        surf_mesh = meshio.Mesh(
            points=mesh.points,
            cells=[("triangle", np.vstack(cell_list))],
        )
        meshio.write(out_path, surf_mesh)
        print(f"[OK] Surface '{label}' (tag {tag}) saved: {out_path}")


def embed_surfaces_in_vtu(vtu_path, output_path, region_id_key="region_id"):
    """
    Lê um VTU com tetraedros tagueados por region_id, calcula as faces de
    fronteira de cada região e as adiciona ao arquivo como células triangle,
    mantendo o region_id correspondente. O resultado é salvo em output_path.
    """
    mesh = meshio.read(vtu_path)

    tetra_block = next(c for c in mesh.cells if c.type == "tetra")
    tetras = tetra_block.data
    reg_ids = mesh.cell_data[region_id_key][0].astype(int).flatten()

    face_combos = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])

    all_tri_faces = []
    all_tri_tags  = []

    for tag in np.unique(reg_ids):
        subset = tetras[reg_ids == tag]
        faces = subset[:, face_combos].reshape(-1, 3)
        faces_sorted = np.sort(faces, axis=1)
        _, idx, counts = np.unique(faces_sorted, axis=0,
                                   return_index=True, return_counts=True)
        boundary = faces[idx[counts == 1]]
        all_tri_faces.append(boundary)
        all_tri_tags.append(np.full(len(boundary), tag, dtype=int))

    tri_faces = np.vstack(all_tri_faces)
    tri_tags  = np.concatenate(all_tri_tags)

    new_cells = mesh.cells + [meshio.CellBlock("triangle", tri_faces)]

    new_cell_data = {}
    for key, blocks in mesh.cell_data.items():
        orig = blocks[0]
        shape = (len(tri_faces),) + orig.shape[1:]
        fill = np.full(shape, -1, dtype=orig.dtype)
        if key == region_id_key:
            fill = tri_tags.reshape(shape).astype(orig.dtype)
        new_cell_data[key] = list(blocks) + [fill]

    out_mesh = meshio.Mesh(
        points=mesh.points,
        cells=new_cells,
        point_data=mesh.point_data,
        cell_data=new_cell_data,
    )
    meshio.write(output_path, out_mesh)
    print(f"[OK] VTU with embedded surfaces saved to: {output_path}")
    print(f"     {len(tri_faces)} boundary triangles added.")


# ==============================================================================