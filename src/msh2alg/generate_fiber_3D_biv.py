import dolfin as df
import ldrb
import meshio
from msh2carp import gmsh2carp


def solve_laplace(mesh, boundary_markers, boundary_values, ldrb_markers):
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


def request_functions(pathMesh, meshname, carpOutput, aux_alpha_endo_lv, aux_alpha_epi_lv, aux_beta_endo_lv, 
                    aux_beta_epi_lv, aux_alpha_endo_sept, aux_alpha_epi_sept, 
                    aux_beta_endo_sept, aux_beta_epi_sept, aux_alpha_endo_rv, 
                    aux_alpha_epi_rv, aux_beta_endo_rv, aux_beta_epi_rv):

    #mesh reading converted by dolfin-convert

    mesh = df.Mesh(meshname + '.xml')
    materials = df.MeshFunction("size_t", mesh, meshname + '_physical_region.xml')
    ffun = df.MeshFunction("size_t", mesh, meshname + '_facet_region.xml')

    V0 = df.FunctionSpace(mesh, "DG", 0) 
    tecido = df.Function(V0) 
    tecido.vector()[:] = materials.array()==2
    tecido.rename("tecido", "tecido")

    ldrb_markers = {
        "base": 10,
        "lv": 30,
        "epi": 40,
        "rv": 20
    }

    # Choose space for the fiber fields
    # This is a string on the form {family}_{degree}
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
    #OpenCarp
    if carpOutput:  # flagCarp
        print(50*"=", flush = True)
        print("Converting to CARP...")
        print(50*"=", flush = True)
        gmsh_path = pathMesh     # ajuste se arquivo estiver noutro diretório
        gmsh2carp(
            gmsh_path,
            carpOutput,
            mesh_fenics=mesh,
            fiber_fn=fiber,
            sheet_fn=sheet,
            normal_fn=sheet_normal,
            round_dec=8,        # ajuste tolerância se necessário
        )
        
    fiber.rename("f_0","f_0")
    sheet.rename("s_0","s_0")
    sheet_normal.rename("n_0","n_0")
    
    print(50*'=')
    # 1) Espaço de funções
    V_fiber = fiber.function_space()
    mesh_f  = V_fiber.mesh()
    dofmap  = V_fiber.dofmap()

    print("=== FunctionSpace do fiber ===")
    print("  Elemento    :", V_fiber.ufl_element())  # tipo do elemento (DG, Lagrange…)
    print("  Grau        :", V_fiber.ufl_element().degree())
    print("  Dim. malha  :", mesh_f.topology().dim())
    print("  # vértices  :", mesh_f.num_vertices())
    print("  # células   :", mesh_f.num_cells())
    print("  # DOFs      :", V_fiber.dim())
    print("  DOFs por célula :", dofmap.cell_dofs(0).shape[0])
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

    # Convert to DG0 Function
    V0 = df.FunctionSpace(mesh, "DG", 0) # Scalar field with piecewise constant values
    region_id = df.Function(V0)
    region_id.vector()[:] = cell_mark.array()
    region_id.rename("region_id", "region_id")

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
        xdmf.write(tecido, 0)
        xdmf.write(u, 0)
        xdmf.write(region_id, 0)   # ← aqui grava sem erro

    convert_xdmf_to_vtu(meshname)

    print("Done.")

def convert_xdmf_to_vtu(meshname):

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