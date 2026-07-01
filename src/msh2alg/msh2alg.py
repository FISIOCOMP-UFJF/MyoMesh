import numpy
import argparse
import os
from src.msh2alg.generate_fiber_3D_biv import *
from src.msh2alg.generate_fiber_3D_biv import _read_msh_robust
import subprocess

def run_command_live(cmd_list, cwd=None):
    """Run a subprocess printing the output in real time, with special formatting for progress lines."""
    process = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        cwd=cwd
    )
    for line in process.stdout:
        if line.startswith("Progress"):
            print('\r' + line.rstrip(), end='', flush=True)
        else:
            print(line, end='', flush=True)

    print()
    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd_list)
    
def convert_msh_to_xml(pathMesh, meshname):
    """Convert the Gmsh mesh (.msh) to the FEniCS XML format using dolfin-convert."""
    # Command to run dolfin-convert and convert the .msh mesh to .xml
    command = f"dolfin-convert {pathMesh} {meshname}.xml"
    os.system(command)
    
    print(f"Mesh successfully converted to {meshname}.xml.")


def run_msh2alg(
    pathMesh,
    meshname,
    carp,
    alg,
    dx=0.5, dy=0.5, dz=0.5,
    discretization=1000,
    alpha_endo_lv=30, alpha_epi_lv=-30, beta_endo_lv=0, beta_epi_lv=0,
    alpha_endo_sept=60, alpha_epi_sept=-60, beta_endo_sept=0, beta_epi_sept=0,
    alpha_endo_rv=80, alpha_epi_rv=-80, beta_endo_rv=0, beta_epi_rv=0,
    log_file=None,
    no_hexa=False,
):
    """
    Converts the .msh to FEniCS XML, computes cardiac fibers with LDRB,
    exports to VTU/XDMF, and optionally converts to ALG and CARP.
    Fiber (alpha) and sheet (beta) angles are configurable per region (LV, septum, RV).
    Set no_hexa=True to skip HexaMeshFromVTK and produce only VTU output.
    """
    # Convert to Gmsh v2 ASCII (required by dolfin-convert)
    import meshio as _meshio
    # Intermediate file: store in patientMsh/intermediate/ instead of the root
    _mesh_dir = os.path.dirname(pathMesh)
    _int_dir = os.path.join(_mesh_dir, "intermediate")
    os.makedirs(_int_dir, exist_ok=True)
    _base = os.path.splitext(os.path.basename(pathMesh))[0]
    v2_msh = os.path.join(_int_dir, _base + "_gmsh22.msh")
    _m = _read_msh_robust(pathMesh)
    _meshio.write(v2_msh, _m, file_format='gmsh22', binary=False)
    pathMesh = v2_msh

    convert_msh_to_xml(pathMesh, meshname)
    request_functions(pathMesh, meshname, carp, alpha_endo_lv, alpha_epi_lv, beta_endo_lv,
                beta_epi_lv, alpha_endo_sept, alpha_epi_sept, beta_endo_sept,
                beta_epi_sept, alpha_endo_rv, alpha_epi_rv,
                beta_endo_rv, beta_epi_rv)

    if no_hexa:
        print("Skipping HexaMeshFromVTK (--no_hexa).")
        return

    print(50*"=", flush=True)
    print("Converting to alg...")
    cmd = [
        './bin/HexaMeshFromVTK',
        '-i', f"../{meshname}.vtu",
        '--dx', str(dx), '--dy', str(dy), '--dz', str(dz),
        '-r', str(discretization),
        '-c', '../src/msh2alg/conf.ini',
        '-o', f"../{alg}.alg"
    ]
    if log_file:
        cmd += ['-l', f"../{log_file}"]
    run_command_live(cmd, cwd='./hexa-mesh-from-VTK_vtk9/')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default='', help='Path to .msh input file')
    parser.add_argument('-o', type=str, default='patient', help='Output base name (.vtu, .alg)')
    parser.add_argument('-r', type=int, default=1000, help='Discretization resolution')
    parser.add_argument('--carp', type=str, default='', help='Output OpenCARP')
    parser.add_argument('--alg', type=str, default='', help='Output OpenAlg')
    parser.add_argument('--log_file', type=str, default=None, help='Path to log file for HexaMeshFromVTK output')
    parser.add_argument('--no_hexa', action='store_true', help='Skip HexaMeshFromVTK; produce VTU/XDMF only (no ALG).')

    parser.add_argument('--dx', type=float, default=0.5)
    parser.add_argument('--dy', type=float, default=0.5)
    parser.add_argument('--dz', type=float, default=0.5)

    parser.add_argument('--alpha_endo_lv', type=float, default=30)
    parser.add_argument('--alpha_epi_lv', type=float, default=-30)
    parser.add_argument('--beta_endo_lv', type=float, default=0)
    parser.add_argument('--beta_epi_lv', type=float, default=0)

    parser.add_argument('--alpha_endo_sept', type=float, default=60)
    parser.add_argument('--alpha_epi_sept', type=float, default=-60)
    parser.add_argument('--beta_endo_sept', type=float, default=0)
    parser.add_argument('--beta_epi_sept', type=float, default=0)

    parser.add_argument('--alpha_endo_rv', type=float, default=80)
    parser.add_argument('--alpha_epi_rv', type=float, default=-80)
    parser.add_argument('--beta_endo_rv', type=float, default=0)
    parser.add_argument('--beta_epi_rv', type=float, default=0)

    args = parser.parse_args()

    run_msh2alg(
        pathMesh=args.i,
        meshname=args.o,
        carp=args.carp,
        alg=args.alg,
        discretization=args.r,
        dx=args.dx, dy=args.dy, dz=args.dz,
        alpha_endo_lv=args.alpha_endo_lv, alpha_epi_lv=args.alpha_epi_lv,
        beta_endo_lv=args.beta_endo_lv, beta_epi_lv=args.beta_epi_lv,
        alpha_endo_sept=args.alpha_endo_sept, alpha_epi_sept=args.alpha_epi_sept,
        beta_endo_sept=args.beta_endo_sept, beta_epi_sept=args.beta_epi_sept,
        alpha_endo_rv=args.alpha_endo_rv, alpha_epi_rv=args.alpha_epi_rv,
        beta_endo_rv=args.beta_endo_rv, beta_epi_rv=args.beta_epi_rv,
        log_file=args.log_file,
        no_hexa=args.no_hexa,
    )