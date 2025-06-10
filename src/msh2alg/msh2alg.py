import numpy
import argparse
import os
from src.msh2alg.generate_fiber_3D_biv import *
import subprocess

def run_command_live(cmd_list, cwd=None):
    process = subprocess.Popen(
        cmd_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        cwd=cwd
    )
    for line in process.stdout:
        print(line, end='')

    process.wait()
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd_list)
    
def convert_msh_to_xml(pathMesh, meshname):
    # Command to run dolfin-convert and convert the .msh mesh to .xml
    command = f"dolfin-convert {pathMesh} {meshname}.xml"
    os.system(command)
    
    print(f"Mesh successfully converted to {meshname}.xml.")


def run_msh2alg(
    pathMesh,
    meshname,
    dx=0.5, dy=0.5, dz=0.5,
    discretization=1000,
    alpha_endo_lv=30, alpha_epi_lv=-30, beta_endo_lv=0, beta_epi_lv=0,
    alpha_endo_sept=60, alpha_epi_sept=-60, beta_endo_sept=0, beta_epi_sept=0,
    alpha_endo_rv=80, alpha_epi_rv=-80, beta_endo_rv=0, beta_epi_rv=0
):

    convert_msh_to_xml(pathMesh, meshname)
    request_functions(meshname, alpha_endo_lv, alpha_epi_lv, beta_endo_lv, 
                beta_epi_lv, alpha_endo_sept, alpha_epi_sept, beta_endo_sept,
                beta_epi_sept, alpha_endo_rv, alpha_epi_rv, 
                beta_endo_rv, beta_epi_rv)    
    print("================================================================================", flush = True)
    print("Converting to alg...")
    run_command_live([
        './bin/HexaMeshFromVTK', '-t',
        '-i', f"../{meshname}.vtu",
        '--dx', str(dx), '--dy', str(dy), '--dz', str(dz),
        '-r', str(discretization),
        '-c', '../src/msh2alg/conf.ini',
        '-o', f"../{meshname}.alg"
    ], cwd='./hexa-mesh-from-VTK_vtk9/')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, default='', help='Path to .msh input file')
    parser.add_argument('-o', type=str, default='patient', help='Output base name (.vtu, .alg)')
    parser.add_argument('-r', type=int, default=1000, help='Discretization resolution')

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
        discretization=args.r,
        dx=args.dx, dy=args.dy, dz=args.dz,
        alpha_endo_lv=args.alpha_endo_lv, alpha_epi_lv=args.alpha_epi_lv,
        beta_endo_lv=args.beta_endo_lv, beta_epi_lv=args.beta_epi_lv,
        alpha_endo_sept=args.alpha_endo_sept, alpha_epi_sept=args.alpha_epi_sept,
        beta_endo_sept=args.beta_endo_sept, beta_epi_sept=args.beta_epi_sept,
        alpha_endo_rv=args.alpha_endo_rv, alpha_epi_rv=args.alpha_epi_rv,
        beta_endo_rv=args.beta_endo_rv, beta_epi_rv=args.beta_epi_rv
    )