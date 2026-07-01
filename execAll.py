import os
import glob
import logging
from datetime import datetime
import subprocess
from src.mat2msh.readMat import readMat
from src.mat2msh.readScar import msh_tag_to_ply
from src.mat2msh.dicom_align import compute_dicom_transform, apply_transform_to_file
import argparse
import shutil
import numpy as np
from scipy.io import loadmat
from src.msh2alg.generate_fiber_3D_biv import *


def setup_logging(log_path):
    """Configure logging to file and console simultaneously, overwriting previous runs."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def execute_commands(input_file):
    """
    Run the full pipeline for a patient .mat file:
    slice reading and alignment -> coordinate export -> PLY/STL surface generation
    -> volumetric mesh with Gmsh -> fibrosis marking -> conversion to ALG/CARP.
    """
    filename = os.path.basename(input_file)
    patient_id = os.path.splitext(filename)[0]
   
    # Create the output folder with the current date
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"./output/{date_str}/{patient_id}"
    os.makedirs(output_dir, exist_ok=True)
    aligned_mat_path = f"{output_dir}/{patient_id}.mat"

    setup_logging(os.path.join(output_dir, "pipeline.log"))
    logging.info("========================================================================================")
    logging.info("Starting the pipeline for patient data processing.")
    logging.info("========================================================================================")

    # Paths for surfaces and intermediate files
    stl_srf = f"{output_dir}/patientSTL"
    msh_srf = f"{output_dir}/patientMsh"
    txt_srf = f"{output_dir}/patientTxt"
    ply_srf = f"{output_dir}/patientPly"
    scar_srf = f"{output_dir}/scarSTL_raw"
    scar_ply_smooth = f"{output_dir}/scarPly"
    scar_stl_smooth = f"{output_dir}/scarSTL"
    mesh_output_base = f"{output_dir}/fenicsFiles/{patient_id}"
    carp_output = f"{output_dir}/carpFiles/{patient_id}"
    alg_output = f"{output_dir}/algFiles/{patient_id}"
    log_dir = os.path.join(output_dir, "logs")
    
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if os.path.exists(scar_ply_smooth):
        shutil.rmtree(scar_ply_smooth)
    os.makedirs(scar_ply_smooth, exist_ok=True)

    if os.path.exists(scar_stl_smooth):
        shutil.rmtree(scar_stl_smooth)
    os.makedirs(scar_stl_smooth, exist_ok=True)

    if os.path.exists(stl_srf):
        shutil.rmtree(stl_srf)
    os.makedirs(stl_srf, exist_ok=True)

    if os.path.exists(msh_srf):
        shutil.rmtree(msh_srf)
    os.makedirs(msh_srf, exist_ok=True)
    msh_int = f"{msh_srf}/intermediate"
    os.makedirs(msh_int, exist_ok=True)

    if os.path.exists(scar_srf):
        shutil.rmtree(scar_srf)
    os.makedirs(scar_srf, exist_ok=True)

    if os.path.exists(txt_srf):
        shutil.rmtree(txt_srf)
    os.makedirs(txt_srf, exist_ok=True)

    if os.path.exists(ply_srf):
        shutil.rmtree(ply_srf)
    os.makedirs(ply_srf, exist_ok=True)

    if os.path.exists(mesh_output_base):
        shutil.rmtree(mesh_output_base)
    os.makedirs(os.path.dirname(mesh_output_base), exist_ok=True)
    
    if os.path.exists(carp_output):
        shutil.rmtree(carp_output)  
    os.makedirs(os.path.dirname(carp_output), exist_ok=True)
    
    if os.path.exists(alg_output):
        shutil.rmtree(alg_output)
    os.makedirs(os.path.dirname(alg_output), exist_ok=True)

    # --no_invert_z flag to propagate to subcommands (inverts Z by default)
    invert_z_flag = "--no_invert_z" if args.no_invert_z else ""

    # DICOM alignment and barycenter regression both run by default: the per-slice
    # respiratory-motion correction is kept (reviewers value it) and the resulting
    # small (~2 mm) non-rigid residual in the DICOM fit is accepted as a trade-off.
    # Disable the regression explicitly with --no_regression if needed.

    # Step 1: Process the .mat file
    logging.info(f"Processing the file: {input_file}")
    try:
        readMat(input_file, output_dir=output_dir, no_regression=args.no_regression)
        logging.info("MAT file processed successfully.")
    except FileNotFoundError:
        logging.error(f"Error: The file {input_file} does not exist.")
        return
    except ValueError as ve:
        logging.error(f"Error: Invalid content in {input_file}: {ve}")
        return
    except Exception as e:
        logging.error(f"Unexpected error while processing {input_file}: {e}")
        return
    logging.info("===================================================")

    # Step 2: Execute saveMsh.py
    try:
        save_msh_command = (
            f"python3 ./src/mat2msh/saveMsh.py "
            f"--mat {aligned_mat_path} "
            f"--output {txt_srf} "
            f"{invert_z_flag}"
        )
        subprocess.run(save_msh_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error executing saveMsh.py: {e}")
        return
    logging.info("===================================================")

    # Step 3: Generate surfaces
    surface_files = [
        f"{txt_srf}/{patient_id}-LVEndo.txt",
        f"{txt_srf}/{patient_id}-LVEpi.txt",
        f"{txt_srf}/{patient_id}-RVEndo.txt",
        f"{txt_srf}/{patient_id}-RVEpi.txt",
    ]

    for surface_file in surface_files:
        try:
            surface_command = (
                f"python3 ./src/mat2msh/makeSurface.py {surface_file} "
                f"--output_dir {ply_srf} --patient_id {patient_id}"
            )
            subprocess.run(surface_command, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error generating surface for {surface_file}: {e}")
            return
    logging.info("===================================================")

    # Step 4: Convert PLY files to STL
    ply_files = [
        f"{ply_srf}/{patient_id}-RVEpi.ply",
        f"{ply_srf}/{patient_id}-RVEndo.ply",
        f"{ply_srf}/{patient_id}-LVEpi.ply",
        f"{ply_srf}/{patient_id}-LVEndo.ply",
    ]

    stl_outputs = [
        f"{stl_srf}/{patient_id}-RVEpi.stl",
        f"{stl_srf}/{patient_id}-RVEndo.stl",
        f"{stl_srf}/{patient_id}-LVEpi.stl",
        f"{stl_srf}/{patient_id}-LVEndo.stl",
    ]

    for ply_file, stl_output in zip(ply_files, stl_outputs):
        if not os.path.exists(ply_file):
            logging.error(f"PLY file not found: {ply_file}")
            return
        try:
            ply_to_stl_command = f"./convertPly2STL/build/bin/PlyToStl {ply_file} {stl_output} 0 1 {args.mesh_relaxation} {args.mesh_iterations} 0"
            subprocess.run(ply_to_stl_command, shell=True, check=True)
            logging.info(f"STL file generated successfully: {stl_output}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error converting {ply_file} to {stl_output}: {e}")
            return

    logging.info(f"STL files generated successfully in: {stl_srf}")
    logging.info("===================================================")

    lv_endo = f"{stl_srf}/{patient_id}-LVEndo.stl"
    rv_endo = f"{stl_srf}/{patient_id}-RVEndo.stl"
    rv_epi  = f"{stl_srf}/{patient_id}-RVEpi.stl"
    lv_epi  = f"{stl_srf}/{patient_id}-LVEpi.stl"

    # Step 4b: Align cardiac STLs to DICOM patient space (enabled by default)
    R_dicom = t_dicom = None
    if not args.no_align_dicom:
        logging.info("===================================================")
        logging.info("Applying DICOM alignment to cardiac STLs...")
        R_dicom, t_dicom, rms = compute_dicom_transform(input_file)
        logging.info(f"  det(R)={np.linalg.det(R_dicom):.4f}  RMSD={rms:.4f} mm")
        for stl_path in [lv_endo, rv_endo, rv_epi, lv_epi]:
            apply_transform_to_file(stl_path, R_dicom, t_dicom)
        logging.info("DICOM transform applied to cardiac STLs.")
        logging.info("===================================================")

    # Step 5: Generate the .msh file using Gmsh
    gmsh = "./scripts/gmsh-2.13.1/bin/gmsh"
    biv_mesh_geo = "./scripts/biv_mesh.geo"

    out_log = f"{log_dir}/{patient_id}_gmsh.log"

    flagScar = False
    try:
        data = loadmat(aligned_mat_path, simplify_cells=True)
        if "setstruct" in data:
            ss = data["setstruct"]

            has_roi = False
            if "Roi" in ss:
                roi_obj = ss["Roi"]
                has_roi = roi_obj is not None and roi_obj != []

            has_gz = False
            try:
                _ = ss["Scar"]["GreyZone"]["map"]
                has_gz = True
            except Exception:
                has_gz = False

            flagScar = bool(has_roi or has_gz)

        if not flagScar:
            logging.info(f"No ROI or GreyZone found in '{aligned_mat_path}'.")
    except Exception as e:
        logging.error(f"Error checking scar in '{aligned_mat_path}': {e}")
        flagScar = False

    # Final mesh always in patientMsh/; intermediate ones in patientMsh/intermediate/
    biv_final_msh = f"{msh_srf}/{patient_id}_biv_final.msh"
    biv_gmsh_msh = f"{msh_int}/{patient_id}_biv_gmsh.msh" if flagScar else biv_final_msh

    if flagScar:
        scar_mode = "greyzone" if has_gz else "roi"
        logging.info("Extracting scars from the .mat file...")
        logging.info("===================================================")
        try:
            read_scar_command = (
                f"python3 ./src/mat2msh/readScar.py {aligned_mat_path} "
                f"--shiftx {output_dir}/endo_shifts_x.txt "
                f"--shifty {output_dir}/endo_shifts_y.txt "
                f"--output_path {output_dir} "
                f"--patient_id {patient_id} "
                f"--mode {scar_mode} "
                f"{invert_z_flag}"
            )
            subprocess.run(read_scar_command, shell=True, check=True)
            logging.info("Scar pipeline executed and fibrosis marked successfully.")
        except subprocess.CalledProcessError as e:
            logging.error(f"Error executing readScar.py: {e}")
            return

        if not args.no_align_dicom and R_dicom is not None:
            scar_stls = glob.glob(f"{scar_srf}/*.stl")
            for stl_path in scar_stls:
                apply_transform_to_file(stl_path, R_dicom, t_dicom)
            logging.info(f"DICOM transform applied to {len(scar_stls)} scar STL(s).")

    logging.info("===================================================")
    logging.info("Generating the mesh with GMSH...")
    logging.info("===================================================")

    os.system('{} -3 {} -merge {} {} {} -clmax {} -clmin {} -o {} 2>&1 {}'.format(
        gmsh, lv_endo, rv_endo, rv_epi, biv_mesh_geo, args.cl_max, args.cl_min, biv_gmsh_msh, out_log))
    logging.info(f"Model generated successfully: {biv_gmsh_msh}")

    # Extract BASE surface from the generated mesh and save as STL
    base_stl = f"{stl_srf}/{patient_id}-BASE.stl"
    extract_base_stl(biv_gmsh_msh, base_stl)

    if flagScar:
        logging.info("===================================================")
        logging.info("Processing scar files...")
        logging.info("===================================================")

        lv_geo = "./scripts/lv_mesh.geo"
        lv_gmsh_msh = f"{msh_int}/{patient_id}_lv_gmsh.msh"
        out_log_lv = f"{log_dir}/{patient_id}_gmsh_lv.log"
        cmd = '{} -3 {} -merge {} {} -o {} > {} 2>&1'.format(
            gmsh, lv_endo, lv_epi, lv_geo, lv_gmsh_msh, out_log_lv
        )

        ret = os.system(cmd)

        if not os.path.exists(lv_gmsh_msh):
            logging.error(f"[FATAL] LV-only mesh not generated. Exit code: {ret}. Log: {out_log_lv}")
            return
        elif ret != 0:
            logging.warning(f"[WARN] Gmsh returned non-zero code ({ret}) but mesh exists. Proceeding.")

        logging.info(f"[OK] LV-only generated: {lv_gmsh_msh}")

        lv_scar_msh = f"{msh_int}/{patient_id}_lv_scar.msh"

        mark_scar_lv_cmd = (
            f"python3 ./src/mat2msh/markFibroseFromMsh.py "
            f"--msh {lv_gmsh_msh} "
            f"--stl_dir {scar_srf} "
            f"--output_path {lv_scar_msh}"
        )
        subprocess.run(mark_scar_lv_cmd, shell=True, check=True)
        logging.info(f"[OK] Fibrosis marked on LV-only: {lv_scar_msh}")

        core_ply = os.path.join(scar_ply_smooth, f"{patient_id}_core_from_lv.ply")
        gz_ply   = os.path.join(scar_ply_smooth, f"{patient_id}_gz_from_lv.ply")

        msh_tag_to_ply(lv_scar_msh, tag=2, ply_path=core_ply)
        msh_tag_to_ply(lv_scar_msh, tag=3, ply_path=gz_ply)

        core_stl = os.path.join(scar_stl_smooth, f"{patient_id}_core_smooth.stl")
        gz_stl   = os.path.join(scar_stl_smooth, f"{patient_id}_gz_smooth.stl")

        logging.info("Smoothing fibrosis surfaces...")
        subprocess.run(
            f"./convertPly2STL/build/bin/PlyToStl {core_ply} {core_stl} 1 1 {args.relaxation} {args.iterations} 2",
            shell=True, check=True
        )
        subprocess.run(
            f"./convertPly2STL/build/bin/PlyToStl {gz_ply} {gz_stl} 1 1 {args.relaxation} {args.iterations} 2",
            shell=True, check=True
        )

        cmd2 = (
            f"python3 ./src/mat2msh/markFibroseFromMsh.py "
            f"--msh {biv_gmsh_msh} "
            f"--stl_dir {scar_stl_smooth} "
            f"--output_path {biv_final_msh}"
        )
        subprocess.run(cmd2, shell=True, check=True)
        logging.info(f"[OK] BIV final mesh with fibrosis: {biv_final_msh}")

    logging.info("===================================================")
    logging.info("Converting msh to alg format...")
    logging.info("===================================================")

    marked_msh_path = biv_final_msh

    # Extract all surfaces (BASE, LV, RV, EPI) from the final mesh as STL files
    surfaces_output_dir = f"{output_dir}/surfaces"
    logging.info("===================================================")
    logging.info("Extracting surfaces from final mesh...")
    extract_all_surfaces_stl(marked_msh_path, surfaces_output_dir, patient_id)
    logging.info(f"Surfaces saved in: {surfaces_output_dir}")
    logging.info("===================================================")

    try:
        hexa_log = f"{log_dir}/{patient_id}_hexa.log"
        no_hexa_flag = "--no_hexa" if args.no_alg else ""
        msh2alg_command = (
            f"PYTHONPATH=. python3 ./src/msh2alg/msh2alg.py "
            f"-i {marked_msh_path} "
            f"-o {mesh_output_base} "
            f"--carp {carp_output} "
            f"--alg {alg_output} "
            f"-r {args.resolution} "
            f"--dx {args.dx} --dy {args.dy} --dz {args.dz} "
            f"--alpha_endo_lv {args.alpha_endo_lv} --alpha_epi_lv {args.alpha_epi_lv} "
            f"--beta_endo_lv {args.beta_endo_lv} --beta_epi_lv {args.beta_epi_lv} "
            f"--alpha_endo_sept {args.alpha_endo_sept} --alpha_epi_sept {args.alpha_epi_sept} "
            f"--beta_endo_sept {args.beta_endo_sept} --beta_epi_sept {args.beta_epi_sept} "
            f"--alpha_endo_rv {args.alpha_endo_rv} --alpha_epi_rv {args.alpha_epi_rv} "
            f"--beta_endo_rv {args.beta_endo_rv} --beta_epi_rv {args.beta_epi_rv} "
            f"--log_file {hexa_log} "
            f"{no_hexa_flag}"
        )
        subprocess.run(msh2alg_command, shell=True, check=True)
        if args.no_alg:
            logging.info("FEniCS/VTU done. ALG conversion skipped (--no_alg).")
        else:
            logging.info("===================================================")
            logging.info("Mesh successfully converted to ALG format.")
            logging.info("===================================================")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error converting msh to alg: {e}")
        return

    logging.info("========================================================================================")
    logging.info("                         Finished processing the patient data.")
    logging.info("========================================================================================")
    # Clean up intermediate files
    os.remove(f"{output_dir}/endo_shifts_x.txt")
    os.remove(f"{output_dir}/endo_shifts_y.txt")
    os.remove(f"{output_dir}/epi_shifts_x.txt")
    os.remove(f"{output_dir}/epi_shifts_y.txt")
    os.remove(f"{output_dir}/{patient_id}.mat")
    shutil.rmtree(f"{output_dir}/slices", ignore_errors=True)
    shutil.rmtree(f"{output_dir}/rois_extruded", ignore_errors=True)

    # Remove directories created during the process
    # If needed, uncomment the following lines

    #shutil.rmtree(txt_srf, ignore_errors=True)
    #shutil.rmtree(stl_srf, ignore_errors=True)
    #shutil.rmtree(f"{output_dir}/scarPly_raw", ignore_errors=True)
    #shutil.rmtree(f"{output_dir}/scarSTL_raw", ignore_errors=True)
    #shutil.rmtree(f"{output_dir}/plyFiles", ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute pipeline for processing a .mat file.")
    parser.add_argument("-i", "--input_file", required=True, help="Full path to the input .mat file")
    parser.add_argument('-r', '--resolution', type=int, default=1000, help='Discretization resolution for the mesh')

    parser.add_argument('-dx', type=float, default=0.5, help='dx')
    parser.add_argument('-dy', type=float, default=0.5, help='dy')
    parser.add_argument('-dz', type=float, default=0.5, help='dz')

    parser.add_argument('--relaxation', type=float, default=0.05, help='Relaxation factor for scar/fibrosis surface smoothing')
    parser.add_argument('--iterations', type=int, default=200, help='Number of iterations for scar/fibrosis surface smoothing')

    parser.add_argument('--mesh_relaxation', type=float, default=0.02, help='Relaxation factor for cardiac surface smoothing (RVEndo, LVEpi, LVEndo)')
    parser.add_argument('--mesh_iterations', type=int, default=40, help='Number of iterations for cardiac surface smoothing (RVEndo, LVEpi, LVEndo)')

    parser.add_argument('--cl_max', type=float, default=2.0, help='Gmsh CharacteristicLengthMax: maximum element size in the mesh')
    parser.add_argument('--cl_min', type=float, default=1.0, help='Gmsh CharacteristicLengthMin: minimum element size in the mesh')

    parser.add_argument('--no_invert_z', action='store_true', help='Disable Z-axis flip (by default Z is mirrored at the center of [z_min, z_max]).')
    parser.add_argument('--no_regression', action='store_true', help='Disable slice alignment by linear regression of barycenters in readMat. Preserves original DICOM pixel coordinates.')
    parser.add_argument('--no_alg', action='store_true', help='Skip ALG/CARP conversion step (msh2alg). Useful for fast mesh inspection.')
    parser.add_argument('--no_align_dicom', action='store_true', help='Disable DICOM alignment (by default the mesh is aligned to patient DICOM space using ImagePosition/Orientation from the .mat).')
    parser.add_argument('--alpha_endo_lv', type=float, default=30, help='Fiber angle on the LV endocardium')
    parser.add_argument('--alpha_epi_lv', type=float, default=-30, help='Fiber angle on the LV epicardium')
    parser.add_argument('--beta_endo_lv', type=float, default=0, help='Sheet angle on the LV endocardium')
    parser.add_argument('--beta_epi_lv', type=float, default=0, help='Sheet angle on the LV epicardium')

    parser.add_argument('--alpha_endo_sept', type=float, default=60, help='Fiber angle on the Septum endocardium')
    parser.add_argument('--alpha_epi_sept', type=float, default=-60, help='Fiber angle on the Septum epicardium')
    parser.add_argument('--beta_endo_sept', type=float, default=0, help='Sheet angle on the Septum endocardium')
    parser.add_argument('--beta_epi_sept', type=float, default=0, help='Sheet angle on the Septum epicardium')

    parser.add_argument('--alpha_endo_rv', type=float, default=80, help='Fiber angle on the RV endocardium')
    parser.add_argument('--alpha_epi_rv', type=float, default=-80, help='Fiber angle on the RV epicardium')
    parser.add_argument('--beta_endo_rv', type=float, default=0, help='Sheet angle on the RV endocardium')
    parser.add_argument('--beta_epi_rv', type=float, default=0, help='Sheet angle on the RV epicardium')
    args = parser.parse_args()

    execute_commands(args.input_file)