import os
from datetime import datetime
import subprocess
from src.mat2msh.readMat import readMat
import argparse
import shutil
from scipy.io import loadmat

def execute_commands(input_file):
   
    filename = os.path.basename(input_file)
    patient_id = os.path.splitext(filename)[0]
   
    print("========================================================================================")
    print("Starting the pipeline for patient data processing.")
    print("========================================================================================")
    # Create the output folder with the current date
    date_str = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = f"./output/{date_str}/{patient_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Paths for surfaces and intermediate files
    stl_srf = f"{output_dir}/stlFiles"
    msh_srf = f"{output_dir}/mshFiles"
    scar_srf = f"{output_dir}/scarSTL"
    txt_srf = f"{output_dir}/txtFiles"
    ply_srf = f"{output_dir}/plyFiles"
    
    if os.path.exists(stl_srf):
        shutil.rmtree(stl_srf)
    os.makedirs(stl_srf, exist_ok=True)

    if os.path.exists(msh_srf):
        shutil.rmtree(msh_srf)
    os.makedirs(msh_srf, exist_ok=True)

    if os.path.exists(scar_srf):
        shutil.rmtree(scar_srf)
    os.makedirs(scar_srf, exist_ok=True)

    if os.path.exists(txt_srf):
        shutil.rmtree(txt_srf)
    os.makedirs(txt_srf, exist_ok=True)

    if os.path.exists(ply_srf):
        shutil.rmtree(ply_srf)
    os.makedirs(ply_srf, exist_ok=True)
    
    # Step 1: Process the .mat file
    print(f"Processing the file: {input_file}")
    try:
        # Attempt to process the .mat file
        readMat(input_file, output_dir=output_dir)

        print("MAT file processed successfully.")
    except FileNotFoundError:
        # Handle the case where the file does not exist
        print(f"Error: The file {input_file} does not exist.")
        return
    except ValueError as ve:
        # Handle cases where the file content is invalid
        print(f"Error: Invalid content in {input_file}: {ve}")
        return
    except Exception as e:
        # Handle any other unforeseen errors
        print(f"Unexpected error while processing {input_file}: {e}")
        return
    print("===================================================")

    # Step 2: Execute saveMsh.py
    try:
        save_msh_command = f"python3 ./src/mat2msh/saveMsh.py --mat {input_file} --output {txt_srf}"
        subprocess.run(save_msh_command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing saveMsh.py: {e}")
        return
    print("===================================================")

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
            print(f"Error generating surface for {surface_file}: {e}")
            return
    print("===================================================")

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
            print(f"Error: PLY file {ply_file} not found.")
            return
        try:
            ply_to_stl_command = f"./convertPly2STL/build/bin/PlyToStl {ply_file} {stl_output} 0"
            subprocess.run(ply_to_stl_command, shell=True, check=True)
            print(f"STL file generated successfully: {stl_output}")
        except subprocess.CalledProcessError as e:
            print(f"Error converting {ply_file} to {stl_output}: {e}")
            return

    # Final Processing
    print(f"STL files generated successfully in: {stl_srf}")
    print("===================================================")
    # Step 5: Generate the `.msh` file using Gmsh

    # Gmsh and generation scripts
    gmsh = "./scripts/gmsh-2.13.1/bin/gmsh"
    biv_mesh_geo = "./scripts/biv_mesh.geo"

    lv_endo = f"{stl_srf}/{patient_id}-LVEndo.stl"
    rv_endo = f"{stl_srf}/{patient_id}-RVEndo.stl"
    rv_epi = f"{stl_srf}/{patient_id}-RVEpi.stl"

    msh_heart = f"{msh_srf}/{patient_id}_model.msh"
    msh = f"{msh_srf}/{patient_id}.msh"
    out_log = f"{msh_srf}/{patient_id}.log"

    flagScar = False
    try:
        data = loadmat(input_file, struct_as_record=False, squeeze_me=True)
        # Checks if 'setstruct' and 'Roi' exist and if 'Roi' is not empty
        if 'setstruct' in data and hasattr(data['setstruct'], 'Roi') and data['setstruct'].Roi.size > 0:
            flagScar = True
        else:
            print(f"No 'Roi' data found in '{input_file}' or 'setstruct' is missing/empty.")
            flagScar = False
    except Exception as e:
        print(f"No ROIs present in '{input_file}'")
        flagScar = False
    
    if flagScar:
        print("Extracting scars from the .mat file...")
        print("===================================================")
        # Step 6: Execute readScar.py
        try:
            aligned_mat_path = f"{output_dir}/aligned_patient.mat"
            print(f"Aligned MAT file path: {aligned_mat_path}")
            msh_path = f"{msh_srf}/{patient_id}.msh"
            output_marked = f"{msh_srf}/{patient_id}_marked.msh"

            read_scar_command = (
                f"python3 ./src/mat2msh/readScar.py {input_file} "
                f"--shiftx {output_dir}/endo_shifts_x.txt "
                f"--shifty {output_dir}/endo_shifts_y.txt "
                f"--output_path {output_dir} "
                f"--patient_id {patient_id}"
            )

            subprocess.run(read_scar_command, shell=True, check=True)
            print("Scar pipeline executed and fibrosis marked successfully.")

        except subprocess.CalledProcessError as e:
            print(f"Error executing readScar.py: {e}")
            return
    
    print("===================================================")
    print("Generating the mesh with GMSH...")
    print("===================================================")
    # Command with os.system
    try:
        os.system('{} -3 {} -merge {} {} {} -o {} 2>&1 {}'.format(
            gmsh, lv_endo, rv_endo, rv_epi, biv_mesh_geo, msh, out_log))
        print(f"Model generated successfully: {msh_heart}")
    except Exception as e:
        print(f"Error generating model: {e}")
        return
    
    if flagScar:
        print("")
        print("===================================================")
        print("Marking the mesh with the scar files...")
        print("===================================================")
        # Step 7: Execute mark_fibrosis_script.py
        try:
            msh_path = f"{msh_srf}/{patient_id}.msh"
            output_marked = f"{msh_srf}/{patient_id}_marked.msh"
            mark_scar_command = (
                f"python3 ./src/mat2msh/markFibroseFromMsh.py "
                f"--msh {msh_path} "
                f"--stl_dir {scar_srf} "
                f"--output_path {output_marked}"
            )
            subprocess.run(mark_scar_command, shell=True, check=True)
            print("===================================================")
            print("Fibrosis successfully marked in the mesh.")
            print("===================================================")

        except subprocess.CalledProcessError as e:
            print(f"Error executing mark_fibrosis_script.py: {e}")
            return

    print("===================================================")
    print("Converting msh to alg format...")
    print("===================================================")
    
    if flagScar:
        marked_msh_path = f"{msh_srf}/{patient_id}_marked.msh"
    else:
        marked_msh_path = f"{msh_srf}/{patient_id}.msh"

    mesh_output_base = f"{output_dir}/conversionFiles/{patient_id}"
    os.makedirs(os.path.dirname(mesh_output_base), exist_ok=True)

    try:
        msh2alg_command = (
            f"PYTHONPATH=. python3 ./src/msh2alg/msh2alg.py "
            f"-i {marked_msh_path} "
            f"-o {mesh_output_base} "
            f"-r {args.resolution} "              
            f"--dx {args.dx} --dy {args.dy} --dz {args.dz} "
            f"--alpha_endo_lv {args.alpha_endo_lv} --alpha_epi_lv {args.alpha_epi_lv} "
            f"--beta_endo_lv {args.beta_endo_lv} --beta_epi_lv {args.beta_epi_lv} "
            f"--alpha_endo_sept {args.alpha_endo_sept} --alpha_epi_sept {args.alpha_epi_sept} "
            f"--beta_endo_sept {args.beta_endo_sept} --beta_epi_sept {args.beta_epi_sept} "
            f"--alpha_endo_rv {args.alpha_endo_rv} --alpha_epi_rv {args.alpha_epi_rv} "
            f"--beta_endo_rv {args.beta_endo_rv} --beta_epi_rv {args.beta_epi_rv}"
        )

        subprocess.run(msh2alg_command, shell=True, check=True)
        print("===================================================")
        print("Mesh successfully converted to ALG format.")
        print("===================================================")

    except subprocess.CalledProcessError as e:
        print(f"Error converting msh to alg: {e}")
        return

    print("========================================================================================")
    print("                         Finished processing the patient data.")
    print("========================================================================================")
    # Clean up intermediate files
    os.remove(f"{output_dir}/endo_shifts_x.txt")
    os.remove(f"{output_dir}/endo_shifts_y.txt")
    os.remove(f"{output_dir}/epi_shifts_x.txt")
    os.remove(f"{output_dir}/epi_shifts_y.txt")
    shutil.rmtree(txt_srf, ignore_errors=True)
    shutil.rmtree(stl_srf, ignore_errors=True)
    shutil.rmtree(f"{output_dir}/scarPly", ignore_errors=True)
    shutil.rmtree(f"{output_dir}/slices", ignore_errors=True)
    shutil.rmtree(f"{output_dir}/scarSTL", ignore_errors=True)
    shutil.rmtree(f"{output_dir}/rois_extruded", ignore_errors=True)
    shutil.rmtree(f"{output_dir}/plyFiles", ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute pipeline for processing a .mat file.")
    parser.add_argument("-i", "--input_file", required=True, help="Full path to the input .mat file")
    parser.add_argument('-r', '--resolution', type=int, default=1000, help='Discretization resolution for the mesh')

    parser.add_argument('-dx', type=float, default=0.5, help='dx')
    parser.add_argument('-dy', type=float, default=0.5, help='dy')
    parser.add_argument('-dz', type=float, default=0.5, help='dz')

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
