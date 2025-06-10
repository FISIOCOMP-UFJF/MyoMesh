import sys, os, glob, subprocess, argparse
from collections import namedtuple, defaultdict

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

ROIEntry = namedtuple('ROIEntry', ['name', 'z', 'points'])


# Reading the .mat file and extracting ROIs
def readScar(mat_filename):
    """
    Reads the .mat file and returns a flat list of ROIEntry.
    Each ROIEntry contains: (e.g., "ROI-1")
      - name: ROI name (e.g., "ROI-1")
      - z: slice index where the ROI was annotated
      - points: list of (x,y) tuples for the ROI points
    Explanation:
    - "setstruct" is the MATLAB structure containing metadata and ROIs.
    - For each ROI, X, Y, and Z arrays and names for each element are extracted.
    - For each sub-slice (index i), the correct name is associated and the points
      are packaged into Python.
    """

    print(f"Reading ROIs from: {mat_filename}")
    data = loadmat(mat_filename)
    rois = data['setstruct'][0][0]['Roi']

    entries = []
    for idx, roi in enumerate(rois):
        # Extracts the list of names for each sub-slice
        raw_names = roi['Name'].flatten()
        slice_names = []
        for element in raw_names:
            # Converts numpy array to string
            if isinstance(element, np.ndarray):
                val = element.flat[0]
            else:
                val = element
            slice_names.append(str(val).strip())
        # Coordinate arrays
        X = roi['X']; Y = roi['Y']; Z = roi['Z']
        # For each sub-slice, package an ROIEntry
        for i in range(len(Z)):
            z_val = int(np.atleast_1d(Z[i]).flat[0])
            x_arr = np.atleast_1d(X[i]).flatten()
            y_arr = np.atleast_1d(Y[i]).flatten()
            if x_arr.size == 0 or y_arr.size == 0:
                continue  # no points in this sub-slice
            name = slice_names[i] if i < len(slice_names) else slice_names[0]
            pts = list(zip(x_arr, y_arr))
            entries.append(ROIEntry(name, z_val, pts))
    return entries


# Grouping of ROIs by slice
def group_by_slice(entries):
    """
    Receives a list of ROIEntry and returns:
      { z: { roi_name: [ (x,y), ... ], ... }, ... }
    """
    fatias = defaultdict(lambda: defaultdict(list))
    for e in entries:
        fatias[e.z][e.name].extend(e.points)
    return fatias


# 2D visualization of the slices
def plot_slices(fatias):
    """
    For verification, plots each slice showing:
      - Points of each ROI in different colors
      - Name and centroid marked
    """
    for z, roi_map in sorted(fatias.items()):
        plt.figure(figsize=(6,6))
        for name, pts in roi_map.items():
            arr = np.array(pts)
            plt.scatter(arr[:,0], arr[:,1], label=name, s=20)
            cen = arr.mean(axis=0)
            plt.text(cen[0], cen[1], name,
                     ha='center', va='center', fontsize=8,
                     bbox=dict(boxstyle='round,pad=0.2', alpha=0.5))
        plt.title(f"Slice Z={z}")
        plt.gca().invert_yaxis()
        plt.legend(fontsize=7)
        plt.xlabel('X'); plt.ylabel('Y')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Alignment and saving slices in .txt
def save_fatias_to_txt(fatias, shifts_x_file, shifts_y_file, output_dir):
    """
    Applies X/Y shifts for each slice and saves in "slices/".
    Files: slice_<z>.txt containing lines "x_aligned y_aligned z".
    """
    os.makedirs(output_dir, exist_ok=True)
    shifts_x = np.loadtxt(shifts_x_file)
    shifts_y = np.loadtxt(shifts_y_file)
    for z, roi_map in sorted(fatias.items()):
        # Selects the appropriate shift or zero if out of range
        sx = shifts_x[z] if 0 <= z < len(shifts_x) else 0
        sy = shifts_y[z] if 0 <= z < len(shifts_y) else 0
        fname = os.path.join(output_dir, f"slice_{z}.txt")
        with open(fname, 'w') as f:
            for pts in roi_map.values():
                for x, y in pts:
                    f.write(f"{x - sx} {y - sy} {z}\n")
        print(f"Saved slice {z} to {fname}")


# Saves separated ROIs in .txt
def save_rois_extruded_to_txt(fatias, mat_filename, output_dir, num_layers=1):

    data = loadmat(mat_filename)
    ss = data['setstruct']
    slice_thickness = float(ss['SliceThickness'][0][0][0][0])
    gap             = float(ss['SliceGap'][0][0][0][0])
    resolution_x    = float(ss['ResolutionX'][0][0][0][0])
    resolution_y    = float(ss['ResolutionY'][0][0][0][0])
    dz = slice_thickness + gap

    os.makedirs(output_dir, exist_ok=True)

    for z, roi_map in sorted(fatias.items()):
        for roi_name, points in roi_map.items():
            safe_name = roi_name.replace(" ", "_").replace("/", "_")
            fname = os.path.join(output_dir, f"roi_{safe_name}_z{z}.txt")
            with open(fname, 'w') as f:
                z_base = z * dz
                z_top  = z_base + dz

                # Generates N layers between z_base and z_top
                for layer in range(num_layers + 1):
                    alpha = layer / num_layers
                    # Interpolates between z_base and z_top
                    z_interp = z_base * (1 - alpha) + z_top * alpha
                    for x, y in points:
                        x_out = x * resolution_x
                        y_out = y * resolution_y
                        f.write(f"{x_out:.6f} {y_out:.6f} {z_interp:.6f}\n")
            print(f"Saved extruded ROI '{roi_name}' (slice {z}) to: {fname}")


# Generation of surfaces (.ply) and STL
def generate_surfaces_and_stl(patient_id, rois_dir, ply_dir, stl_dir):
    """
    For each file roi_<name>_z<z>.txt in rois_extruded/:
      1) calls makeSurface.py to generate .ply
      2) converts .ply to .stl using PlyToStl
    """
    # Prepares output directories
    os.makedirs(ply_dir, exist_ok=True)
    os.makedirs(stl_dir, exist_ok=True)

    txts = sorted(glob.glob(os.path.join(rois_dir, "roi_*.txt")))
    for txt in txts:
        base = os.path.splitext(os.path.basename(txt))[0]
        ply = os.path.join(ply_dir, f"{base}.ply")
        stl = os.path.join(stl_dir, f"{base}.stl")

        # 1) Generates the PLY with mandatory parameters
        surface_command = (
            f"python3 src/mat2msh/makeSurface.py {txt} "
            f"--output_dir {ply_dir} --patient_id {patient_id} --cover-both-ends"
        )
        try:
            subprocess.run(surface_command, shell=True, check=True)
            print(f"Surface for {txt} generated at {ply}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating PLY for {txt}: {e}")
            continue

        # 2) Converts PLY -> STL
        if os.path.exists(ply):
            try:
                subprocess.run(
                    f"./convertPly2STL/build/bin/PlyToStl {ply} {stl} 1",
                    shell=True, check=True
                )
                print(f"STL created: {stl}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {ply} to STL: {e}")
        else:
            print(f"PLY file not found for {txt}, skipping STL conversion.")


# MAIN: full execution
def main():
    parser = argparse.ArgumentParser(description="Full scar pipeline")
    parser.add_argument('matfile', help='Path to .mat file')
    parser.add_argument('--shiftx', default='endo_shifts_x.txt')
    parser.add_argument('--shifty', default='endo_shifts_y.txt')
    parser.add_argument('--output_path', required=True,
                    help='Base path to save output folders like slices, rois_extruded, plyFiles, scarFiles')
    parser.add_argument('--patient_id', required=True,
                    help='Patient identifier for naming and organization')
    args = parser.parse_args()
    
    # 1-3: reading, grouping, and plotting
    entries = readScar(args.matfile)
    slices = group_by_slice(entries)
    plot_slices(slices)
    # 4: save aligned slices
    slices_dir = os.path.join(args.output_path, "slices")
    save_fatias_to_txt(slices, args.shiftx, args.shifty, slices_dir)

    print("===================================================")
    print("Construction extruded and saved slices scar files")
    print("===================================================")
    # 5) Simple extrusion of each ROI
    rois_dir = os.path.join(args.output_path, "rois_extruded")
    print(f"Saving extruded ROIs to: {rois_dir}")
    
    save_rois_extruded_to_txt(slices, args.matfile, output_dir=rois_dir)
    print("===================================================")
    print("Generate surfaces and STL files")
    print("===================================================")

    # 6) Generation of surfaces and STL from the extrusions
    ply_dir = os.path.join(args.output_path, "scarPly")
    stl_dir = os.path.join(args.output_path, "scarSTL")
    generate_surfaces_and_stl(args.patient_id, rois_dir, ply_dir, stl_dir)


if __name__ == '__main__':
    main()
